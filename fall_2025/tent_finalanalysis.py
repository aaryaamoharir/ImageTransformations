import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import copy
from scipy.stats import entropy as scipy_entropy

# ======= CIFAR-C DATASET CLASS =======
class CIFARC_Dataset(Dataset):
    def __init__(self, root, corruption, severity, transform=None):
        self.transform = transform
        corruption_file = os.path.join(root, f"{corruption}.npy")
        labels_file = os.path.join(root, "labels.npy")
        all_images = np.load(corruption_file)
        all_labels = np.load(labels_file)
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        self.data = all_images[start_idx:end_idx]
        self.labels = all_labels[start_idx:end_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ======= CONFIG =======
USE_DATASET = "cifar100c"
BATCH_SIZE = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if USE_DATASET == "cifar10c":
    NUM_CLASSES = 10
    cifarc_root = "/home/diversity_project/aaryaa/attacks/Cifar-10/cifar_npy_files"
    model_name = "cifar10_resnet56"
else:
    NUM_CLASSES = 100
    cifarc_root = "/home/diversity_project/aaryaa/attacks/Cifar-100/CIFAR-100-C"
    model_name = "cifar100_resnet56"

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

corruptions = [
    'defocus_blur', 'glass_blur', 'motion_blur',
    'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

ENTROPY_THRESHOLD = 1.16

# ======= MODEL =======
print("Loading model...")
model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    model_name,
    pretrained=True
).to(device)


# ======= HELPERS =======
def compute_shannon_entropy(image_tensor):
    img = image_tensor.clone().cpu()
    for c in range(3):
        img[c] = img[c] * CIFAR_STD[c] + CIFAR_MEAN[c]
    img = torch.clamp(img, 0, 1)
    img_np = img.numpy().flatten()
    hist, _ = np.histogram(img_np, bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0]
    return scipy_entropy(hist, base=2)

def configure_model_for_tent(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.requires_grad_(True)
            module.reset_running_stats()
            module.momentum = 0.1
    return model

def collect_bn_params(model):
    params = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for param in module.parameters():
                if param.requires_grad:
                    params.append(param)
    return params

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()
def tent_forward_and_adapt(model, x, optimizer):
    outputs = model(x)
    loss = softmax_entropy(outputs).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(collect_bn_params(model), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    return outputs


# ======= FLIP ANALYSIS FUNCTION =======
def compute_flip_analysis(model, dataloader, lr=0.001, entropy_threshold=ENTROPY_THRESHOLD):
    """
    For every image, pairs the original (baseline) prediction with the TENT prediction
    and records which of the 4 possible transitions occurred:

      original=correct, tent=correct  →  "stayed_correct"   (good, no change)
      original=correct, tent=incorrect→  "flipped_wrong"     (BAD — TENT degraded it)
      original=incorrect, tent=correct →  "flipped_right"    (GOOD — TENT fixed it)
      original=incorrect, tent=incorrect→ "stayed_wrong"     (no change, still wrong)

    LOW entropy images (<= threshold):
      - Only passed through the baseline model. TENT is never applied.
      - Reported separately with no flip breakdown (nothing to flip).

    HIGH entropy images (> threshold):
      - Baseline prediction recorded first (original).
      - TENT applied via per-batch reset on only the high-entropy sub-batch.
      - TENT prediction recorded (tent).
      - All 4 flip transitions tracked per image.

    Returns:
      {
        'low':  { 'correct': int, 'incorrect': int, 'total': int },
        'high': {
            'stayed_correct': int,   # original correct  → tent correct   (want HIGH)
            'flipped_wrong':  int,   # original correct  → tent incorrect (want LOW)
            'flipped_right':  int,   # original incorrect→ tent correct   (want HIGH)
            'stayed_wrong':   int,   # original incorrect→ tent incorrect (want LOW)
            'total': int
        }
      }
    """
    model_state = copy.deepcopy(model.state_dict())
    model_eval  = copy.deepcopy(model)
    model_eval.eval()

    results = {
        'low': {'correct': 0, 'incorrect': 0, 'total': 0},
        'high': {
            'stayed_correct': 0,
            'flipped_wrong':  0,
            'flipped_right':  0,
            'stayed_wrong':   0,
            'total': 0,
        }
    }

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        labels_cpu = labels.cpu()

        # ---- Entropy split ----
        entropies = [compute_shannon_entropy(images[i]) for i in range(batch_size)]
        high_mask = torch.tensor([e > entropy_threshold for e in entropies])  # bool, CPU
        low_mask  = ~high_mask

        # ================================================================
        # LOW ENTROPY — baseline only, never touched by TENT
        # ================================================================
        if low_mask.any():
            low_imgs   = images[low_mask.to(device)]
            low_labels = labels_cpu[low_mask]
            with torch.no_grad():
                low_preds = model_eval(low_imgs).argmax(dim=1).cpu()
            correct = (low_preds == low_labels)
            results['low']['correct']   += correct.sum().item()
            results['low']['incorrect'] += (~correct).sum().item()
            results['low']['total']     += low_labels.size(0)

        # ================================================================
        # HIGH ENTROPY — get original prediction, then apply TENT, compare
        # ================================================================
        if high_mask.any():
            high_imgs   = images[high_mask.to(device)]
            high_labels = labels_cpu[high_mask]

            # --- Original prediction (frozen baseline, no adaptation) ---
            with torch.no_grad():
                original_preds = model_eval(high_imgs).argmax(dim=1).cpu()

            original_correct = (original_preds == high_labels)  # bool tensor

            # --- TENT prediction (per-batch reset, adapt on high-entropy only) ---
            model.load_state_dict(model_state)
            model = configure_model_for_tent(model)
            optimizer = torch.optim.Adam(collect_bn_params(model), lr=lr)
            tent_outputs = tent_forward_and_adapt(model, high_imgs, optimizer)
            tent_preds   = tent_outputs.argmax(dim=1).detach().cpu()

            tent_correct = (tent_preds == high_labels)  # bool tensor

            # --- Record all 4 transitions per image ---
            # original correct   & tent correct   → stayed_correct
            # original correct   & tent incorrect → flipped_wrong   (BAD)
            # original incorrect & tent correct   → flipped_right   (GOOD)
            # original incorrect & tent incorrect → stayed_wrong
            results['high']['stayed_correct'] += ( original_correct &  tent_correct).sum().item()
            results['high']['flipped_wrong']  += ( original_correct & ~tent_correct).sum().item()
            results['high']['flipped_right']  += (~original_correct &  tent_correct).sum().item()
            results['high']['stayed_wrong']   += (~original_correct & ~tent_correct).sum().item()
            results['high']['total']          += high_labels.size(0)

    model.load_state_dict(model_state)
    return results


# ======= PRETTY PRINT =======
def print_flip_results(results, label=""):
    if label:
        print(f"  [{label}]")

    # LOW group
    low   = results['low']
    total = low['total']
    if total > 0:
        acc = 100 * low['correct'] / total
        print(f"  LOW entropy (<= {ENTROPY_THRESHOLD})  [{total} images]  — baseline only, TENT never applied")
        print(f"    Correct:   {low['correct']:>6}  ({acc:.1f}%)")
        print(f"    Incorrect: {low['incorrect']:>6}  ({100-acc:.1f}%)")
    else:
        print(f"  LOW entropy: 0 images")

    print()

    # HIGH group — the flip matrix
    high  = results['high']
    total = high['total']
    if total > 0:
        orig_correct   = high['stayed_correct'] + high['flipped_wrong']
        orig_incorrect = high['flipped_right']  + high['stayed_wrong']
        orig_acc  = 100 * orig_correct   / total
        tent_acc  = 100 * (high['stayed_correct'] + high['flipped_right']) / total
        delta     = tent_acc - orig_acc

        print(f"  HIGH entropy (> {ENTROPY_THRESHOLD})  [{total} images]  — TENT applied")
        print(f"  {'':40s}  {'TENT correct':>14}  {'TENT incorrect':>14}")
        print(f"  {'Original correct   ' + f'({orig_correct} imgs, {orig_acc:.1f}%)':40s}  "
              f"{'stayed_correct':>6}: {high['stayed_correct']:>6}  "
              f"{'flipped_wrong':>6}: {high['flipped_wrong']:>6}  ← want LOW")
        print(f"  {'Original incorrect ' + f'({orig_incorrect} imgs, {100-orig_acc:.1f}%)':40s}  "
              f"{'flipped_right':>6}: {high['flipped_right']:>6}  "
              f"{'stayed_wrong':>6}: {high['stayed_wrong']:>6}  ← want flipped_right HIGH")
        print()
        print(f"    Original accuracy:  {orig_acc:.2f}%  ({orig_correct}/{total})")
        print(f"    TENT accuracy:      {tent_acc:.2f}%  ({high['stayed_correct'] + high['flipped_right']}/{total})")
        print(f"    Delta:              {delta:+.2f}%")
        print()
        pct_degraded = 100 * high['flipped_wrong']  / max(orig_correct, 1)
        pct_improved = 100 * high['flipped_right']  / max(orig_incorrect, 1)
        print(f"    Of originally CORRECT images:   {high['flipped_wrong']:>5} degraded  "
              f"({pct_degraded:.1f}% of {orig_correct})  ← want LOW")
        print(f"    Of originally INCORRECT images: {high['flipped_right']:>5} recovered "
              f"({pct_improved:.1f}% of {orig_incorrect})  ← want HIGH")
    else:
        print(f"  HIGH entropy: 0 images")


# ======= MAIN LOOP =======
print("\n" + "="*80)
print("TENT FLIP ANALYSIS")
print(f"Threshold: {ENTROPY_THRESHOLD}  |  LOW = baseline only  |  HIGH = TENT applied")
print("Goal: flipped_wrong LOW, flipped_right HIGH")
print("="*80)

# Global aggregators
global_results = {
    'low':  {'correct': 0, 'incorrect': 0, 'total': 0},
    'high': {'stayed_correct': 0, 'flipped_wrong': 0,
             'flipped_right': 0,  'stayed_wrong': 0, 'total': 0},
}

for corr in corruptions:
    print(f"\n{'='*60}")
    print(f"Corruption: {corr}")
    print(f"{'='*60}")

    corr_results = {
        'low':  {'correct': 0, 'incorrect': 0, 'total': 0},
        'high': {'stayed_correct': 0, 'flipped_wrong': 0,
                 'flipped_right': 0,  'stayed_wrong': 0, 'total': 0},
    }

    for severity in range(1, 6):
        print(f"\n  Severity {severity}:")
        testset = CIFARC_Dataset(
            root=cifarc_root, corruption=corr, severity=severity, transform=transform
        )
        testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

        res = compute_flip_analysis(model, testloader, lr=0.001, entropy_threshold=ENTROPY_THRESHOLD)
        print_flip_results(res)

        for key in corr_results['low']:
            corr_results['low'][key]   += res['low'][key]
            global_results['low'][key] += res['low'][key]
        for key in corr_results['high']:
            corr_results['high'][key]   += res['high'][key]
            global_results['high'][key] += res['high'][key]

    print(f"\n  --- {corr} TOTAL (all severities) ---")
    print_flip_results(corr_results)


# ======= GLOBAL SUMMARY =======
print("\n" + "="*80)
print("GLOBAL SUMMARY — all corruptions, all severities")
print("="*80)
print_flip_results(global_results)

# Final headline numbers
h = global_results['high']
if h['total'] > 0:
    print("\n" + "="*80)
    print("HEADLINE NUMBERS")
    print("="*80)
    orig_correct   = h['stayed_correct'] + h['flipped_wrong']
    orig_incorrect = h['flipped_right']  + h['stayed_wrong']
    print(f"  flipped_wrong  (correct → incorrect after TENT): {h['flipped_wrong']:>7}  "
          f"({100*h['flipped_wrong']/max(orig_correct,1):.2f}% of originally correct)")
    print(f"  flipped_right  (incorrect → correct after TENT): {h['flipped_right']:>7}  "
          f"({100*h['flipped_right']/max(orig_incorrect,1):.2f}% of originally incorrect)")
    ratio = h['flipped_right'] / max(h['flipped_wrong'], 1)
    print(f"\n  Improvement ratio (flipped_right / flipped_wrong): {ratio:.2f}x")
    print(f"  (ratio > 1 means TENT is recovering more than it's breaking)")
