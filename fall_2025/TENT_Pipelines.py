import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import copy

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
BATCH_SIZE = 200  # Larger batch for stability
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
CIFAR_STD = (0.2023, 0.1994, 0.2010)

corruptions = [
    'defocus_blur', 'glass_blur', 'motion_blur',
    'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# ======= MODEL =======
print("Loading model...")
model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    model_name,
    pretrained=True
).to(device)

# ======= SIMPLE TENT IMPLEMENTATION (MORE STABLE) =======
def configure_model_for_tent(model):
    """Configure model: freeze all except BN."""
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.requires_grad_(True)
            # Reset BN stats for better adaptation
            module.reset_running_stats()
            module.momentum = 0.1
    
    return model

def collect_bn_params(model):
    """Collect BatchNorm parameters."""
    params = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for param in module.parameters():
                if param.requires_grad:
                    params.append(param)
    return params

def softmax_entropy(x):
    """Compute entropy of softmax."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def compute_shannon_entropy(image_tensor):
    """
    Compute Shannon entropy of an image tensor.
    Args:
        image_tensor: torch.Tensor of shape (C, H, W) normalized
    Returns:
        float: Shannon entropy value
    """
    from scipy.stats import entropy as scipy_entropy
    
    # Denormalize to [0, 1] range
    img = image_tensor.clone().cpu()
    for c in range(3):
        img[c] = img[c] * CIFAR_STD[c] + CIFAR_MEAN[c]
    img = torch.clamp(img, 0, 1)
    
    # Compute histogram
    img_np = img.numpy().flatten()
    hist, _ = np.histogram(img_np, bins=256, range=(0, 1), density=True)
    
    # Remove zero entries and compute entropy
    hist = hist[hist > 0]
    return scipy_entropy(hist, base=2)

@torch.enable_grad()
def tent_forward_and_adapt(model, x, optimizer):
    """
    Simple TENT: just minimize entropy, no fancy tricks.
    """
    # Forward
    outputs = model(x)
    
    # Compute entropy loss
    loss = softmax_entropy(outputs).mean()
    
    # Backward and update
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(collect_bn_params(model), max_norm=1.0)
    
    optimizer.step()
    optimizer.zero_grad()
    
    return outputs

# ======= EVALUATION FUNCTIONS =======
def evaluate_baseline(model, dataloader):
    """Baseline without adaptation."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return 100 * correct / total

def evaluate_with_tent_simple(model, dataloader, lr=0.00025, steps=1):
    """
    Simple TENT adaptation with safeguards.
    """
    # Save original model
    model_state = copy.deepcopy(model.state_dict())
    
    # Configure for TENT
    model = configure_model_for_tent(model)
    params = collect_bn_params(model)
    
    if len(params) == 0:
        print("WARNING: No parameters to adapt!")
        model.load_state_dict(model_state)
        return evaluate_baseline(model, dataloader)
    
    optimizer = torch.optim.Adam(params, lr=lr)
    
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Adapt
        for _ in range(steps):
            outputs = tent_forward_and_adapt(model, images, optimizer)
        
        # Evaluate (no grad)
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    
    # Restore
    model.load_state_dict(model_state)
    
    return accuracy

def evaluate_with_tent_reset(model, dataloader, lr=0.001):
    """
    TENT with per-batch reset - most stable approach.
    Reset model to original state after each batch.
    """
    # Save original model
    model_state = copy.deepcopy(model.state_dict())
    
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Reset model for this batch
        model.load_state_dict(model_state)
        model = configure_model_for_tent(model)
        params = collect_bn_params(model)
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # Adapt on this batch
        outputs = tent_forward_and_adapt(model, images, optimizer)
        
        # Evaluate
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    
    # Restore original
    model.load_state_dict(model_state)
    
    return accuracy

def evaluate_with_tent_conditional(model, dataloader, lr=0.001, entropy_threshold=7.16):
    """
    TENT with Shannon entropy threshold.
    Only adapts images with Shannon entropy > threshold.
    Uses per-batch reset for stability.
    """
    # Save original model
    model_state = copy.deepcopy(model.state_dict())
    
    # Keep eval version for low-entropy images
    model_eval = copy.deepcopy(model)
    model_eval.eval()
    
    correct = 0
    total = 0
    high_entropy_count = 0
    adapted_batch_count = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # Check Shannon entropy for each image in batch
        shannon_entropies = []
        for i in range(batch_size):
            shannon_ent = compute_shannon_entropy(images[i])
            shannon_entropies.append(shannon_ent)
        
        # Count high-entropy images
        high_entropy_mask = [ent > entropy_threshold for ent in shannon_entropies]
        num_high_entropy = sum(high_entropy_mask)
        high_entropy_count += num_high_entropy
        
        # Decide whether to adapt this batch
        # If more than 50% are high-entropy, adapt the whole batch
        if num_high_entropy > batch_size / 2:
            # High entropy batch: Apply TENT
            adapted_batch_count += 1
            
            # Reset model for this batch
            model.load_state_dict(model_state)
            model = configure_model_for_tent(model)
            params = collect_bn_params(model)
            optimizer = torch.optim.Adam(params, lr=lr)
            
            # Adapt
            outputs = tent_forward_and_adapt(model, images, optimizer)
        else:
            # Low entropy batch: Use baseline model
            with torch.no_grad():
                outputs = model_eval(images)
        
        # Evaluate
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    high_entropy_ratio = 100 * high_entropy_count / total
    
    # Restore original
    model.load_state_dict(model_state)
    
    return accuracy, high_entropy_count, total, adapted_batch_count

# ======= MAIN EVALUATION LOOP =======
ENTROPY_THRESHOLD = 1.16  # Shannon entropy threshold

results = {
    'baseline': {},
    'tent_simple': {},
    'tent_reset': {},
    'tent_conditional': {}
}

print("\n" + "="*80)
print("SIMPLIFIED TENT WITH STABILITY FIXES")
print("="*80)
print("Testing three approaches:")
print("1. TENT Simple: Continuous adaptation (lr=0.00025)")
print("2. TENT Reset: Reset model after each batch (lr=0.001, most stable)")
print(f"3. TENT Conditional: Only adapt when Shannon entropy > {ENTROPY_THRESHOLD}")
print("="*80)

for corr in corruptions:
    print(f"\n{'='*80}")
    print(f"Corruption: {corr}")
    print(f"{'='*80}")
    
    results['baseline'][corr] = []
    results['tent_simple'][corr] = []
    results['tent_reset'][corr] = []
    results['tent_conditional'][corr] = []
    
    for severity in range(1, 6):
        print(f"\n--- Severity: {severity} ---")
        
        # Load dataset
        testset = CIFARC_Dataset(
            root=cifarc_root,
            corruption=corr,
            severity=severity,
            transform=transform
        )
        testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Baseline
        print("Evaluating baseline...")
        baseline_acc = evaluate_baseline(model, testloader)
        results['baseline'][corr].append(baseline_acc)
        print(f"Baseline: {baseline_acc:.2f}%")
        
        # TENT Simple
        print("Evaluating TENT Simple...")
        tent_simple_acc = evaluate_with_tent_simple(model, testloader, lr=0.00025)
        results['tent_simple'][corr].append(tent_simple_acc)
        print(f"TENT Simple: {tent_simple_acc:.2f}% ({tent_simple_acc - baseline_acc:+.2f}%)")
        
        # TENT Reset (most stable)
        print("Evaluating TENT Reset...")
        tent_reset_acc = evaluate_with_tent_reset(model, testloader, lr=0.001)
        results['tent_reset'][corr].append(tent_reset_acc)
        print(f"TENT Reset: {tent_reset_acc:.2f}% ({tent_reset_acc - baseline_acc:+.2f}%)")
        
        # TENT Conditional (with Shannon entropy threshold)
        print(f"Evaluating TENT Conditional (Shannon > {ENTROPY_THRESHOLD})...")
        tent_cond_acc, high_ent_count, total, adapted_batches = evaluate_with_tent_conditional(
            model, testloader, lr=0.001, entropy_threshold=ENTROPY_THRESHOLD
        )
        results['tent_conditional'][corr].append(tent_cond_acc)
        high_ent_ratio = 100 * high_ent_count / total
        print(f"TENT Conditional: {tent_cond_acc:.2f}% ({tent_cond_acc - baseline_acc:+.2f}%)")
        print(f"  High-entropy images: {high_ent_count}/{total} ({high_ent_ratio:.1f}%)")
        print(f"  Batches adapted: {adapted_batches}/{len(testloader)}")

# ======= FINAL SUMMARY =======
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

for corr in corruptions:
    print(f"\n{corr}:")
    for sev in range(5):
        baseline = results['baseline'][corr][sev]
        simple = results['tent_simple'][corr][sev]
        reset = results['tent_reset'][corr][sev]
        cond = results['tent_conditional'][corr][sev]
        print(f"  Sev {sev+1}: Base:{baseline:.1f}% | Simple:{simple:.1f}%({simple-baseline:+.1f}%) | Reset:{reset:.1f}%({reset-baseline:+.1f}%) | Cond:{cond:.1f}%({cond-baseline:+.1f}%)")
    
    avg_base = np.mean(results['baseline'][corr])
    avg_simple = np.mean(results['tent_simple'][corr])
    avg_reset = np.mean(results['tent_reset'][corr])
    avg_cond = np.mean(results['tent_conditional'][corr])
    print(f"  Avg: Base:{avg_base:.1f}% | Simple:{avg_simple:.1f}%({avg_simple-avg_base:+.1f}%) | Reset:{avg_reset:.1f}%({avg_reset-avg_base:+.1f}%) | Cond:{avg_cond:.1f}%({avg_cond-avg_base:+.1f}%)")

# Overall
all_base = [acc for corr in corruptions for acc in results['baseline'][corr]]
all_simple = [acc for corr in corruptions for acc in results['tent_simple'][corr]]
all_reset = [acc for corr in corruptions for acc in results['tent_reset'][corr]]
all_cond = [acc for corr in corruptions for acc in results['tent_conditional'][corr]]

print(f"\n{'='*80}")
print(f"OVERALL STATISTICS")
print(f"{'='*80}")
print(f"Baseline: {np.mean(all_base):.2f}%")
print(f"TENT Simple: {np.mean(all_simple):.2f}% ({np.mean(all_simple)-np.mean(all_base):+.2f}%)")
print(f"TENT Reset: {np.mean(all_reset):.2f}% ({np.mean(all_reset)-np.mean(all_base):+.2f}%)")
print(f"TENT Conditional: {np.mean(all_cond):.2f}% ({np.mean(all_cond)-np.mean(all_base):+.2f}%)")
print(f"\nNote: Conditional TENT only adapts when Shannon entropy > {ENTROPY_THRESHOLD}")
