import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
n_classes = 100

### Normalization utilities ###
def denormalize_tensor(tensor):
    mean = torch.tensor(CIFAR100_MEAN).view(3,1,1).to(tensor.device)
    std  = torch.tensor(CIFAR100_STD).view(3,1,1).to(tensor.device)
    return tensor * std + mean

def normalize_tensor(tensor):
    mean = torch.tensor(CIFAR100_MEAN).view(3,1,1).to(tensor.device)
    std  = torch.tensor(CIFAR100_STD).view(3,1,1).to(tensor.device)
    return (tensor - mean) / std

def tensor_to_pil(t):
    t = torch.clamp(denormalize_tensor(t), 0, 1)
    return T.ToPILImage()(t.cpu())

def pil_to_tensor(p):
    t = T.ToTensor()(p)
    return normalize_tensor(t)

### Augmentation functions ###
def vert_flip(img): return img.transpose(Image.FLIP_LEFT_RIGHT)
def rand_crop(img):
    w, h = img.size
    cs = int(0.78 * w)
    x, y = np.random.randint(0, w - cs + 1), np.random.randint(0, h - cs + 1)
    return img.crop((x, y, x + cs, y + cs)).resize((32, 32))
def rand_brightness(img):
    return ImageEnhance.Brightness(img).enhance(np.random.uniform(0.5, 1.5))
def rand_contrast(img):
    return ImageEnhance.Contrast(img).enhance(np.random.uniform(0.8, 1.2))

AUG_FN = [vert_flip, rand_crop, rand_brightness, rand_contrast]
POLICIES = [
    [0,1], [0,2], [1,2], [0,1,2], [0,1,3], [0,2,3], [1,2,3], [0,1,2,3]
]

### Dataset ###
testset = torchvision.datasets.CIFAR100(
    './data', train=False, download=True,
    transform=T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
)
valset = torch.utils.data.Subset(testset, range(0, 1000))
val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)
testset = torch.utils.data.Subset(testset, range(1000, 2000))
test_loader = torch.utils.data.DataLoader(testset, batch_size=1)

### Model ###
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet56', pretrained=True).to(device)
model.eval()

### Updated: Get logits instead of probabilities ###
def get_aug_logits_for_policy(model, img, policy_idxs, n_aug=5):
    pil_img = tensor_to_pil(img.cpu())
    with torch.no_grad():
        orig_logit = model(img.unsqueeze(0).to(device))[0].cpu()
    aug_logits = []
    for aug_idx in policy_idxs:
        reps = []
        for _ in range(n_aug):
            aug_t = pil_to_tensor(AUG_FN[aug_idx](pil_img)).unsqueeze(0).to(device)
            with torch.no_grad():
                rep_logit = model(aug_t)[0].cpu()
            reps.append(rep_logit)
        aug_logits.append(torch.stack(reps))
    return orig_logit, aug_logits

### Binary search for optimal rho ###
def find_optimal_rho(orig, weighted_Z, c0, tol=1e-3):
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        p = F.softmax((1 - mid) * orig + mid * weighted_Z, dim=0)
        if p.argmax().item() == c0:
            lo = mid
        else:
            hi = mid
    return lo

### Metrics ###
def brier_score(pred_probs, true_labels):
    return np.mean((pred_probs[np.arange(len(true_labels)), true_labels] - 1) ** 2)

def mc_brier_score(pred_probs, true_labels):
    onehot = np.eye(n_classes)[true_labels]
    return np.mean(np.sum((pred_probs - onehot) ** 2, axis=1))

def nll(pred_probs, true_labels):
    eps = 1e-12
    return -np.mean(np.log(pred_probs[np.arange(len(true_labels)), true_labels] + eps))

def compute_ece(probs, labels, n_bins=15):
    probs, labels = np.array(probs), np.array(labels)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (conf > bins[i]) & (conf <= bins[i+1])
        if np.any(in_bin):
            acc = (labels[in_bin] == preds[in_bin]).mean()
            avg_conf = conf[in_bin].mean()
            ece += np.abs(acc - avg_conf) * in_bin.mean()
    return ece

### --- Train M-ATTA and V-ATTA (unchanged except logits) --- ###
best_matta = None
best_matta_score = np.inf
best_vatta = None
best_vatta_score = np.inf

for policy_idxs in POLICIES:
    print(f"\nRunning policy: {policy_idxs}")
    val_cache, val_labels = [], []
    for img, label in tqdm(val_loader, desc=f'Cache policy {policy_idxs}'):
        orig, aug = get_aug_logits_for_policy(model, img[0], policy_idxs)
        val_cache.append((orig, aug))
        val_labels.append(label.item())

    # --- M-ATTA optimization ---
    W = torch.ones(n_classes, len(policy_idxs), requires_grad=True, device=device)
    optimizer = Adam([W], lr=0.001)

    for epoch in range(500):
        idxs = np.random.choice(len(val_cache), min(500, len(val_cache)), replace=True)
        loss = 0
        for idx in idxs:
            orig, aug = val_cache[idx]
            Z = torch.stack([a.mean(0) for a in aug], dim=1).to(device)
            c0 = orig.argmax().item()
            weighted_Z = (W * Z).sum(dim=1).cpu()
            rho = find_optimal_rho(orig, weighted_Z, c0)
            p_vec = (1 - rho) * orig + rho * weighted_Z
            p = F.softmax(p_vec, dim=0)
            loss += -torch.log(p[val_labels[idx]] + 1e-12)
        optimizer.zero_grad()
        (loss / len(idxs)).backward()
        optimizer.step()

    all_probs = []
    for orig, aug in val_cache:
        Z = torch.stack([a.mean(0) for a in aug], dim=1).to(device)
        c0 = orig.argmax().item()
        weighted_Z = (W.detach().cpu() * Z.cpu()).sum(dim=1)
        rho = find_optimal_rho(orig, weighted_Z, c0)
        p = F.softmax((1 - rho) * orig + rho * weighted_Z, dim=0)
        all_probs.append(p.numpy())
    score = brier_score(np.array(all_probs), np.array(val_labels))
    print(f"Brier score (M-ATTA, policy {policy_idxs}): {score:.4f}")
    if score < best_matta_score:
        best_matta_score = score
        best_matta = (policy_idxs, W.detach().cpu())

    # --- V-ATTA optimization ---
    Wd = torch.ones(len(policy_idxs), requires_grad=True, device=device)
    optimizer = Adam([Wd], lr=0.001)
    for epoch in range(500):
        idxs = np.random.choice(len(val_cache), min(500, len(val_cache)), replace=True)
        loss = 0
        for idx in idxs:
            orig, aug = val_cache[idx]
            Z = torch.stack([a.mean(0) for a in aug], dim=1).to(device)
            c0 = orig.argmax().item()
            weighted_Z = (Z.cpu() * Wd.cpu()).sum(dim=1)
            rho = find_optimal_rho(orig, weighted_Z, c0)
            p = F.softmax((1 - rho) * orig + rho * weighted_Z, dim=0)
            loss += -torch.log(p[val_labels[idx]] + 1e-12)
        optimizer.zero_grad()
        (loss / len(idxs)).backward()
        optimizer.step()

    all_probs = []
    for orig, aug in val_cache:
        Z = torch.stack([a.mean(0) for a in aug], dim=1).to(device)
        c0 = orig.argmax().item()
        weighted_Z = (Z.cpu() * Wd.detach().cpu()).sum(dim=1)
        rho = find_optimal_rho(orig, weighted_Z, c0)
        p = F.softmax((1 - rho) * orig + rho * weighted_Z, dim=0)
        all_probs.append(p.numpy())
    score = brier_score(np.array(all_probs), np.array(val_labels))
    print(f"Brier score (V-ATTA, policy {policy_idxs}): {score:.4f}")
    if score < best_vatta_score:
        best_vatta_score = score
        best_vatta = (policy_idxs, Wd.detach().cpu())

print("\n============================================================")
print("BEST M-ATTA policy:", best_matta[0])
print("BEST V-ATTA policy:", best_vatta[0])
print("============================================================")

### --- Test Evaluation (same corrections) --- ###
def apply_matta(orig, aug, W):
    Z = torch.stack([a.mean(0) for a in aug], dim=1)
    c0 = orig.argmax().item()
    weighted_Z = (W * Z).sum(dim=1)
    rho = find_optimal_rho(orig, weighted_Z, c0)
    return F.softmax((1 - rho) * orig + rho * weighted_Z, dim=0).cpu().numpy()

def apply_vatta(orig, aug, Wd):
    Z = torch.stack([a.mean(0) for a in aug], dim=1)
    c0 = orig.argmax().item()
    weighted_Z = (Z * Wd).sum(dim=1)
    rho = find_optimal_rho(orig, weighted_Z, c0)
    return F.softmax((1 - rho) * orig + rho * weighted_Z, dim=0).cpu().numpy()

results = {'vanilla': [], 'matta': [], 'vatta': [], 'label': []}
matta_policy_idxs, W_m = best_matta
vatta_policy_idxs, Wd_v = best_vatta

for img, label in tqdm(test_loader, desc='Test set'):
    orig_m, aug_m = get_aug_logits_for_policy(model, img[0], matta_policy_idxs)
    orig_v, aug_v = get_aug_logits_for_policy(model, img[0], vatta_policy_idxs)
    results['vanilla'].append(F.softmax(orig_m, dim=0).numpy())
    results['matta'].append(apply_matta(orig_m, aug_m, W_m))
    results['vatta'].append(apply_vatta(orig_v, aug_v, Wd_v))
    results['label'].append(label.item())

y_true = np.array(results['label'])
y_vanilla = np.array(results['vanilla'])
y_matta = np.array(results['matta'])
y_vatta = np.array(results['vatta'])

print("\n============================================================")
print("FINAL TEST RESULTS:")
print("============================================================")
print(f"Brier (vanilla): {brier_score(y_vanilla, y_true):.4f}")
print(f"Brier (M-ATTA):  {brier_score(y_matta, y_true):.4f}")
print(f"Brier (V-ATTA):  {brier_score(y_vatta, y_true):.4f}")
print(f"\nNLL (vanilla): {nll(y_vanilla, y_true):.4f}")
print(f"NLL (M-ATTA):  {nll(y_matta, y_true):.4f}")
print(f"NLL (V-ATTA):  {nll(y_vatta, y_true):.4f}")
print(f"\nECE (vanilla): {compute_ece(y_vanilla, y_true):.4f}")
print(f"ECE (M-ATTA):  {compute_ece(y_matta, y_true):.4f}")
print(f"ECE (V-ATTA):  {compute_ece(y_vatta, y_true):.4f}")

# Optional: print accuracies
vanilla_acc = (y_vanilla.argmax(1) == y_true).mean() * 100
matta_acc = (y_matta.argmax(1) == y_true).mean() * 100
vatta_acc = (y_vatta.argmax(1) == y_true).mean() * 100
print(f"\nAccuracy (vanilla): {vanilla_acc:.2f}%")
print(f"Accuracy (M-ATTA):  {matta_acc:.2f}%")
print(f"Accuracy (V-ATTA):  {vatta_acc:.2f}%")
print("============================================================")

