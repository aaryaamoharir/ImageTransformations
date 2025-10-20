import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
from scipy.optimize import minimize

# -------------------
# Device & constants
# -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# -------------------
# Tensor <-> PIL helpers
# -------------------
def denormalize_tensor(tensor, mean=CIFAR100_MEAN, std=CIFAR100_STD, device=device):
    mean_t = torch.tensor(mean, device=tensor.device if isinstance(tensor, torch.Tensor) else device).view(3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device if isinstance(tensor, torch.Tensor) else device).view(3, 1, 1)
    return tensor * std_t + mean_t

def normalize_tensor(tensor, mean=CIFAR100_MEAN, std=CIFAR100_STD, device=device):
    mean_t = torch.tensor(mean, device=tensor.device if isinstance(tensor, torch.Tensor) else device).view(3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device if isinstance(tensor, torch.Tensor) else device).view(3, 1, 1)
    return (tensor - mean_t) / std_t

def tensor_to_pil(tensor):
    # tensor expected normalized in CIFAR stats
    denorm = denormalize_tensor(tensor)
    denorm = torch.clamp(denorm, 0, 1)
    return T.ToPILImage()(denorm.cpu())

def pil_to_tensor(pil_image):
    tensor = T.ToTensor()(pil_image)  # [0,1]
    return normalize_tensor(tensor)

# -------------------
# Augmentations
# -------------------
def vert_flip(image: Image.Image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def rand_crop(image: Image.Image):
    w, h = image.size
    crop_size = int(w * 0.78)
    x = np.random.randint(0, w - crop_size + 1)
    y = np.random.randint(0, h - crop_size + 1)
    return image.crop((x, y, x + crop_size, y + crop_size)).resize((32, 32))

def rand_brightness(image: Image.Image):
    factor = np.random.uniform(0.5, 1.5)
    return ImageEnhance.Brightness(image).enhance(factor)

def rand_contrast(image: Image.Image):
    factor = np.random.uniform(0.8, 1.2)
    return ImageEnhance.Contrast(image).enhance(factor)

AUG_TYPES = [vert_flip, rand_crop, rand_brightness, rand_contrast]

# -------------------
# Forward + augmentation probability collection
# -------------------
def get_augmented_probs(model, image_tensor, aug_types, n_aug=5):
    """
    image_tensor: normalized tensor shape [3,32,32] (on CPU or device). We'll convert to PIL and run augmentations.
    Returns:
      orig_probs: torch.tensor shape [n_classes] (cpu)
      aug_probs: list of torch.tensor shape [n_aug, n_classes] (cpu) for each aug type
    """
    # Convert to PIL (denormalize and clamp inside tensor_to_pil)
    pil_img = tensor_to_pil(image_tensor)
    # Run original through model (make sure input is normalized and on device)
    model_input = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        orig_logits = model(model_input)
        orig_probs = F.softmax(orig_logits, dim=1).cpu()[0]  # cpu tensor [n_classes]
    aug_probs = []
    with torch.no_grad():
        for aug in aug_types:
            aug_type_probs = []
            for _ in range(n_aug):
                aug_img = aug(pil_img)
                aug_tensor = pil_to_tensor(aug_img).unsqueeze(0).to(device)
                prob = F.softmax(model(aug_tensor), dim=1).cpu()[0]
                aug_type_probs.append(prob)
            aug_probs.append(torch.stack(aug_type_probs))  # [n_aug, n_classes] cpu
    return orig_probs, aug_probs

# -------------------
# ECE computation
# -------------------
def compute_ece(probs, labels, n_bins=15):
    """
    probs: list or array shape [N, n_classes]
    labels: array shape [N]
    """
    probs = np.array(probs)
    labels = np.array(labels)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
        prop_in_bin = in_bin.mean() if in_bin.size > 0 else 0.0
        if prop_in_bin > 0:
            acc_in_bin = (labels[in_bin] == predictions[in_bin]).mean()
            avg_conf = confidences[in_bin].mean()
            ece += np.abs(acc_in_bin - avg_conf) * prop_in_bin
    return float(ece)

# -------------------
# Load data & model
# -------------------
transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
testset_full = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Validation subset: first 1000 images
valset = torch.utils.data.Subset(testset_full, range(0, 1000))
val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)

# Load pretrained model
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet56', pretrained=True).to(device)
model.eval()

# -------------------
# Cache augmented probabilities on validation set
# -------------------
n_aug = 5  # number of stochastic augmentations per aug-type
val_probs_vecs = []  # list of tuples (orig_prob_cpu, aug_probs_list_of_[n_aug, n_classes]_cpu)
val_labels = []

print("Caching augmentations for validation set (this may take a while)...")
for img, label in tqdm(val_loader, desc='Caching Augmentations'):
    # img shape [1,3,32,32]
    orig_prob, aug_probs = get_augmented_probs(model, img[0], AUG_TYPES, n_aug=n_aug)
    val_probs_vecs.append((orig_prob, aug_probs))
    val_labels.append(label.item())

print("Done caching validation augmentations.")

# -------------------
# Adaptive-TTA mixing & optimization
# -------------------
def adaptive_tta_mix(orig_prob, aug_probs, lam):
    """
    orig_prob: torch tensor [C] (cpu)
    aug_probs: list of torch tensors [n_aug, C] (cpu)
    lam: iterable length 1 + len(AUG_TYPES) that sums to 1 (numpy or torch)
    returns: torch tensor [C] (cpu)
    """
    # Convert lam to torch on CPU (orig_prob is CPU)
    lam_t = torch.tensor(lam, dtype=orig_prob.dtype)
    mix = lam_t[0] * orig_prob
    for i in range(len(AUG_TYPES)):
        avg_aug = aug_probs[i].mean(dim=0)  # [C]
        mix = mix + lam_t[i + 1] * avg_aug
    return mix

def ece_loss_obj(lam_vec):
    """
    lam_vec: size 5 (λ0..λ4) or size 5 as optimizer variable. We'll ensure positivity and normalization.
    We'll apply a soft penalty for class flips rather than a hard block.
    Returns scalar loss: ECE + flip_penalty
    """
    # Force non-negative
    lam = np.clip(lam_vec, 0.0, 1.0)
    # Normalize to sum 1 (convex mixture)
    if lam.sum() == 0:
        lam = np.ones_like(lam) / lam.size
    else:
        lam = lam / lam.sum()

    calibrated_probs = []
    flip_penalty = 0.0
    for orig_prob, aug_probs in val_probs_vecs:
        cal_prob = adaptive_tta_mix(orig_prob, aug_probs, lam)
        orig_class = int(orig_prob.argmax().item())
        new_class = int(cal_prob.argmax().item())
        if orig_class != new_class:
            # Soft penalty: small cost for flipping top-1
            flip_penalty += 0.05
        calibrated_probs.append(cal_prob.detach().numpy())
    ece_val = compute_ece(calibrated_probs, np.array(val_labels), n_bins=15)
    total_loss = ece_val + flip_penalty
    return float(total_loss)

# Start optimization: we optimize 5 lambdas (λ0..λ4) directly with constraint sum=1
init_lam = np.array([0.6, 0.1, 0.1, 0.1, 0.1])  # start favoring original
bounds = [(0.0, 1.0)] * (1 + len(AUG_TYPES))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},)

print("Optimizing lambda vector (SLSQP). This may take a while...")
res = minimize(ece_loss_obj, init_lam, method='SLSQP', bounds=bounds, constraints=constraints,
               options={'maxiter': 200, 'ftol': 1e-6, 'disp': True})

if not res.success:
    print("Optimizer warning:", res.message)

best_lam = np.clip(res.x, 0.0, 1.0)
best_lam = best_lam / best_lam.sum()
print("Optimized Lambda:", best_lam)

# -------------------
# Evaluation on next 1000 images (1000..1999)
# -------------------
testset = torch.utils.data.Subset(testset_full, range(1000, 2000))
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

y_true = []
y_orig = []
y_cal = []
all_cal_probs = []

print("Running Adaptive-TTA on test subset (next 1000 images)...")
for img, label in tqdm(test_loader, desc='Adaptive-TTA Test'):
    orig_prob, aug_probs = get_augmented_probs(model, img[0], AUG_TYPES, n_aug=n_aug)
    cal_prob = adaptive_tta_mix(orig_prob, aug_probs, best_lam)
    y_true.append(label.item())
    y_orig.append(int(orig_prob.argmax().item()))
    y_cal.append(int(cal_prob.argmax().item()))
    all_cal_probs.append(cal_prob.detach().numpy())

y_true = np.array(y_true)
y_orig = np.array(y_orig)
y_cal = np.array(y_cal)

orig_acc = float((y_true == y_orig).mean())
cal_acc = float((y_true == y_cal).mean())
cal_ece = compute_ece(all_cal_probs, y_true, n_bins=15)

# flips
correct_to_incorrect = int(np.sum((y_true == y_orig) & (y_true != y_cal)))
incorrect_to_correct = int(np.sum((y_true != y_orig) & (y_true == y_cal)))

print(f'Original Accuracy: {orig_acc:.4f}, Adaptive-TTA Accuracy: {cal_acc:.4f}, Adaptive-TTA ECE: {cal_ece:.4f}')
print(f"# correct initially -> incorrect after TTA: {correct_to_incorrect}")
print(f"# incorrect initially -> correct after TTA: {incorrect_to_correct}")
