import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
from scipy.optimize import minimize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-100 normalization constants
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Paper-specified constants
EPS = 0.01  # Step size for decreasing omega
N_ECE_BINS = 15  # Number of bins for ECE calculation
N_OPT_AVG = 10  # Average repeats for optimization (stochastic augmentations)

# ============================================================================
# Helper Functions
# ============================================================================

def denormalize_tensor(tensor, mean=CIFAR100_MEAN, std=CIFAR100_STD):
    """Denormalize tensor for visualization/augmentation"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def normalize_tensor(tensor, mean=CIFAR100_MEAN, std=CIFAR100_STD):
    """Normalize tensor for model input"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor - mean) / std

def tensor_to_pil(tensor):
    """Convert normalized tensor to PIL Image"""
    denorm = denormalize_tensor(tensor)
    denorm = torch.clamp(denorm, 0, 1)
    return T.ToPILImage()(denorm)

def pil_to_tensor(pil_image):
    """Convert PIL Image to normalized tensor"""
    tensor = T.ToTensor()(pil_image)
    return normalize_tensor(tensor)

# ============================================================================
# Augmentation Functions (as per paper)
# ============================================================================

def horiz_flip(image):
    """Horizontal flip augmentation"""
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def rand_crop(image):
    """Random crop with ratio τ ≈ 0.78 (as per paper)"""
    w, h = image.size
    crop_size = int(w * 0.78)
    x = np.random.randint(0, w - crop_size + 1)
    y = np.random.randint(0, h - crop_size + 1)
    return image.crop((x, y, x + crop_size, y + crop_size)).resize((32, 32))

def rand_brightness(image):
    """Random brightness adjustment (β ∈ [-0.5, 0.5] as per paper)"""
    factor = np.random.uniform(0.5, 1.5)  # Equivalent to β ∈ [-0.5, 0.5]
    return ImageEnhance.Brightness(image).enhance(factor)

def rand_contrast(image):
    """Random contrast adjustment (α ∈ [-0.2, 0.2] as per paper)"""
    factor = np.random.uniform(0.8, 1.2)  # Equivalent to α ∈ [-0.2, 0.2]
    return ImageEnhance.Contrast(image).enhance(factor)

# Augmentation types in order: flip, crop, brightness, contrast
AUG_TYPES = [horiz_flip, rand_crop, rand_brightness, rand_contrast]

# Augmentation counts per paper: flip=1, others=5
AUG_COUNTS = [1, 5, 5, 5]

# ============================================================================
# Core Adaptive-TTA Logic (Equations 11-12 from paper)
# ============================================================================

def mix_probs_with_params(orig_prob, aug_probs_per_type, omega_star, omega_vec):
    """
    Compute mixed probability vector using Adaptive-TTA method.
    
    Implements equations (11) and (12) from the paper:
    - p(ω) = (1 - ω) * p0 + ω * weighted_avg_augmentations
    - ω̄ = max{ω ∈ [0, ω*] : argmax(p(ω)) = argmax(p0)}
    
    Args:
        orig_prob: torch.Tensor of shape (C,) - original prediction probabilities
        aug_probs_per_type: list of m tensors, each shape (ni, C) - augmentation probabilities
        omega_star: float in [0,1] - optimized maximum omega value
        omega_vec: numpy array of shape (m,) - optimized weights per augmentation type
    
    Returns:
        torch.Tensor of shape (C,) - mixed probability vector p(ω̄)
    """
    m = len(aug_probs_per_type)
    
    # Precompute average probability for each augmentation type (for efficiency)
    avg_aug = []
    for i in range(m):
        avg_i = aug_probs_per_type[i].mean(dim=0)  # Shape (C,)
        avg_aug.append(avg_i)
    
    def compute_p_omega(omega):
        """Compute p(ω) using equation (11)"""
        # Weighted sum of augmentation averages using |ωi| / Σ|ωi|
        abs_sum = np.sum(np.abs(omega_vec)) + 1e-12  # Avoid division by zero
        weighted_aug = torch.zeros_like(orig_prob)
        for i in range(m):
            weighted_aug += (abs(omega_vec[i]) / abs_sum) * avg_aug[i]
        
        # Mix: p(ω) = (1-ω)*p0 + ω*weighted_avg
        p = (1.0 - omega) * orig_prob + omega * weighted_aug
        return p
    
    # Find ω̄ by decreasing from ω* by ε until argmax preserved (equation 12)
    omega_t = float(omega_star)
    orig_pred = orig_prob.argmax().item()
    
    while omega_t >= 0:
        p_t = compute_p_omega(omega_t)
        if p_t.argmax().item() == orig_pred:
            return p_t  # Found ω̄
        omega_t -= EPS
    
    # If no valid ω̄ found (shouldn't happen), return original
    return orig_prob

# ============================================================================
# ECE Calculation (15 bins as per paper)
# ============================================================================

def compute_ece(probs, labels, n_bins=N_ECE_BINS):
    """
    Compute Expected Calibration Error with specified number of bins.
    
    Args:
        probs: numpy array of shape (N, C) - predicted probabilities
        labels: numpy array of shape (N,) - true labels
        n_bins: int - number of bins (15 as per paper)
    
    Returns:
        float - ECE value
    """
    probs = np.array(probs)
    labels = np.array(labels)
    
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(labels)
    
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        prop_in_bin = in_bin.mean() if N > 0 else 0.0
        
        if prop_in_bin > 0:
            acc_in_bin = (labels[in_bin] == predictions[in_bin]).mean()
            avg_conf = confidences[in_bin].mean()
            ece += np.abs(acc_in_bin - avg_conf) * prop_in_bin
    
    return ece

# ============================================================================
# Load Model and Data
# ============================================================================

print("Loading model and data...")

# Load pre-trained ResNet-56 for CIFAR-100
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet56', pretrained=True)
model = model.to(device)
model.eval()

# Load CIFAR-100 test set
testset_all = torchvision.datasets.CIFAR100(
    root='./data', 
    train=False, 
    download=True, 
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
)

# Split as per paper: first 1000 for validation, remaining 9000 for test
val_indices = list(range(0, 1000))
test_indices = list(range(1000, 10000))

valset = torch.utils.data.Subset(testset_all, val_indices)
testset = torch.utils.data.Subset(testset_all, test_indices)

val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# ============================================================================
# Cache Validation Set Augmented Probabilities
# ============================================================================

print("\nCaching validation set augmented probabilities...")

val_cache = []
val_labels = []

with torch.no_grad():
    for img, label in tqdm(val_loader, desc='Validation cache'):
        # Get original prediction
        orig_prob = F.softmax(model(img.to(device)), dim=1)[0].cpu()  # Shape: (C,)
        
        # Convert to PIL for augmentation
        pil_img = tensor_to_pil(img[0].cpu())
        
        # For each augmentation type, compute ni repeats and store as (ni, C) tensor
        per_type_repeats = []
        for aug_fn, ni in zip(AUG_TYPES, AUG_COUNTS):
            reps = []
            for _ in range(ni):
                aug_pil = aug_fn(pil_img)
                aug_tensor = pil_to_tensor(aug_pil).unsqueeze(0).to(device)
                prob = F.softmax(model(aug_tensor), dim=1)[0].cpu()
                reps.append(prob)
            
            per_type_repeats.append(torch.stack(reps))  # Shape: (ni, C)
        
        val_cache.append((orig_prob, per_type_repeats))
        val_labels.append(label.item())

print(f"Cached {len(val_cache)} validation samples")

# ============================================================================
# Optimize Parameters (ω*, ω1, ..., ωm) using Nelder-Mead
# ============================================================================

print("\nOptimizing Adaptive-TTA parameters...")

m = len(AUG_TYPES)  # Number of augmentation types

def params_to_omega(params):
    """
    Convert optimization parameters to omega values.
    
    Args:
        params: numpy array of length (1 + m)
                params[0] = s (will be sigmoid-transformed to ω*)
                params[1:] = ω1, ..., ωm
    
    Returns:
        omega_star: float in (0,1)
        omega_vec: numpy array of shape (m,)
    """
    s = params[0]
    omega_star = 1.0 / (1.0 + np.exp(-s))  # Sigmoid to ensure ∈ (0,1)
    omega_vec = np.array(params[1:])
    return omega_star, omega_vec

def objective_ece(params):
    """
    Optimization objective: minimize ECE on validation set.
    Averaged over N_OPT_AVG experiments due to stochastic augmentations.
    
    Args:
        params: numpy array of length (1 + m)
    
    Returns:
        float - mean ECE across experiments
    """
    omega_star, omega_vec = params_to_omega(params)
    
    ece_values = []
    for rep in range(N_OPT_AVG):
        calibrated_probs = []
        
        for (orig_prob, per_type_repeats), true_label in zip(val_cache, val_labels):
            # Apply Adaptive-TTA mixing
            mixed = mix_probs_with_params(orig_prob, per_type_repeats, omega_star, omega_vec)
            calibrated_probs.append(mixed.numpy())
        
        # Compute ECE for this experiment
        ece = compute_ece(calibrated_probs, val_labels, n_bins=N_ECE_BINS)
        ece_values.append(ece)
    
    return float(np.mean(ece_values))

# Initialize with multiple starting points for robustness
x0_initial = np.concatenate(([0.0], np.zeros(m)))

initial_guesses = [
    x0_initial,
    np.concatenate(([0.0], 0.05 * np.ones(m))),
    np.concatenate(([-1.0], 0.1 * np.ones(m))),
    np.concatenate(([1.0], 0.2 * np.ones(m)))
]

best_result = None
best_ece = float('inf')

for i, x0 in enumerate(initial_guesses):
    print(f"  Optimization attempt {i+1}/{len(initial_guesses)}...")
    result = minimize(
        objective_ece, 
        x0, 
        method='Nelder-Mead',
        options={'maxiter': 800, 'xatol': 1e-4, 'fatol': 1e-4}
    )
    
    if result.fun < best_ece:
        best_ece = result.fun
        best_result = result
        print(f"    New best ECE: {best_ece:.6f}")

# Extract optimized parameters
omega_star_opt, omega_vec_opt = params_to_omega(best_result.x)

print(f"\n{'='*70}")
print(f"Optimization complete!")
print(f"{'='*70}")
print(f"Optimized ω*: {omega_star_opt:.4f}")
print(f"Optimized ω_vec: {omega_vec_opt}")
print(f"Validation ECE: {best_ece:.6f}")
print(f"{'='*70}\n")

# ============================================================================
# Apply Adaptive-TTA to Test Set
# ============================================================================

print("Applying Adaptive-TTA to test set...")

y_true = []
y_orig = []
y_cal = []
all_cal_probs = []

with torch.no_grad():
    for img, label in tqdm(test_loader, desc='Test set evaluation'):
        # Get original prediction
        orig_prob = F.softmax(model(img.to(device)), dim=1)[0].cpu()
        
        # Convert to PIL for augmentation
        pil_img = tensor_to_pil(img[0].cpu())
        
        # Generate augmented predictions
        per_type_repeats = []
        for aug_fn, ni in zip(AUG_TYPES, AUG_COUNTS):
            reps = []
            for _ in range(ni):
                aug_pil = aug_fn(pil_img)
                aug_tensor = pil_to_tensor(aug_pil).unsqueeze(0).to(device)
                prob = F.softmax(model(aug_tensor), dim=1)[0].cpu()
                reps.append(prob)
            
            per_type_repeats.append(torch.stack(reps))
        
        # Apply Adaptive-TTA mixing
        mixed = mix_probs_with_params(orig_prob, per_type_repeats, omega_star_opt, omega_vec_opt)
        
        # Store results
        all_cal_probs.append(mixed.numpy())
        y_true.append(label.item())
        y_orig.append(orig_prob.argmax().item())
        y_cal.append(mixed.argmax().item())

# Convert to numpy arrays
y_true = np.array(y_true)
y_orig = np.array(y_orig)
y_cal = np.array(y_cal)

# ============================================================================
# Compute and Display Results
# ============================================================================

# Compute metrics
orig_acc = np.mean(y_true == y_orig)
cal_acc = np.mean(y_true == y_cal)
cal_ece = compute_ece(all_cal_probs, y_true, n_bins=N_ECE_BINS)

# Detailed accuracy analysis
correct_to_incorrect = np.sum((y_true == y_orig) & (y_true != y_cal))
incorrect_to_correct = np.sum((y_true != y_orig) & (y_true == y_cal))
total_correct_initially = np.sum(y_true == y_orig)
total_incorrect_initially = np.sum(y_true != y_orig)

print(f"\n{'='*70}")
print(f"FINAL RESULTS (Test Set: 9000 images)")
print(f"{'='*70}")
print(f"Original Accuracy:     {orig_acc:.4f} ({total_correct_initially}/{len(y_true)})")
print(f"Adaptive-TTA Accuracy: {cal_acc:.4f}")
print(f"Adaptive-TTA ECE:      {cal_ece:.6f}")
print(f"{'='*70}")
print(f"\nAccuracy Change Analysis:")
print(f"  Correct → Incorrect: {correct_to_incorrect}")
print(f"  Incorrect → Correct: {incorrect_to_correct}")
print(f"  Net change:          {incorrect_to_correct - correct_to_incorrect:+d}")
print(f"{'='*70}\n")
