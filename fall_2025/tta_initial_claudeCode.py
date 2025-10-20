import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
from scipy.optimize import minimize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

def denormalize_tensor(tensor, mean=CIFAR100_MEAN, std=CIFAR100_STD):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def normalize_tensor(tensor, mean=CIFAR100_MEAN, std=CIFAR100_STD):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor - mean) / std

def tensor_to_pil(tensor):
    denorm = denormalize_tensor(tensor)
    denorm = torch.clamp(denorm, 0, 1)
    return T.ToPILImage()(denorm)

def pil_to_tensor(pil_image):
    tensor = T.ToTensor()(pil_image)
    return normalize_tensor(tensor)

# --- Augmentation Types ---
def horiz_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def rand_crop(image):
    w, h = image.size
    crop_size = int(w * 0.78)
    x = np.random.randint(0, w - crop_size + 1)
    y = np.random.randint(0, h - crop_size + 1)
    return image.crop((x, y, x + crop_size, y + crop_size)).resize((32, 32))

def rand_brightness(image):
    factor = np.random.uniform(0.5, 1.5)
    return ImageEnhance.Brightness(image).enhance(factor)

def rand_contrast(image):
    factor = np.random.uniform(0.8, 1.2)
    return ImageEnhance.Contrast(image).enhance(factor)

AUG_TYPES = [horiz_flip, rand_crop, rand_brightness, rand_contrast]

def get_augmented_probs(model, image, aug_types, n_aug=5):
    """Apply each type n_aug times, collect results."""
    pil_img = tensor_to_pil(image.cpu())
    orig_probs = F.softmax(model(image.unsqueeze(0).to(device)), dim=1).cpu()[0]
    aug_probs = []
    with torch.no_grad():
        for aug in aug_types:
            aug_type_probs = []
            for _ in range(n_aug):
                aug_img = aug(pil_img)
                aug_tensor = pil_to_tensor(aug_img).unsqueeze(0).to(device)
                prob = F.softmax(model(aug_tensor), dim=1).cpu()[0]
                aug_type_probs.append(prob)
            aug_probs.append(torch.stack(aug_type_probs))
    return orig_probs, aug_probs

# ---- Calibration ECE ----
def compute_ece(probs, labels, n_bins=15):
    probs = np.array(probs)
    labels = np.array(labels)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = (labels[in_bin] == predictions[in_bin]).mean()
            avg_conf = confidences[in_bin].mean()
            ece += np.abs(acc_in_bin - avg_conf) * prop_in_bin
    return ece

# ---- Validation Split ----
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
]))
valset = torch.utils.data.Subset(testset, range(0, 1000))
val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)

model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet56', pretrained=True).to(device)
model.eval()

# --- Cache augmented prob vectors for val set ---
val_probs_vecs = []
val_labels = []
for img, label in tqdm(val_loader, desc='Caching Augmentations'):
    orig_prob, aug_probs = get_augmented_probs(model, img[0], AUG_TYPES, n_aug=5)
    val_probs_vecs.append((orig_prob, aug_probs))
    val_labels.append(label.item())

# --- Adaptive-TTA optimization (find best λ) ---
def adaptive_tta_mix(orig_prob, aug_probs, lam):
    n_aug = aug_probs[0].size(0)
    mix = lam[0] * orig_prob
    for i in range(len(AUG_TYPES)):
        avg_aug = aug_probs[i].mean(dim=0)
        mix += lam[i+1] * avg_aug
    return mix

def combined_loss_obj(lam_vec):
    """Optimize for both ECE and accuracy"""
    lam_full = np.concatenate(([max(0, 1 - lam_vec.sum())], lam_vec))
    calibrated_probs = []
    correct_count = 0
    
    for (orig_prob, aug_probs), true_label in zip(val_probs_vecs, val_labels):
        cal_prob = adaptive_tta_mix(orig_prob, aug_probs, lam_full)
        calibrated_probs.append(cal_prob.detach().numpy())
        
        # Track accuracy
        if cal_prob.argmax().item() == true_label:
            correct_count += 1
    
    accuracy = correct_count / len(val_labels)
    ece = compute_ece(calibrated_probs, np.array(val_labels), n_bins=15)
    
    # Combined objective: minimize ECE while maximizing accuracy
    # Weight ECE more heavily but allow accuracy to influence
    return ece - 0.5 * accuracy  # Lower is better

# Optimize with multiple restarts to find better solutions
best_loss = float('inf')
best_lam = None

init_configs = [
    np.array([0.05, 0.05, 0.05, 0.05]),
    np.array([0.1, 0.1, 0.1, 0.1]),
    np.array([0.15, 0.15, 0.15, 0.15]),
    np.array([0.2, 0.1, 0.05, 0.05]),
]

for init_lam in init_configs:
    res = minimize(combined_loss_obj, init_lam, method='Nelder-Mead', 
                   options={'maxiter': 200, 'xatol': 1e-4})
    if res.fun < best_loss:
        best_loss = res.fun
        best_lam = np.concatenate(([max(0, 1 - res.x.sum())], res.x))

print('Optimized Lambda:', best_lam)

# -- Apply to test images --
testset = torch.utils.data.Subset(testset, range(1000, 2000))
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

y_true = []
y_orig = []
y_cal = []
all_cal_probs = []

for img, label in tqdm(test_loader, desc='Adaptive-TTA Test'):
    orig_prob, aug_probs = get_augmented_probs(model, img[0], AUG_TYPES, n_aug=5)
    cal_prob = adaptive_tta_mix(orig_prob, aug_probs, best_lam)
    y_true.append(label.item())
    y_orig.append(orig_prob.argmax().item())
    y_cal.append(cal_prob.argmax().item())
    all_cal_probs.append(cal_prob.detach().numpy())

y_true = np.array(y_true)
y_orig = np.array(y_orig)
y_cal = np.array(y_cal)

orig_acc = np.mean(y_true == y_orig)
cal_acc = np.mean(y_true == y_cal)
cal_ece = compute_ece(all_cal_probs, y_true, n_bins=15)

# Detailed accuracy analysis
correct_to_incorrect = np.sum((y_true == y_orig) & (y_true != y_cal))
incorrect_to_correct = np.sum((y_true != y_orig) & (y_true == y_cal))
total_correct_initially = np.sum(y_true == y_orig)
total_incorrect_initially = np.sum(y_true != y_orig)

print(f'\nOriginal Accuracy: {orig_acc:.4f}, Adaptive-TTA Accuracy: {cal_acc:.4f}, Adaptive-TTA ECE: {cal_ece:.4f}')
print(f"Total images correct initially: {total_correct_initially}")
print(f"Total images incorrect initially: {total_incorrect_initially}")
print(f"Of images correct initially, # that became incorrect after TTA: {correct_to_incorrect}")
print(f"Of images incorrect initially, # that became correct after TTA: {incorrect_to_correct}")
print(f"Net accuracy gain: {incorrect_to_correct - correct_to_incorrect}")
