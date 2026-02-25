import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import entropy
from PIL import Image
import os

# ======= CIFAR-C DATASET CLASS =======
class CIFARC_Dataset(Dataset):
    """Dataset class for CIFAR-10-C or CIFAR-100-C corruptions."""
    
    def __init__(self, root, corruption, severity, transform=None):
        """
        Args:
            root: Root directory containing CIFAR-C data
            corruption: Type of corruption (e.g., 'gaussian_noise')
            severity: Corruption severity level (1-5)
            transform: Optional transform to apply
        """
        self.transform = transform
        
        # Load corrupted images
        corruption_file = os.path.join(root, f"{corruption}.npy")
        labels_file = os.path.join(root, "labels.npy")
        
        # Load data
        all_images = np.load(corruption_file)
        all_labels = np.load(labels_file)
        
        # Extract images for this severity level
        # CIFAR-C stores 10,000 images per severity, ordered by severity
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        
        self.data = all_images[start_idx:end_idx]
        self.labels = all_labels[start_idx:end_idx]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label

# ======= CONFIG =======
USE_DATASET = "cifar100c"
ENTROPY_THRESHOLD = 7.46
BATCH_SIZE = 128
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

# ======= TRANSFORMS =======
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# Adaptive transforms (applied when entropy > threshold)
# These are more effective than simple rotation for corrupted images
adaptive_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.RandomPerspective(distortion_scale=0.2, p=0.5),
])

# ======= SHANNON ENTROPY CALCULATION =======
def compute_shannon_entropy(image_tensor):
    """
    Compute Shannon entropy of an image tensor.
    Args:
        image_tensor: torch.Tensor of shape (C, H, W) in range [0, 1]
    Returns:
        float: Shannon entropy value
    """
    # Convert to numpy and flatten
    img_np = image_tensor.cpu().numpy()
    
    # Compute histogram across all channels
    hist, _ = np.histogram(img_np.flatten(), bins=256, range=(0, 1), density=True)
    
    # Remove zero entries
    hist = hist[hist > 0]
    
    # Compute Shannon entropy
    return entropy(hist, base=2)

# ======= MODEL =======
model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    model_name,
    pretrained=True
).eval().to(device)

# ======= EVALUATION FUNCTIONS =======
def evaluate_with_entropy_adaptation(model, dataloader, apply_transforms=False):
    """
    Evaluate model with optional entropy-based adaptive transforms.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with images
        apply_transforms: If True, apply adaptive transforms for high-entropy images
    
    Returns:
        accuracy, high_entropy_count, total_count
    """
    correct = 0
    total = 0
    high_entropy_count = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            batch_size = images.size(0)
            
            for i in range(batch_size):
                img = images[i]  # Shape: (C, H, W), normalized
                label = labels[i]
                
                # Denormalize to compute entropy
                img_denorm = img.clone()
                for c in range(3):
                    img_denorm[c] = img_denorm[c] * CIFAR_STD[c] + CIFAR_MEAN[c]
                img_denorm = torch.clamp(img_denorm, 0, 1)
                
                # Compute Shannon entropy
                img_entropy = compute_shannon_entropy(img_denorm)
                
                # Apply adaptive transform if entropy exceeds threshold
                if apply_transforms and img_entropy > ENTROPY_THRESHOLD:
                    high_entropy_count += 1
                    
                    # Convert back to PIL for transforms
                    img_pil = T.ToPILImage()(img_denorm)
                    
                    # Apply adaptive transforms
                    img_transformed = adaptive_transforms(img_pil)
                    
                    # Convert back to tensor and normalize
                    img_processed = transform(img_transformed).unsqueeze(0).to(device)
                else:
                    # Use original image
                    img_processed = img.unsqueeze(0).to(device)
                
                # Get prediction
                output = model(img_processed)
                pred = output.argmax(dim=1)
                
                correct += (pred == label.to(device)).item()
                total += 1
    
    accuracy = 100 * correct / total
    return accuracy, high_entropy_count, total

# ======= MAIN EVALUATION LOOP =======
results = {}

for corr in corruptions:
    print(f"\n{'='*60}")
    print(f"Corruption: {corr}")
    print(f"{'='*60}")
    
    corr_results = {
        'initial_acc': [],
        'adapted_acc': [],
        'high_entropy_ratio': []
    }
    
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
        
        # Initial accuracy (no adaptive transforms)
        print("Evaluating initial accuracy...")
        initial_acc, _, total = evaluate_with_entropy_adaptation(
            model, testloader, apply_transforms=False
        )
        
        # Adapted accuracy (with entropy-based transforms)
        print("Evaluating with adaptive transforms...")
        adapted_acc, high_entropy_count, total = evaluate_with_entropy_adaptation(
            model, testloader, apply_transforms=True
        )
        
        high_entropy_ratio = 100 * high_entropy_count / total
        
        # Store results
        corr_results['initial_acc'].append(initial_acc)
        corr_results['adapted_acc'].append(adapted_acc)
        corr_results['high_entropy_ratio'].append(high_entropy_ratio)
        
        # Print results
        print(f"Initial Accuracy: {initial_acc:.2f}%")
        print(f"Adapted Accuracy: {adapted_acc:.2f}%")
        print(f"Improvement: {adapted_acc - initial_acc:+.2f}%")
        print(f"High-entropy images: {high_entropy_count}/{total} ({high_entropy_ratio:.1f}%)")
    
    results[corr] = corr_results

# ======= FINAL SUMMARY =======
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

for corr in corruptions:
    print(f"\n{corr}:")
    for sev in range(5):
        initial = results[corr]['initial_acc'][sev]
        adapted = results[corr]['adapted_acc'][sev]
        improvement = adapted - initial
        print(f"  Severity {sev+1}: {initial:.2f}% → {adapted:.2f}% ({improvement:+.2f}%)")
    
    avg_initial = np.mean(results[corr]['initial_acc'])
    avg_adapted = np.mean(results[corr]['adapted_acc'])
    avg_improvement = avg_adapted - avg_initial
    print(f"  Average: {avg_initial:.2f}% → {avg_adapted:.2f}% ({avg_improvement:+.2f}%)")

# Overall statistics
all_initial = [acc for corr in corruptions for acc in results[corr]['initial_acc']]
all_adapted = [acc for corr in corruptions for acc in results[corr]['adapted_acc']]

print(f"\n{'='*60}")
print(f"OVERALL STATISTICS")
print(f"{'='*60}")
print(f"Mean Initial Accuracy: {np.mean(all_initial):.2f}%")
print(f"Mean Adapted Accuracy: {np.mean(all_adapted):.2f}%")
print(f"Mean Improvement: {np.mean(all_adapted) - np.mean(all_initial):+.2f}%")
