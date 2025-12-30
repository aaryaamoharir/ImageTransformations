import torch 
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

# ======= CIFAR-C DATASET CLASS =======
class CIFARC_Dataset(Dataset):
    """Dataset class for CIFAR-10-C or CIFAR-100-C corruptions."""
    
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
        img = self.data[idx]
        label = self.labels[idx]
        # Return PIL image without transform for TTA
        img = Image.fromarray(img)
        return img, label

# ======= CONFIG =======
USE_DATASET = "cifar100c"
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

# ======= BASE TRANSFORM =======
base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# ======= TEST-TIME AUGMENTATION STRATEGIES =======
# Strategy 1: Gentle geometric transforms (works for most corruptions)
tta_transforms_gentle = [
    T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    T.Compose([
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    T.Compose([
        T.RandomCrop(32, padding=4, padding_mode='reflect'),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    T.Compose([
        T.RandomHorizontalFlip(p=1.0),
        T.RandomCrop(32, padding=4, padding_mode='reflect'),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
]

# Strategy 2: Multi-scale (good for blur/pixelate)
tta_transforms_multiscale = [
    T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    T.Compose([
        T.Resize(36),
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    T.Compose([
        T.Resize(40),
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    T.Compose([
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
]

# Strategy 3: Color adjustments (good for brightness/contrast/weather)
tta_transforms_color = [
    T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    T.Compose([
        T.ColorJitter(brightness=0.2),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    T.Compose([
        T.ColorJitter(contrast=0.2),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    T.Compose([
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
]

# Map corruption types to best TTA strategy
corruption_to_strategy = {
    'defocus_blur': 'multiscale',
    'glass_blur': 'gentle',
    'motion_blur': 'multiscale',
    'zoom_blur': 'multiscale',
    'snow': 'color',
    'frost': 'color',
    'fog': 'color',
    'brightness': 'color',
    'contrast': 'color',
    'elastic_transform': 'gentle',
    'pixelate': 'multiscale',
    'jpeg_compression': 'gentle',
}

def get_tta_transforms(corruption_type):
    """Get appropriate TTA transforms for corruption type."""
    strategy = corruption_to_strategy.get(corruption_type, 'gentle')
    
    if strategy == 'multiscale':
        return tta_transforms_multiscale
    elif strategy == 'color':
        return tta_transforms_color
    else:
        return tta_transforms_gentle

# ======= MODEL =======
model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    model_name,
    pretrained=True
).eval().to(device)

# ======= CUSTOM COLLATE FUNCTION =======
def collate_pil_images(batch):
    """Custom collate function to handle PIL images in DataLoader."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels

# ======= TEST-TIME AUGMENTATION PREDICTION =======
def predict_with_tta(model, image_pil, tta_transforms):
    """
    Make prediction using Test-Time Augmentation.
    NO parameter updates - just ensemble predictions!
    
    Args:
        model: Pretrained model (frozen)
        image_pil: PIL Image
        tta_transforms: List of transform compositions
    
    Returns:
        Averaged probability distribution
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            # Apply transform
            img_tensor = transform(image_pil).unsqueeze(0).to(device)
            
            # Get prediction
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs)
    
    # Average probabilities across all augmentations
    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs

# ======= EVALUATION FUNCTIONS =======
def evaluate_baseline(model, dataloader):
    """Baseline: no augmentation, single prediction."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images_pil, labels in dataloader:
            for i in range(len(images_pil)):
                img_pil = images_pil[i]
                label = labels[i]
                
                # Single prediction
                img_tensor = base_transform(img_pil).unsqueeze(0).to(device)
                output = model(img_tensor)
                pred = output.argmax(dim=1)
                
                correct += (pred.cpu() == label).item()
                total += 1
    
    return 100 * correct / total

def evaluate_with_tta(model, dataloader, corruption_type):
    """TTA: ensemble multiple augmented predictions."""
    model.eval()
    correct = 0
    total = 0
    
    # Get appropriate TTA transforms for this corruption
    tta_transforms = get_tta_transforms(corruption_type)
    
    for images_pil, labels in dataloader:
        for i in range(len(images_pil)):
            img_pil = images_pil[i]
            label = labels[i]
            
            # TTA prediction (averaged across augmentations)
            avg_probs = predict_with_tta(model, img_pil, tta_transforms)
            pred = avg_probs.argmax(dim=1)
            
            correct += (pred.cpu() == label).item()
            total += 1
    
    return 100 * correct / total

# ======= MAIN EVALUATION LOOP =======
results = {
    'baseline': {},
    'tta': {}
}

print("="*80)
print("TEST-TIME AUGMENTATION (TTA) - ZERO PARAMETER UPDATES")
print("="*80)
print("Method: Ensemble predictions across augmented views of each test image")
print("No model training or parameter adaptation required!")
print("="*80)

for corr in corruptions:
    print(f"\n{'='*80}")
    print(f"Corruption: {corr} (Strategy: {corruption_to_strategy[corr]})")
    print(f"{'='*80}")
    
    results['baseline'][corr] = []
    results['tta'][corr] = []
    
    for severity in range(1, 6):
        print(f"\n--- Severity: {severity} ---")
        
        # Load dataset (without transform)
        testset = CIFARC_Dataset(
            root=cifarc_root,
            corruption=corr,
            severity=severity,
            transform=None  # We'll apply transforms manually
        )
        testloader = DataLoader(
            testset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            collate_fn=collate_pil_images  # Custom collate for PIL images
        )
        
        # Baseline (single prediction)
        print("Evaluating baseline (single prediction)...")
        baseline_acc = evaluate_baseline(model, testloader)
        results['baseline'][corr].append(baseline_acc)
        print(f"Baseline Accuracy: {baseline_acc:.2f}%")
        
        # TTA (ensemble predictions)
        print("Evaluating TTA (ensemble of 4 augmentations)...")
        tta_acc = evaluate_with_tta(model, testloader, corr)
        results['tta'][corr].append(tta_acc)
        improvement = tta_acc - baseline_acc
        print(f"TTA Accuracy: {tta_acc:.2f}% ({improvement:+.2f}%)")

# ======= FINAL SUMMARY =======
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

for corr in corruptions:
    print(f"\n{corr}:")
    for sev in range(5):
        baseline = results['baseline'][corr][sev]
        tta = results['tta'][corr][sev]
        improvement = tta - baseline
        print(f"  Severity {sev+1}: {baseline:.2f}% → {tta:.2f}% ({improvement:+.2f}%)")
    
    avg_baseline = np.mean(results['baseline'][corr])
    avg_tta = np.mean(results['tta'][corr])
    avg_improvement = avg_tta - avg_baseline
    print(f"  Average: {avg_baseline:.2f}% → {avg_tta:.2f}% ({avg_improvement:+.2f}%)")

# Overall statistics
all_baseline = [acc for corr in corruptions for acc in results['baseline'][corr]]
all_tta = [acc for corr in corruptions for acc in results['tta'][corr]]

print(f"\n{'='*80}")
print(f"OVERALL STATISTICS")
print(f"{'='*80}")
print(f"Mean Baseline Accuracy: {np.mean(all_baseline):.2f}%")
print(f"Mean TTA Accuracy: {np.mean(all_tta):.2f}%")
print(f"Mean Improvement: {np.mean(all_tta) - np.mean(all_baseline):+.2f}%")
print(f"\n✓ Zero parameter updates - pretrained model used as-is!")
print(f"✓ Fast inference - 4 forward passes per image")
print(f"✓ No training required - works with any pretrained model")
