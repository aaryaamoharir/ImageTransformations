import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained CIFAR-10 ResNet-56 model
print("Loading pre-trained CIFAR-10 ResNet-56 model...")
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
model.eval()  # Set the model to evaluation mode

# Freeze the model's parameters so they are not updated
for param in model.parameters():
    param.requires_grad = False

# CIFAR-10 data normalization values
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

# Corrected normalize and denormalize functions using broadcasting
def normalize(tensor):
    mean = torch.tensor(cifar10_mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(cifar10_std, device=tensor.device).view(3, 1, 1)
    return (tensor - mean) / std

def denormalize(tensor):
    mean = torch.tensor(cifar10_mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(cifar10_std, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean

# Corruption function (Gaussian noise)
def add_gaussian_noise(images, mean=0.0, std=0.1):
    """Apply Gaussian noise to unnormalized images [0,1]"""
    noise = torch.randn_like(images) * std + mean
    return (images + noise).clamp(0, 1)

# Additional corruption functions for testing
def add_brightness(images, factor=0.3):
    """Add brightness to unnormalized images [0,1]"""
    return (images + factor).clamp(0, 1)

def add_contrast(images, factor=1.5):
    """Modify contrast of unnormalized images [0,1]"""
    return ((images - 0.5) * factor + 0.5).clamp(0, 1)

# Patch parameters
patch_size = (3, 20, 20)  # C, H, W for a color patch of 20x20 pixels on a 32x32 image
patch = torch.zeros(patch_size, requires_grad=True, device=device)

# Function to apply the patch to a batch of images
def apply_patch(images, patch, position='center'):
    """
    Apply patch to images at specified position
    Args:
        images: batch of unnormalized images [0,1]
        patch: patch tensor [0,1]
        position: where to place the patch ('center', 'random', etc.)
    """
    patched_images = images.clone()
    h, w = images.shape[2], images.shape[3]
    p_h, p_w = patch.shape[1], patch.shape[2]
    
    if position == 'center':
        # Apply the patch to the center of the image
        y_start, x_start = (h - p_h) // 2, (w - p_w) // 2
        patched_images[:, :, y_start:y_start+p_h, x_start:x_start+p_w] = patch
    elif position == 'random':
        # Apply patch at random positions (different for each image in batch)
        for i in range(images.shape[0]):
            y_start = torch.randint(0, h - p_h + 1, (1,)).item()
            x_start = torch.randint(0, w - p_w + 1, (1,)).item()
            patched_images[i, :, y_start:y_start+p_h, x_start:x_start+p_w] = patch
    
    return patched_images

# Loss function and optimizer parameters
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01

# Data transforms for patch training (without normalization initially)
transform_train_patch = transforms.Compose([
    transforms.ToTensor(),
])

# Image data loading for patch training
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_patch)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

print("Starting angelic patch training...")
print("Goal: Minimize loss to improve model performance on corrupted images")

# Training loop for angelic patches
num_epochs = 50
corruption_aware = True  # Set to False for corruption-agnostic training

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        if corruption_aware:
            # Corruption-aware training: apply corruption during training
            # Apply patch first, then corruption
            patched_images = apply_patch(images, patch)
            corrupted_patched_images = add_gaussian_noise(patched_images)
            
            # Normalize for model input
            model_input = normalize(corrupted_patched_images)
        else:
            # Corruption-agnostic training: train without corruption
            patched_images = apply_patch(images, patch)
            model_input = normalize(patched_images)

        # Forward pass
        outputs = model(model_input)
        loss = criterion(outputs, labels)

        # Zero gradients of the patch
        if patch.grad is not None:
            patch.grad.zero_()

        # Backward pass to compute gradients of the loss w.r.t the patch
        loss.backward()

        # REVERSED FGSM update step - MINIMIZE loss (subtract gradient)
        with torch.no_grad():
            # This is the key change: we SUBTRACT the signed gradient to minimize loss
            patch.data.sub_(learning_rate * torch.sign(patch.grad.data))
            # Clip the patch values to the valid image range [0, 1]
            patch.data.clamp_(0, 1)

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

print("Angelic patch training complete!")

# Create test dataset and loader
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Enhanced evaluation function
def evaluate(data_loader, patch=None, corruption_func=None, corruption_name=""):
    """
    Evaluate model performance
    Args:
        data_loader: test data loader
        patch: angelic patch to apply (optional)
        corruption_func: corruption function to apply (optional)
        corruption_name: name for printing
    """
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Apply corruption first if specified
            if corruption_func:
                images = corruption_func(images)
            
            # Apply patch if specified
            if patch is not None:
                images = apply_patch(images, patch)
            
            # Normalize for model input
            normalized_images = normalize(images)
            
            # Forward pass
            outputs = model(normalized_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Comprehensive evaluation
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# Test on clean images
clean_acc_no_patch = evaluate(test_loader)
print(f"Clean images (no patch):                    {clean_acc_no_patch:.2f}%")

clean_acc_with_patch = evaluate(test_loader, patch=patch)
print(f"Clean images (with patch):                  {clean_acc_with_patch:.2f}%")

# Test on corrupted images
corruption_tests = [
    (add_gaussian_noise, "Gaussian noise"),
    (add_brightness, "Brightness"),
    (add_contrast, "Contrast")
]

for corruption_func, corruption_name in corruption_tests:
    print(f"\n{corruption_name} corruption:")
    
    corrupted_acc_no_patch = evaluate(test_loader, corruption_func=corruption_func)
    print(f"  No patch:     {corrupted_acc_no_patch:.2f}%")
    
    corrupted_acc_with_patch = evaluate(test_loader, patch=patch, corruption_func=corruption_func)
    print(f"  With patch:   {corrupted_acc_with_patch:.2f}%")
    
    improvement = corrupted_acc_with_patch - corrupted_acc_no_patch
    print(f"  Improvement:  {improvement:+.2f}%")

print("\n" + "="*60)

# Visualize the learned patch
print(f"\nLearned patch statistics:")
print(f"  Min value: {patch.min().item():.4f}")
print(f"  Max value: {patch.max().item():.4f}")
print(f"  Mean value: {patch.mean().item():.4f}")
print(f"  Std value: {patch.std().item():.4f}")

# Optional: Save the patch for later use
torch.save(patch.detach().cpu(), 'angelic_patch_cifar10.pth')
print("\nAngelic patch saved as 'angelic_patch_cifar10.pth'")
