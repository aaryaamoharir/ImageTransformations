import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Correct way to load the model you specified
print("Loading pre-trained CIFAR-10 ResNet-56 model...")
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
model.eval()  # Set the model to evaluation mode

# Freeze the model's parameters so they are not updated
for param in model.parameters():
    param.requires_grad = False

# CIFAR-10 data normalization values
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

# Correct data transforms with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

# Define a separate transform for the patch training data
# This transform should only apply ToTensor(), as the corruption will be applied later
transform_train_patch = transforms.Compose([
    transforms.ToTensor(),
])

# Corrected normalize and denormalize functions using broadcasting
def normalize(tensor):
    mean = torch.tensor(cifar10_mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(cifar10_std, device=tensor.device).view(3, 1, 1)
    return (tensor - mean) / std

def denormalize(tensor):
    mean = torch.tensor(cifar10_mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(cifar10_std, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


# A simple corruption function (e.g., adding Gaussian noise)
def add_gaussian_noise(images, mean=0.0, std=0.1):
    # Ensure this function works on unnormalized images
    noise = torch.randn_like(images) * std + mean
    return (images + noise).clamp(0, 1)

# Patch parameters
patch_size = (3, 20, 20)  # C, H, W for a color patch of 20x20 pixels on a 32x32 image
patch = torch.zeros(patch_size, requires_grad=True, device=device)

# A simple function to apply the patch to a batch of images
def apply_patch(images, patch):
    patched_images = images.clone()
    h, w = images.shape[2], images.shape[3]
    p_h, p_w = patch.shape[1], patch.shape[2]
    # Apply the patch to the center of the image
    y_start, x_start = (h - p_h) // 2, (w - p_w) // 2
    patched_images[:, :, y_start:y_start+p_h, x_start:x_start+p_w] = patch
    return patched_images

# Loss function and optimizer for the patch
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01

# Image data loading for patch training (without normalization)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_patch)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Training loop
num_epochs = 40
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Apply the current patch to the images
        patched_images = apply_patch(images, patch)

        # Now, normalize the patched images before feeding them to the model
        normalized_patched_images = normalize(patched_images)

        # Forward pass and loss calculation
        outputs = model(normalized_patched_images)
        loss = criterion(outputs, labels)

        # Zero gradients of the patch
        if patch.grad is not None:
            patch.grad.zero_()

        # Backward pass to compute gradients of the loss w.r.t the patch
        loss.backward()

        # Reversed FGSM update step
        with torch.no_grad():
            patch.data.sub_(learning_rate * torch.sign(patch.grad.data))
            # Clip the patch values to the valid image range [0, 1]
            patch.data.clamp_(0, 1)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete. The angelic patch has been generated.")

# Create a test set and loader with normalization
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluation function
def evaluate(data_loader, patch=None, corruption=None):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Apply corruption to images that have been converted to [0,1] range
            if corruption:
                images_unnorm = denormalize(images.clone()) # Denormalize to apply corruption
                images_unnorm = corruption(images_unnorm)
                images = normalize(images_unnorm)

            # Apply patch if specified
            if patch is not None:
                images_unnorm = denormalize(images.clone())
                images_unnorm = apply_patch(images_unnorm, patch)
                images = normalize(images_unnorm)
                
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Evaluate performance on clean and corrupted images with and without the patch
print("\nEvaluation:")
clean_acc = evaluate(test_loader)
print(f"Accuracy on clean images (no patch): {clean_acc:.2f}%")

corrupted_acc_no_patch = evaluate(test_loader, corruption=add_gaussian_noise)
print(f"Accuracy on gaussian noise  corrupted images (no patch): {corrupted_acc_no_patch:.2f}%")

corrupted_acc_with_patch = evaluate(test_loader, patch=patch, corruption=add_gaussian_noise)
print(f"Accuracy on corrupted images (with patch): {corrupted_acc_with_patch:.2f}%")
