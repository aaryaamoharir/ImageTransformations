import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import random

# --- Hyperparameters and Device ---
PATCH_SIZE = 8
PATCH_LOCATION = (0, 0)  # top-left
EPOCHS = 5
BATCH_SIZE = 64
EPSILON = 0.01  # step size for reverse FGSM
SCALE_MIN = 0.8  # Min random scaling factor
SCALE_MAX = 1.2  # Max random scaling factor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Preparation ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
model.eval()

# --- Patch Transformation Function ---
def apply_transformed_patch(images, patch, loc, scale_min=SCALE_MIN, scale_max=SCALE_MAX):
    scale = random.uniform(scale_min, scale_max)
    new_size = max(1, int(PATCH_SIZE * scale))
    scaled_patch = F.interpolate(patch, size=(new_size, new_size), mode='bilinear', align_corners=False)
    patched = images.clone()
    r, c = loc
    # Make sure patch fits within image
    r = min(r, images.size(2) - new_size)
    c = min(c, images.size(3) - new_size)
    patched[:, :, r:r+new_size, c:c+new_size] = scaled_patch
    return patched

# --- Evaluation Function ---
def evaluate(model, dataloader, device, patch=None, patch_loc=(0,0)):
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            if patch is not None:
                imgs = apply_transformed_patch(imgs, patch, patch_loc, scale_min=1.0, scale_max=1.0)  # No scale for eval
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# --- Patch Init ---
patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE, device=device, requires_grad=True)

criterion = nn.CrossEntropyLoss()

# --- Training Loop: transformation then reverse FGSM ---
for epoch in range(EPOCHS):
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)
        patch.requires_grad = True
        patched_imgs = apply_transformed_patch(imgs, patch, PATCH_LOCATION)  # random scale each time
        outputs = model(patched_imgs)
        loss = criterion(outputs, labels)
        grad = torch.autograd.grad(loss, patch)[0]
        with torch.no_grad():
            patch -= EPSILON * grad.sign()  # Reverse FGSM step
            patch.clamp_(0, 1)
        patch.requires_grad_()
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss.item():.4f}")

# --- Evaluation ---
orig_acc = evaluate(model, testloader, device)
patched_acc = evaluate(model, testloader, device, patch=patch, patch_loc=PATCH_LOCATION)
print(f"Original accuracy (no patch): {orig_acc:.2f}%")
print(f"Accuracy with learned angelic patch: {patched_acc:.2f}%")

