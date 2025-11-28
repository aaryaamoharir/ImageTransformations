import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import random

# --- Hyperparameters and Device ---
PATCH_SIZE = 8
EPOCHS = 5
BATCH_SIZE = 64
EPSILON = 0.01        # step size for reverse FGSM
N_EXPECT = 3          # expectation samples per image
SCALE_MIN = 0.8       # Min random scaling factor
SCALE_MAX = 1.2       # Max random scaling factor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Preparation ---
#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465),
#                         (0.2023, 0.1994, 0.2010))
#])

#trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
#testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model ---
#model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
#model.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# --- Model Loading ---
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet56', pretrained=True).to(device)
model.eval()
criterion = nn.CrossEntropyLoss()

# --- Corruption (Δ) ---
def add_gaussian_noise(img, std=0.05):
    return img + torch.randn_like(img) * std

def gaussian_blur_img(img, kernel_size=3):
    # img: C,H,W
    return TF.gaussian_blur(img, kernel_size=kernel_size)

CORRUPTIONS = [
    lambda x: x,                          # identity (no corruption)
    lambda x: add_gaussian_noise(x, 0.05),
    lambda x: gaussian_blur_img(x, 3),
]

# --- Adet-like operator for classification ---
def adet_operator(imgs, patch, scale_min=SCALE_MIN, scale_max=SCALE_MAX):
    """
    imgs: [B,3,H,W], patch: [1,3,Ph,Pw]
    For each image:
      - sample corruption δ in Δ
      - sample random scale
      - sample random location
      - apply scaled patch at that location
    Returns patched images.
    """
    B, C, H, W = imgs.shape
    patched_imgs = imgs.clone()

    for i in range(B):
        img = imgs[i]

        # Sample a corruption δ from Δ and apply
        corr_fn = random.choice(CORRUPTIONS)
        img_corr = corr_fn(img)

        # Sample random scale
        scale = random.uniform(scale_min, scale_max)
        new_size = max(1, int(PATCH_SIZE * scale))

        scaled_patch = F.interpolate(
            patch, size=(new_size, new_size),
            mode='bilinear', align_corners=False
        )[0]  # [3, new_size, new_size]

        # Random location ensuring patch fits
        r_max = H - new_size
        c_max = W - new_size
        r = random.randint(0, max(0, r_max))
        c = random.randint(0, max(0, c_max))

        img_patched = img_corr.clone()
        img_patched[:, r:r+new_size, c:c+new_size] = scaled_patch

        patched_imgs[i] = img_patched

    return patched_imgs

# --- Evaluation (no randomness for patch) ---
def apply_fixed_patch(imgs, patch, loc=(0,0), size=PATCH_SIZE):
    B, C, H, W = imgs.shape
    r, c = loc
    new_size = size
    scaled_patch = F.interpolate(
        patch, size=(new_size, new_size),
        mode='bilinear', align_corners=False
    )
    out = imgs.clone()
    # Clamp location if needed
    r = min(r, H - new_size)
    c = min(c, W - new_size)
    out[:, :, r:r+new_size, c:c+new_size] = scaled_patch
    return out

def evaluate(model, dataloader, device, patch=None, patch_loc=(0,0)):
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            if patch is not None:
                imgs = apply_fixed_patch(imgs, patch, patch_loc, size=PATCH_SIZE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# --- Patch Init ---
patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE, device=device, requires_grad=True)

# --- Training Loop: E over transforms/corruptions, reverse FGSM ---
for epoch in range(EPOCHS):
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)

        patch.requires_grad_(True)

        # Explicit expectation over N_EXPECT samples
        losses = []
        for _ in range(N_EXPECT):
            patched_imgs = adet_operator(imgs, patch)     # Adet + Δ
            outputs = model(patched_imgs)
            loss = criterion(outputs, labels)
            losses.append(loss)

        loss_batch = torch.stack(losses).mean()

        grad = torch.autograd.grad(loss_batch, patch)[0]

        with torch.no_grad():
            # Reverse FGSM step (angelic): move patch to MINIMIZE loss
            patch -= EPSILON * grad.sign()
            patch.clamp_(0, 1)

        patch.requires_grad_(False)

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss_batch.item():.4f}")

# --- Evaluation ---
orig_acc = evaluate(model, testloader, device)
patched_acc = evaluate(model, testloader, device, patch=patch, patch_loc=(0,0))
print(f"Original accuracy (no patch): {orig_acc:.2f}%")
print(f"Accuracy with learned angelic patch: {patched_acc:.2f}%")

