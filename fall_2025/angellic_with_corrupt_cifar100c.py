import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
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

#CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
#CIFAR10_STD = (0.2023, 0.1994, 0.2010)

#transform = T.Compose([
#    T.ToTensor(),
#    T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
#])

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# ---------------- CIFAR-10-C Dataset ----------------
class CIFAR10C(Dataset):
    def __init__(self, root, corruption, severity=1, transform=None):
        """
        root: path to CIFAR-10-C directory (contains *.npy files)
        corruption: e.g. 'gaussian_noise', 'fog', 'brightness', ...
        severity: 1..5
        """
        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        # Each corruption file is shape (50000, 32, 32, 3)
        # 50000 images for 5 severities -> 10000 per severity
        imgs_path = os.path.join(root, f"{corruption}.npy")
        labels_path = os.path.join(root, "labels.npy")

        self.data = np.load(imgs_path)    # shape (50000, 32, 32, 3)
        self.labels = np.load(labels_path)  # shape (50000,)

        assert 1 <= severity <= 5
        n_per_sev = 10000
        start = (severity - 1) * n_per_sev
        end = severity * n_per_sev

        self.data = self.data[start:end]
        self.labels = self.labels[start:end]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]  # H,W,3, uint8
        label = int(self.labels[idx])

        # convert to PIL-like tensor
        img = img.astype(np.uint8)
        img = T.functional.to_pil_image(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

# -------------- Choose corruption/severity -------------

cifar100c_root = "/home/diversity_project/aaryaa/attacks/Cifar-100/CIFAR-100-C"

#"/home/diversity_project/aaryaa/attacks/Cifar-10/cifar_npy_files"  # adjust

corruption_name = "fog"    # e.g. 'fog', 'brightness', 'defocus_blur', etc.
severity = 3               # 1..5

trainset = CIFAR10C(
    root=cifar100c_root,
    corruption=corruption_name,
    severity=severity,
    transform=transform
)
testset = CIFAR10C(
    root=cifar100c_root,
    corruption=corruption_name,
    severity=severity,
    transform=transform
)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model ---
model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    'cifar100_resnet56',
    pretrained=True
).eval().to(device)

criterion = nn.CrossEntropyLoss()

# --- Adet-like operator for classification (still random transforms) ---

def adet_operator(imgs, patch, scale_min=SCALE_MIN, scale_max=SCALE_MAX):
    """
    imgs: [B,3,H,W], patch: [1,3,Ph,Pw]
    For each image:
      - sample random scale
      - sample random location
      - apply scaled patch at that location
    (Note: CIFAR-10-C already contains corruption; we don't add extra δ here.)
    """
    B, C, H, W = imgs.shape
    patched_imgs = imgs.clone()

    for i in range(B):
        img = imgs[i]

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

        img_patched = img.clone()
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

# --- Training Loop: E over Adet, reverse FGSM ---
for epoch in range(EPOCHS):
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)

        patch.requires_grad_(True)

        losses = []
        for _ in range(N_EXPECT):
            patched_imgs = adet_operator(imgs, patch)
            outputs = model(patched_imgs)
            loss = criterion(outputs, labels)
            losses.append(loss)

        loss_batch = torch.stack(losses).mean()

        grad = torch.autograd.grad(loss_batch, patch)[0]

        with torch.no_grad():
            patch -= EPSILON * grad.sign()
            patch.clamp_(0, 1)

        patch.requires_grad_(False)

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss_batch.item():.4f}")

# --- Evaluation on CIFAR-10-C (same corruption/severity) ---
orig_acc = evaluate(model, testloader, device)
patched_acc = evaluate(model, testloader, device, patch=patch, patch_loc=(0,0))
print(f"Original accuracy (no patch): {orig_acc:.2f}%")
print(f"Accuracy with learned angelic patch: {patched_acc:.2f}%")
