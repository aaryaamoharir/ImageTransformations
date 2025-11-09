import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE = 8
PATCH_LOCATION = (0, 0)  # top-left
EPOCHS = 5
BATCH_SIZE = 64
EPSILON = 0.01  # step size for reverse FGSM

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
model.eval()  # Don't train the classifier itself

# --- Evaluate Function ---
def evaluate(model, dataloader, device, patch=None, patch_loc=(0,0)):
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            if patch is not None:
                imgs = apply_patch_batch(imgs, patch, patch_loc)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# --- Patch Application Function ---
def apply_patch_batch(images, patch, loc):
    patched = images.clone()
    r, c = loc
    patched[:, :, r:r+PATCH_SIZE, c:c+PATCH_SIZE] = patch
    return patched

orig_acc = evaluate(model, testloader, device)
print(f"Original model accuracy (no patch): {orig_acc:.2f}%")

# --- Patch Initialization ---
patch = torch.rand(1, 3, PATCH_SIZE, PATCH_SIZE, device=device, requires_grad=True)

# --- Angelic Patch Training (Reverse FGSM) ---
criterion = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)
        # Patch is applied to all images at fixed location
        patched_imgs = apply_patch_batch(imgs, patch, PATCH_LOCATION)
        outputs = model(patched_imgs)
        loss = criterion(outputs, labels)
        # Compute gradient w.r.t patch
        grad = torch.autograd.grad(loss, patch, retain_graph=False)[0]
        # Reverse FGSM step: subtract sign of grad to minimize loss
        with torch.no_grad():
            patch -= EPSILON * grad.sign()
            patch.clamp_(0, 1)  # keep patch in valid image range
        patch.requires_grad_()
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss.item():.4f}")

patched_acc = evaluate(model, testloader, device, patch=patch, patch_loc=PATCH_LOCATION)
print(f"Accuracy with universal angelic patch: {patched_acc:.2f}%")

