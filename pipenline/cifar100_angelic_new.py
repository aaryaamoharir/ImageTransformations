import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

# --- Configuration and Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data Loading ---
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

# --- Confidence-Boosting Patch Generation ---

# This function calculates the entropy of the model's output probabilities.
# The goal is now to MINIMIZE this entropy, thereby making the model more confident.
def entropy_loss(outputs):
    probs = F.softmax(outputs, dim=1)
    log_probs = F.log_softmax(outputs, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean()

# Utility to apply a patch to images at the top-left corner
def apply_patch(images, patch, patch_size):
    patched = images.clone()
    patched[:, :, :patch_size, :patch_size] += patch[None, :, :, :]
    return torch.clamp(patched, 0, 1)

# Patch parameters and initialization
patch_size = 8
epsilon = 0.1
patch = torch.randn(3, patch_size, patch_size, requires_grad=True, device=device)

def evaluate_accuracy(model, dataloader, patch=None, patch_size=None):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            if patch is not None:
                images = apply_patch(images, patch, patch_size)
            
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# --- Main Execution ---
if __name__ == "__main__":
    # --- 1. Initial Evaluation (before patch optimization) ---
    print("\nEvaluating initial accuracy...")
    initial_accuracy = evaluate_accuracy(model, testloader)
    print(f'Initial Unpatched Test Accuracy: {initial_accuracy:.4f}')

    # --- 2. Patch Optimization Loop ---
    print("\nStarting confidence-boosting patch optimization loop...")
    for epoch in range(10):
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            
            patched_images = apply_patch(images, patch, patch_size)

            outputs = model(patched_images)
            
            loss = entropy_loss(outputs)
            
            patch.grad = None
            loss.backward()

            with torch.no_grad():
                patch -= epsilon * patch.grad.sign()
                patch.clamp_(-1, 1)
                
    # --- 3. Final Evaluation (after patch optimization) ---
    print("\nEvaluating model with the optimized patch...")
    patched_accuracy = evaluate_accuracy(model, testloader, patch, patch_size)
    print(f'Final Patched Test Accuracy: {patched_accuracy:.4f}')
