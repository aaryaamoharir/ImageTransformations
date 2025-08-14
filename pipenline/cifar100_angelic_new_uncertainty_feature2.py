import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np

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

# --- Restorative Patch Generation ---
def target_loss(outputs, labels):
    return F.cross_entropy(outputs, labels)

def apply_patch(images, patch, patch_size):
    patched = images.clone()
    patched[:, :, :patch_size, :patch_size] += patch[None, :, :, :]
    return torch.clamp(patched, 0, 1)

patch_size = 8
epsilon = 0.1
patch = torch.randn(3, patch_size, patch_size, requires_grad=True, device=device)

def calculate_shannon_entropy(outputs):
    probs = F.softmax(outputs, dim=1)
    log_probs = F.log_softmax(outputs, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

def analyze_and_restore_uncertain_images(model, dataloader, patch, patch_size, entropy_threshold):
    total_images_processed = 0
    total_uncertain_images = 0
    total_uncertain_correct = 0
    total_uncertain_incorrect = 0
    total_uncertain_patched_correct = 0
    total_uncertain_patched_incorrect = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Analyzing and Restoring"):
            images, labels = images.to(device), labels.to(device)
            
            outputs_initial = model(images)
            entropies = calculate_shannon_entropy(outputs_initial)
            
            uncertain_indices = torch.where(entropies > entropy_threshold)[0]
            
            if uncertain_indices.size(0) > 0:
                uncertain_images = images[uncertain_indices]
                uncertain_labels = labels[uncertain_indices]
                
                _, initial_preds = outputs_initial[uncertain_indices].max(1)
                total_uncertain_correct += (initial_preds == uncertain_labels).sum().item()
                total_uncertain_incorrect += (initial_preds != uncertain_labels).sum().item()
                
                patched_images = apply_patch(uncertain_images, patch, patch_size)
                
                outputs_patched = model(patched_images)
                _, patched_preds = outputs_patched.max(1)
                
                total_uncertain_patched_correct += (patched_preds == uncertain_labels).sum().item()
                total_uncertain_patched_incorrect += (patched_preds != uncertain_labels).sum().item()

            total_images_processed += images.size(0)
            total_uncertain_images += uncertain_indices.size(0)
    
    print(f"\n--- Analysis for Images with Entropy > {entropy_threshold:.2f} ---")
    print(f"Total images processed: {total_images_processed}")
    print(f"Total uncertain images identified: {total_uncertain_images}")
    
    if total_uncertain_images > 0:
        print(f"\nAccuracy of uncertain images (before patching): {total_uncertain_correct / total_uncertain_images:.4f}")
        print(f"Accuracy of uncertain images (after patching): {total_uncertain_patched_correct / total_uncertain_images:.4f}")
    else:
        print("\nNo images identified as uncertain above the threshold.")
    
    print("\n--- Correct/Incorrect Counts for the Uncertain Subset ---")
    print(f"Correctly classified before patching: {total_uncertain_correct}")
    print(f"Incorrectly classified before patching: {total_uncertain_incorrect}")
    print(f"Correctly classified after patching: {total_uncertain_patched_correct}")
    print(f"Incorrectly classified after patching: {total_uncertain_patched_incorrect}")

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

if __name__ == "__main__":
    # --- 1. Patch Optimization Loop ---
    print("\nStarting restorative patch optimization loop...")
    for epoch in range(10):
        # FIX: Move labels to the correct device
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            
            patched_images = apply_patch(images, patch, patch_size)
            outputs = model(patched_images)
            loss = target_loss(outputs, labels)
            
            patch.grad = None
            loss.backward()

            with torch.no_grad():
                patch -= epsilon * patch.grad.sign()
                patch.clamp_(-1, 1)

    # --- 2. Final Evaluation (after patch optimization) ---
    SHANNON_ENTROPY_THRESHOLD = 1.16
    print("\n--- Final Evaluation ---")
    
    analyze_and_restore_uncertain_images(model, testloader, patch, patch_size, SHANNON_ENTROPY_THRESHOLD)
    
    patched_overall_accuracy = evaluate_accuracy(model, testloader, patch, patch_size)
    print(f'\nFinal Overall Patched Test Accuracy: {patched_overall_accuracy:.4f}')
