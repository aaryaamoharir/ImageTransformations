import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np
from PIL import Image

# --- Configuration and Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Data Loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Use batch_size=1 for per-image analysis
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# --- Model Loading ---
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet56', pretrained=True).to(device)
model.eval()

# --- On-the-fly Patch Generation Functions ---

def apply_patch(image, patch, patch_size):
    patched = image.clone()
    patched[:, :, :patch_size, :patch_size] += patch
    return torch.clamp(patched, 0, 1)

def entropy_loss(outputs):
    probs = F.softmax(outputs, dim=1)
    log_probs = F.log_softmax(outputs, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean()

def get_restorative_patch(model, image, patch_size=8, epsilon=0.1, num_iterations=20):
    patch = torch.randn(1, 3, patch_size, patch_size, requires_grad=True, device=device)
    optimizer = optim.Adam([patch], lr=epsilon)
    
    with torch.no_grad():
        original_output = model(image)

    for i in range(num_iterations):
        patched_image = apply_patch(image, patch, patch_size)
        
        outputs = model(patched_image)
        
        # Use entropy loss, which doesn't require a ground truth label
        loss = entropy_loss(outputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            patched_output = model(patched_image)
            # You can add a convergence check here, for example, if the confidence
            # for the predicted class is high enough.
            probs = F.softmax(patched_output, dim=1)
            max_prob = torch.max(probs)
            if max_prob.item() > 0.95: # Converge if confidence is high
                break
    
    return patch.detach()

def calculate_shannon_entropy(outputs):
    probs = F.softmax(outputs, dim=1)
    log_probs = F.log_softmax(outputs, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

def evaluate_single_image(model, image, label, patch=None, patch_size=None):
    with torch.no_grad():
        if patch is not None:
            image = apply_patch(image, patch, patch_size)
        
        outputs = model(image)
        _, pred = outputs.max(1)
        is_correct = (pred.item() == label.item())
    
    return is_correct

# --- Main Execution ---
if __name__ == "__main__":
    SHANNON_ENTROPY_THRESHOLD = 1.16
    PATCH_SIZE = 8
    
    total_images_processed = 0
    total_uncertain_images = 0
    total_uncertain_correct = 0
    total_uncertain_incorrect = 0
    total_uncertain_patched_correct = 0
    total_uncertain_patched_incorrect = 0
    
    print("--- Starting per-image analysis and patching ---")
    model.eval()
    
    for image, label in tqdm(testloader, total=len(testloader), desc="Processing All Images"):
        image, label = image.to(device), label.to(device)
        
        with torch.no_grad():
            outputs_initial = model(image)
            entropy = calculate_shannon_entropy(outputs_initial)
        
        total_images_processed += 1
        
        if entropy.item() > SHANNON_ENTROPY_THRESHOLD:
            total_uncertain_images += 1
            
            is_correct_initial = evaluate_single_image(model, image, label)
            if is_correct_initial:
                total_uncertain_correct += 1
            else:
                total_uncertain_incorrect += 1
            
            restorative_patch = get_restorative_patch(model, image, patch_size=PATCH_SIZE, num_iterations=30)
            
            is_correct_after_patch = evaluate_single_image(model, image, label, patch=restorative_patch, patch_size=PATCH_SIZE)
            if is_correct_after_patch:
                total_uncertain_patched_correct += 1
            else:
                total_uncertain_patched_incorrect += 1

    print(f"\n--- Analysis for Images with Entropy > {SHANNON_ENTROPY_THRESHOLD:.2f} ---")
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
