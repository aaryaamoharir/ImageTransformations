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
        
        loss = entropy_loss(outputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            patched_output = model(patched_image)
            probs = F.softmax(patched_output, dim=1)
            max_prob = torch.max(probs)
            if max_prob.item() > 0.95:
                break
    
    return patch.detach()

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
    PATCH_SIZE = 8
    
    total_images_processed = 0
    total_mispredicted_images = 0
    total_mispredicted_corrected = 0
    total_mispredicted_still_incorrect = 0

    print("--- Starting per-image analysis and patching on mispredicted inputs ---")
    model.eval()
    
    for image, label in tqdm(testloader, total=len(testloader), desc="Processing All Images"):
        image, label = image.to(device), label.to(device)
        
        total_images_processed += 1
        
        # Evaluate initial correctness
        is_correct_initial = evaluate_single_image(model, image, label)
        
        # Check if the image is mispredicted
        if not is_correct_initial:
            total_mispredicted_images += 1
            
            # Generate and apply a new patch for this mispredicted image
            restorative_patch = get_restorative_patch(model, image, patch_size=PATCH_SIZE, num_iterations=30)
            
            # Evaluate after patching
            is_correct_after_patch = evaluate_single_image(model, image, label, patch=restorative_patch, patch_size=PATCH_SIZE)
            
            if is_correct_after_patch:
                total_mispredicted_corrected += 1
            else:
                total_mispredicted_still_incorrect += 1

    print("\n--- Analysis for Mispredicted Images ---")
    print(f"Total images processed: {total_images_processed}")
    print(f"Total mispredicted images identified: {total_mispredicted_images}")
    
    if total_mispredicted_images > 0:
        accuracy_before = total_mispredicted_corrected / total_mispredicted_images
        print(f"\nAccuracy of mispredicted images (after patching): {accuracy_before:.4f}")
    else:
        print("\nNo mispredicted images found in the test set.")
    
    print("\n--- Correct/Incorrect Counts for the Mispredicted Subset ---")
    print(f"Mispredicted before patching: {total_mispredicted_images}")
    print(f"Corrected after patching: {total_mispredicted_corrected}")
    print(f"Still mispredicted after patching: {total_mispredicted_still_incorrect}")
