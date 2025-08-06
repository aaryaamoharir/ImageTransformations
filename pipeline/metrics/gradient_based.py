import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. SETUP AND DATA LOADING ---
print("Setting up data and model...")

# Define transformations for the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load the CIFAR-10 validation set
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the pre-trained CIFAR-10 ResNet-56 model
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
model.eval()

# --- 2. GRADIENT-BASED UNCERTAINTY CALCULATION ---

def calculate_gradient_uncertainty(model, input_data, epsilon=0.01):
    """
    Calculates gradient-based uncertainty for an input batch.
    Uncertainty is defined as the L2 norm of the gradient of the logits with 
    respect to the input image, using a small perturbation.
    """
    model.eval()
    input_data.requires_grad = True

    # Get the model's logits for the input
    original_logits = model(input_data)
    
    # Define a simple "consistency loss" as the L2 distance between
    # the original logits and the logits of a slightly perturbed input.
    # Note: A real-world method would use a more sophisticated loss.
    
    # Create a small perturbation
    perturbation = torch.randn_like(input_data) * epsilon
    
    # Get the logits for the perturbed input
    perturbed_logits = model(input_data + perturbation)

    # Compute a simple consistency loss
    consistency_loss = nn.MSELoss()(original_logits, perturbed_logits)
    
    # Backpropagate to get the gradient with respect to the input
    model.zero_grad()
    consistency_loss.backward()
    
    # The uncertainty is the L2 norm of the gradient
    gradient_norm = torch.norm(input_data.grad.view(input_data.size(0), -1), dim=1)
    
    # Get the predictions from the original logits
    probabilities = torch.softmax(original_logits, dim=1)
    _, predictions = torch.max(probabilities, dim=1)
    
    return predictions, gradient_norm

# --- 3. PLOTTING UNCERTAINTY ---

def plot_gradient_bar_chart(model, val_loader, device, num_bins=20):
    """
    Plots the number of correct and incorrect predictions vs. gradient uncertainty.
    """
    all_predictions = []
    all_uncertainties = []
    all_labels = []
    
    print("Evaluating model with gradient-based uncertainty...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # The `calculate_gradient_uncertainty` function requires a grad-enabled tensor,
            # so we run it outside the `torch.no_grad()` block to allow backprop.
            with torch.enable_grad():
                preds, uncertainties = calculate_gradient_uncertainty(model, inputs)
            
            all_predictions.extend(preds.cpu().numpy())
            all_uncertainties.extend(uncertainties.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    all_labels = np.array(all_labels)
    
    corrects = (all_predictions == all_labels)
    under_correct = 0
    under_incorrect = 0
    threshhold = 1.0
    for i in range(len(all_predictions)):
        if (all_uncertainties[i] > 0.03):
            if( all_predictions[i] == all_labels[i]):
                under_correct = under_correct + 1 
            else:
                under_incorrect = under_incorrect + 1 
    print("Number of correct predictions with high uncertainty:", under_correct)
    print("Number of incorrect predictions with high uncertainty:", under_incorrect)
  
    # Define bins for the histogram
    max_uncertainty = np.max(all_uncertainties)
    bins = np.linspace(0, max_uncertainty, num_bins + 1)
    
    correct_counts = np.zeros(num_bins)
    incorrect_counts = np.zeros(num_bins)

    for i in range(num_bins):
        in_bin = (all_uncertainties >= bins[i]) & (all_uncertainties < bins[i+1])
        correct_counts[i] = np.sum(corrects[in_bin])
        incorrect_counts[i] = np.sum(~corrects[in_bin])
    
    # Plot the bar chart
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.35
    
    plt.figure(figsize=(15, 8))
    plt.bar(bin_centers - width/2, correct_counts, width=width, color='green', label='Correct Predictions')
    plt.bar(bin_centers + width/2, incorrect_counts, width=width, color='red', label='Incorrect Predictions')
    
    plt.xlabel('Gradient-Based Uncertainty (L2 Norm of Gradient)', fontsize=14)
    plt.ylabel('Number of Predictions', fontsize=14)
    plt.title('Correct vs. Incorrect Predictions by Gradient Uncertainty', fontsize=16)
    plt.xticks(bin_centers, [f'{b:.2f}' for b in bins[:-1]], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('gradient_uncertainty_bar_chart.png')
    print("Gradient-based uncertainty bar chart saved as 'gradient_uncertainty_bar_chart.png'")

# Call the function to generate the plots
plot_gradient_bar_chart(model, val_loader, device)
