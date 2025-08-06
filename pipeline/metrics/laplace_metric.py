import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from laplace import Laplace

# --- 1. SETUP AND DATA LOADING ---
print("Setting up data and model...")

# Define transformations for the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load the CIFAR-10 training set as the reference dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)

# Load the CIFAR-10 validation set for evaluation
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the pre-trained CIFAR-10 ResNet-56 model
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
model.eval()

# --- 2. LAPLACE APPROXIMATION MODEL SETUP ---

# We use the pre-trained model as the core of our Laplace model.
la_model = Laplace(model, 'classification', subset_of_weights='last_layer', hessian_structure='diag')

# Fit the Laplace model on a subset of the data. 
# This calibrates the model without re-training its weights.
print("Fitting Laplace Approximation model...")
la_model.fit(train_loader)
print("Laplace model fitted successfully.")

# --- 3. PREDICTION AND UNCERTAINTY CALCULATION ---

def plot_laplace_uncertainty_bar_chart(la_model, val_loader, device, num_bins=20):
    """
    Calculates Laplace predictive uncertainty and plots the number of correct and
    incorrect predictions vs. uncertainty.
    """
    all_predictions = []
    all_uncertainties = []
    all_labels = []

    print("Evaluating Laplace model on the validation set...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Predict with the Laplace model
            logits = la_model(inputs)
            
            # Use predictive variance as the uncertainty measure
            # A higher variance indicates higher uncertainty.
            predictive_variance = la_model.predictive_variance(inputs)
            
            # Predictions are still based on the logits
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_predictions.extend(preds)
            all_uncertainties.extend(predictive_variance.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    all_labels = np.array(all_labels)

    corrects = (all_predictions == all_labels)
    
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
    
    plt.xlabel('Laplace Uncertainty (Predictive Variance)', fontsize=14)
    plt.ylabel('Number of Predictions', fontsize=14)
    plt.title('Correct vs. Incorrect Predictions by Laplace Uncertainty', fontsize=16)
    plt.xticks(bin_centers, [f'{b:.2f}' for b in bins[:-1]], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('laplace_uncertainty_bar_chart.png')
    print("Laplace uncertainty bar chart saved as 'laplace_uncertainty_bar_chart.png'")

# Call the function to generate the plots
plot_laplace_uncertainty_bar_chart(la_model, val_loader, device)
