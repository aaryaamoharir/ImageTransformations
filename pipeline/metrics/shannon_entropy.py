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

# --- 2. PREDICTIVE ENTROPY CALCULATION AND PLOTTING ---

def plot_entropy_bar_chart(model, val_loader, device, num_bins=20):
    """
    Calculates predictive entropy and plots the number of correct and incorrect 
    predictions across different entropy levels.
    """
    model.eval()
    entropies = []
    corrects = []

    print("Calculating predictive entropies and predictions...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get logits and convert to probabilities
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            
            # Calculate predictive entropy for each prediction
            # Using log2 for standard entropy measure
            entropy_values = -torch.sum(probabilities * torch.log2(probabilities + 1e-10), dim=1)
            
            # Get predictions for correctness check
            _, preds = torch.max(probabilities, dim=1)
            is_correct = (preds == labels).cpu().numpy()
            
            entropies.extend(entropy_values.cpu().numpy())
            corrects.extend(is_correct)

    entropies = np.array(entropies)
    corrects = np.array(corrects)
    threshold = 1.66  # or any threshold you want

    below_threshold = entropies > threshold
    correct_under_threshold = np.sum(corrects[below_threshold])
    incorrect_under_threshold = np.sum(~corrects[below_threshold])

    print(f"\nAt uncertainty > {threshold}:")
    print(f"  Number of correct predictions with high uncertainty: {correct_under_threshold}")
    print(f"  Number of incorrect predictions with high uncertainty: {incorrect_under_threshold}")


    # Define bins for the histogram. Max entropy for 10 classes is log2(10) ≈ 3.32.
    max_entropy = np.log2(10)
    bins = np.linspace(0, max_entropy, num_bins + 1)
    
    correct_counts = np.zeros(num_bins)
    incorrect_counts = np.zeros(num_bins)

    for i in range(num_bins):
        bin_start = bins[i]
        bin_end = bins[i+1]
        
        in_bin = (entropies >= bin_start) & (entropies < bin_end)
        
        correct_counts[i] = np.sum(corrects[in_bin])
        incorrect_counts[i] = np.sum(~corrects[in_bin])
    
    # Plot the bar chart with two separate bars per bin
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.35  # Adjust width for two bars
    
    plt.figure(figsize=(15, 8))
    plt.bar(bin_centers - width/2, correct_counts, width=width, color='green', label='Correct Predictions')
    plt.bar(bin_centers + width/2, incorrect_counts, width=width, color='red', label='Incorrect Predictions')
    
    plt.xlabel('Predictive Entropy (bits)', fontsize=14)
    plt.ylabel('Number of Predictions', fontsize=14)
    plt.title('Correct vs. Incorrect Predictions by Predictive Entropy', fontsize=16)
    plt.xticks(bin_centers, [f'{b:.2f}' for b in bins[:-1]], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('predictive_entropy_bar_chart.png')
    print("Predictive entropy bar chart saved as 'predictive_entropy_bar_chart.png'")

# Call the function to generate the plot
plot_entropy_bar_chart(model, val_loader, device)
