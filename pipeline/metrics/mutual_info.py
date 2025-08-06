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
# Note: This model doesn't have dropout by default, so we'll need to modify it.
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)

# --- 2. MONTE CARLO DROPOUT MODIFICATION ---

# Add dropout layers to the model for uncertainty estimation
# This requires iterating through the model's layers and adding dropout
for name, module in model.named_modules():
    if isinstance(module, (nn.ReLU, nn.GELU)):
        # Insert a dropout layer after activation functions
        # This is a common practice for Monte Carlo dropout
        new_dropout = nn.Dropout(p=0.5)
        setattr(model, name.split('.')[-1], nn.Sequential(module, new_dropout))
print("Dropout layers added to the model for MC Dropout.")
model.eval()

# --- 3. MUTUAL INFORMATION ENTROPY CALCULATION AND PLOTTING ---

def plot_mutual_information_bar_chart(model, val_loader, device, num_bins=20, T=50):
    """
    Calculates Mutual Information (MI) uncertainty using Monte Carlo Dropout
    and plots the distribution of correct vs. incorrect predictions.
    
    Args:
        model: The PyTorch model with dropout layers.
        val_loader: The validation data loader.
        device: The device to run the model on.
        num_bins: The number of bins for the plot.
        T: The number of Monte Carlo samples.
    """
    model.train() # Enable dropout layers during inference
    
    mutual_informations = []
    corrects = []
    
    print(f"Calculating Mutual Information with T={T} Monte Carlo samples...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Run T forward passes with dropout enabled
            probabilities = []
            for _ in range(T):
                logits = model(inputs)
                probabilities.append(torch.softmax(logits, dim=1))
            
            # Stack and average the probabilities across samples
            probabilities = torch.stack(probabilities, dim=0)
            average_probabilities = probabilities.mean(dim=0)
            
            # --- Calculate Mutual Information ---
            
            # 1. Predictive Entropy: Entropy of the average probability distribution
            predictive_entropy = -torch.sum(average_probabilities * torch.log2(average_probabilities + 1e-10), dim=1)
            
            # 2. Expected Entropy: Average of the entropies of individual samples
            individual_entropies = -torch.sum(probabilities * torch.log2(probabilities + 1e-10), dim=2)
            expected_entropy = individual_entropies.mean(dim=0)
            
            # 3. Mutual Information: Difference between the two
            mutual_information_values = predictive_entropy - expected_entropy
            
            # Get predictions for correctness check (from the average probabilities)
            _, preds = torch.max(average_probabilities, dim=1)
            is_correct = (preds == labels).cpu().numpy()
            
            mutual_informations.extend(mutual_information_values.cpu().numpy())
            corrects.extend(is_correct)

    mutual_informations = np.array(mutual_informations)
    corrects = np.array(corrects)

    threshold = 1.82  # or any threshold you want

    above_threshold = mutual_informations > threshold

    correct_above_threshold = np.sum(corrects[above_threshold])
    incorrect_above_threshold = np.sum(~corrects[above_threshold])

    print(f"\nAt uncertainty > {threshold}:")
    print(f"  Number of correct predictions with high uncertainty: {correct_above_threshold}")
    print(f"  Number of incorrect predictions with high uncertainty: {incorrect_above_threshold}")


    # Define bins for the histogram
    # MI values are typically small, so we use a range from 0 to 1
    bins = np.linspace(0, np.max(mutual_informations), num_bins + 1)
    
    correct_counts = np.zeros(num_bins)
    incorrect_counts = np.zeros(num_bins)

    for i in range(num_bins):
        bin_start = bins[i]
        bin_end = bins[i+1]
        
        in_bin = (mutual_informations >= bin_start) & (mutual_informations < bin_end)
        
        correct_counts[i] = np.sum(corrects[in_bin])
        incorrect_counts[i] = np.sum(~corrects[in_bin])
    
    # Plot the bar chart
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.35
    
    plt.figure(figsize=(15, 8))
    plt.bar(bin_centers - width/2, correct_counts, width=width, color='green', label='Correct Predictions')
    plt.bar(bin_centers + width/2, incorrect_counts, width=width, color='red', label='Incorrect Predictions')
    
    plt.xlabel('Mutual Information (bits)', fontsize=14)
    plt.ylabel('Number of Predictions', fontsize=14)
    plt.title('Correct vs. Incorrect Predictions by Mutual Information (MC Dropout)', fontsize=16)
    plt.xticks(bin_centers, [f'{b:.2f}' for b in bins[:-1]], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('mutual_information_bar_chart.png')
    print("Mutual information bar chart saved as 'mutual_information_bar_chart.png'")

# Call the function to generate the plot
plot_mutual_information_bar_chart(model, val_loader, device)
