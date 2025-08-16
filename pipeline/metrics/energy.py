import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Common Setup ---
print("Setting up data and model for CIFAR-100 with Energy Confidence...")
transform = transforms.Compose([
    transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))#transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
model.eval()

# --- Energy Confidence Uncertainty Calculation ---
def get_energy_confidence_uncertainty(model, dataloader, device):
    all_uncertainties = []
    all_predictions = []
    all_labels = []

    print("\nCalculating Energy Confidence uncertainties...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            
            energy_scores = -torch.logsumexp(logits, dim=1)
            
            probabilities = torch.softmax(logits, dim=1)
            _, preds = torch.max(probabilities, dim=1)
            
            all_uncertainties.extend(energy_scores.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_predictions), np.array(all_uncertainties), np.array(all_labels)

# --- Plotting and Thresholding Logic (reused from MSP) ---
def plot_and_threshold_uncertainty(predictions, uncertainties, labels, method_name, filename_prefix, num_bins=20, threshold=0.5):
    corrects = (predictions == labels)
    
    correct_above_threshold = np.sum(corrects[uncertainties > threshold])
    incorrect_above_threshold = np.sum(~corrects[uncertainties > threshold])
    correct_below_threshold = np.sum(corrects[uncertainties <= threshold])
    incorrect_below_threshold = np.sum(~corrects[uncertainties <= threshold])

    print(f"\n--- {method_name} Uncertainty Results (Threshold = {threshold:.2f}) ---")
    print(f"Correct predictions with uncertainty > {threshold:.2f}: {correct_above_threshold}")
    print(f"Incorrect predictions with uncertainty > {threshold:.2f}: {incorrect_above_threshold}")
    print(f"Correct predictions with uncertainty <= {threshold:.2f}: {correct_below_threshold}")
    print(f"Incorrect predictions with uncertainty <= {threshold:.2f}: {incorrect_below_threshold}")
    print(f"Total predictions: {len(predictions)}")

    max_uncertainty = np.max(uncertainties)
    min_uncertainty = np.min(uncertainties)
    bins = np.linspace(min_uncertainty, max_uncertainty, num_bins + 1)
    
    correct_counts = np.zeros(num_bins)
    incorrect_counts = np.zeros(num_bins)

    for i in range(num_bins):
        in_bin = (uncertainties >= bins[i]) & (uncertainties < bins[i+1])
        correct_counts[i] = np.sum(corrects[in_bin])
        incorrect_counts[i] = np.sum(~corrects[in_bin])
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.35
    
    plt.figure(figsize=(15, 8))
    plt.bar(bin_centers - width/2, correct_counts, width=width, color='green', label='Correct Predictions')
    plt.bar(bin_centers + width/2, incorrect_counts, width=width, color='red', label='Incorrect Predictions')
    
    plt.xlabel(f'{method_name} Uncertainty', fontsize=14)
    plt.ylabel('Number of Predictions', fontsize=14)
    plt.title(f'Correct vs. Incorrect Predictions by {method_name} Uncertainty', fontsize=16)
    plt.xticks(bin_centers, [f'{b:.2f}' for b in bins[:-1]], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_bar_chart.png')
    print(f"Bar chart saved as '{filename_prefix}_bar_chart.png'")

# --- Run Energy Confidence Uncertainty ---
predictions, uncertainties, labels = get_energy_confidence_uncertainty(model, val_loader, device)

# --- NEW: Calculate and print AUC score ---
correctness = (predictions == labels)
misclassification_labels = (~correctness).astype(int)

auc_score = roc_auc_score(misclassification_labels, uncertainties)

print(f"\nEnergy Confidence Uncertainty AUC Score: {auc_score:.4f}")

plot_and_threshold_uncertainty(predictions, uncertainties, labels,
                               "Energy Confidence", "energy_confidence",
                               threshold=-5.13)
