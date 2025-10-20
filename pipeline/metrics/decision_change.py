import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score # New import

# --- Common Setup ---
print("Setting up data and model for CIFAR-10 with Decision Change Uncertainty...")

transform = transforms.Compose([
    transforms.ToTensor(),
    # Use CIFAR-10 specific normalization values
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
model.eval()

# --- Decision Change Uncertainty Calculation ---
def get_decision_change_uncertainty(model, dataloader, device, epsilon=0.01):
    all_uncertainties = []
    all_predictions = []
    all_labels = []

    print(f"\nCalculating Decision Change uncertainties with epsilon={epsilon}...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Original prediction
            original_logits = model(inputs)
            original_probs = torch.softmax(original_logits, dim=1)
            _, original_preds = torch.max(original_probs, dim=1)

            # Perturb input
            perturbed_inputs = inputs + epsilon * torch.randn_like(inputs)
            perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1) # Keep pixel values valid

            # Prediction on perturbed input
            perturbed_logits = model(perturbed_inputs)
            perturbed_probs = torch.softmax(perturbed_logits, dim=1)
            
            # Uncertainty is 1 - MSP of the perturbed input
            max_perturbed_probs, _ = torch.max(perturbed_probs, dim=1)
            uncertainty = 1 - max_perturbed_probs
            
            all_uncertainties.extend(uncertainty.cpu().numpy())
            all_predictions.extend(original_preds.cpu().numpy()) # Use original prediction for correctness
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_predictions), np.array(all_uncertainties), np.array(all_labels)

# --- New function to calculate and print AUC-ROC ---
def calculate_and_print_auc(uncertainties, corrects):
    # The 'y_true' labels for AUC are whether the prediction was correct (0) or incorrect (1)
    # The 'y_score' is our uncertainty measure.
    # To get a high AUC for a good uncertainty metric, we need higher scores for incorrect predictions.
    # Our `uncertainties` are already in this format (high uncertainty for incorrect predictions).
    
    # We will use '1 - corrects' to represent incorrect predictions (1 for incorrect, 0 for correct)
    y_true = 1 - corrects.astype(int)
    y_score = uncertainties
    
    try:
        auc_score = roc_auc_score(y_true, y_score)
        print(f"Area Under the ROC Curve (AUC-ROC): {auc_score:.4f}")
    except ValueError:
        print("AUC-ROC could not be calculated. This may happen if all predictions are correct or incorrect.")

# --- Plotting and Thresholding Logic (updated) ---
def plot_and_threshold_uncertainty(predictions, uncertainties, labels, method_name, filename_prefix, num_bins=20, threshold=0.5):
    corrects = (predictions == labels)
    
    # Calculate and print AUC-ROC
    calculate_and_print_auc(uncertainties, corrects)
    
    # Thresholding
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

    # Plotting
    max_uncertainty = np.max(uncertainties)
    bins = np.linspace(0, max_uncertainty, num_bins + 1)
    
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

# --- Run Decision Change Uncertainty ---
predictions, uncertainties, labels = get_decision_change_uncertainty(model, val_loader, device, epsilon=0.01)
plot_and_threshold_uncertainty(predictions, uncertainties, labels, 
                               "Decision Change (1-MSP on Perturbed Input)", "decision_change", 
                               threshold=0.5) # Example threshold
