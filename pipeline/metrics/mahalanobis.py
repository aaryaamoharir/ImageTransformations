import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score # Added roc_auc_score

# --- 1. SETUP AND DATA LOADING ---
print("Setting up data and model for CIFAR-100 with Mahalanobis distance...")

transform = transforms.Compose([
    transforms.ToTensor(),
    # Use CIFAR-10 specific normalization values
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)

val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
model.eval()

# --- 2. FEATURE EXTRACTION AND UNCERTAINTY CALCULATION ---
def resnet56_forward_features(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.avgpool(out)
    out = torch.flatten(out, 1)
    return out

model.forward_features = resnet56_forward_features.__get__(model, type(model))

def extract_features_and_labels(model, dataloader, device):
    features_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting reference features"):
            inputs = inputs.to(device)
            features = model.forward_features(inputs).cpu().numpy()
            features_list.append(features)
            labels_list.append(labels.numpy())
    return np.concatenate(features_list), np.concatenate(labels_list)

def get_mahalanobis_uncertainty(model, val_loader, device, all_train_features):
    train_features_mean = np.mean(all_train_features, axis=0)
    train_features_cov = np.cov(all_train_features.T)
    precision_matrix_reg = np.linalg.pinv(train_features_cov + np.eye(train_features_cov.shape[0]) * 1e-4)
    
    uncertainty_scores = []
    corrects = []
    
    print("\nCalculating Mahalanobis distances on validation set...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            test_features = model.forward_features(images).cpu().numpy()
            
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            is_correct = (preds == labels.cpu().numpy())
            
            for f in test_features:
                f_centered = f - train_features_mean
                uncertainty = f_centered.T @ precision_matrix_reg @ f_centered
                uncertainty_scores.append(uncertainty)
            
            corrects.extend(is_correct)

    return np.array(uncertainty_scores), np.array(corrects)

all_train_features, _ = extract_features_and_labels(model, train_loader, device)
uncertainties, corrects = get_mahalanobis_uncertainty(model, val_loader, device, all_train_features)

# --- NEW: AUC and Thresholding Logic ---
misclassification_labels = (~corrects).astype(int)
auc_score = roc_auc_score(misclassification_labels, uncertainties)
print(f"\nMahalanobis Distance Uncertainty AUC Score: {auc_score:.4f}")

threshold = 250 # User-specified threshold
correct_above_threshold = np.sum(corrects[uncertainties > threshold])
incorrect_above_threshold = np.sum(~corrects[uncertainties > threshold])
correct_below_threshold = np.sum(corrects[uncertainties <= threshold])
incorrect_below_threshold = np.sum(~corrects[uncertainties <= threshold])

print(f"\n--- Mahalanobis Uncertainty Results (Threshold = {threshold:.2f}) ---")
print(f"Correct predictions with uncertainty > {threshold:.2f}: {correct_above_threshold}")
print(f"Incorrect predictions with uncertainty > {threshold:.2f}: {incorrect_above_threshold}")
print(f"Correct predictions with uncertainty <= {threshold:.2f}: {correct_below_threshold}")
print(f"Incorrect predictions with uncertainty <= {threshold:.2f}: {incorrect_below_threshold}")
print(f"Total predictions: {len(uncertainties)}")

# --- 3. GRAPHING LOGIC ---
def plot_uncertainty_bar_chart(uncertainties, corrects, num_bins=20):
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
    
    plt.xlabel('Mahalanobis Distance (Uncertainty)', fontsize=14)
    plt.ylabel('Number of Predictions', fontsize=14)
    plt.title('Correct vs. Incorrect Predictions by Mahalanobis Distance', fontsize=16)
    plt.xticks(bin_centers, [f'{b:.2f}' for b in bins[:-1]], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('mahalanobis_uncertainty_bar_chart.png')
    print("Bar chart saved as 'mahalanobis_uncertainty_bar_chart.png'")

plot_uncertainty_bar_chart(uncertainties, corrects)
