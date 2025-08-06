import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

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

# --- 2. DkNN FEATURE EXTRACTION AND CLASSIFIER ---

class DkNNClassifier:
    def __init__(self, model, reference_loader, device, k=10, num_classes=10):
        self.model = model
        self.reference_loader = reference_loader
        self.device = device
        self.k = k
        self.num_classes = num_classes
        self.reference_features = self._extract_features()
        self.reference_labels = self._extract_labels()
        self.knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        self._build_knn_index()

    def _extract_features(self):
        """Extracts features from the penultimate layer of the ResNet model."""
        features_list = []
        with torch.no_grad():
            for inputs, _ in tqdm(self.reference_loader, desc="Extracting reference features"):
                inputs = inputs.to(self.device)
                features = self.model.forward_features(inputs).cpu().numpy()
                features_list.append(features)
        return np.concatenate(features_list)

    def _extract_labels(self):
        """Extracts labels from the reference dataset."""
        labels_list = []
        for _, labels in self.reference_loader:
            labels_list.append(labels.numpy())
        return np.concatenate(labels_list)

    def _build_knn_index(self):
        """Builds the k-NN index using the extracted features."""
        print("Building k-NN index...")
        self.knn.fit(self.reference_features)
        print("k-NN index built successfully.")

    def predict_and_get_uncertainty(self, input_data):
        """
        Makes a DkNN prediction and calculates uncertainty for a given input.
        Uncertainty is defined as the entropy of the neighbors' labels.
        """
        self.model.eval()
        with torch.no_grad():
            features = self.model.forward_features(input_data).cpu().numpy()

        distances, indices = self.knn.kneighbors(features)
        
        uncertainties = []
        predictions = []

        for i in range(features.shape[0]):
            neighbor_labels = self.reference_labels[indices[i]]
            
            # Calculate class probabilities based on neighbor labels
            class_probs = np.bincount(neighbor_labels, minlength=self.num_classes) / self.k
            
            # Prediction is the class with the highest probability
            prediction = np.argmax(class_probs)
            predictions.append(prediction)
            
            # Fixed uncertainty calculation to handle zero probabilities properly
            # Only include non-zero probabilities in entropy calculation
            nonzero_probs = class_probs[class_probs > 0]
            if len(nonzero_probs) > 0:
                entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
            else:
                entropy = 0.0  # If all probabilities are zero (shouldn't happen), set entropy to 0
            uncertainties.append(entropy)
            
        return np.array(predictions), np.array(uncertainties)

# Modify ResNet56 to expose features from an internal layer
def resnet56_forward_features(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.avgpool(out)
    out = torch.flatten(out, 1)
    return out

# Bind the new method to the model class
model.forward_features = resnet56_forward_features.__get__(model, type(model))

# Initialize the DkNN classifier
dknn = DkNNClassifier(model, train_loader, device, k=10)

# --- 3. PLOTTING UNCERTAINTY ---

def plot_dknn_bar_chart(dknn_classifier, val_loader, num_bins=20):
    """
    Plots the number of correct and incorrect predictions vs. DkNN uncertainty.
    """
    all_predictions = []
    all_uncertainties = []
    all_labels = []
    
    total_images_in_dataset = len(val_loader.dataset)
    print(f"Total images in validation dataset: {total_images_in_dataset}")

    print("Evaluating DkNN on the validation set...")
    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(device)
        preds, uncertainties = dknn_classifier.predict_and_get_uncertainty(inputs)
        
        all_predictions.extend(preds)
        all_uncertainties.extend(uncertainties)
        all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    all_labels = np.array(all_labels)

    num_predictions_collected = len(all_predictions)
    print(f"Total predictions collected: {num_predictions_collected}")

    # Diagnostic information
    print(f"Uncertainty statistics:")
    print(f"  Min uncertainty: {np.min(all_uncertainties)}")
    print(f"  Max uncertainty: {np.max(all_uncertainties)}")
    print(f"  Mean uncertainty: {np.mean(all_uncertainties)}")
    print(f"  Std uncertainty: {np.std(all_uncertainties)}")

    # Check for NaN and Inf values in uncertainties
    nan_count = np.sum(np.isnan(all_uncertainties))
    inf_count = np.sum(np.isinf(all_uncertainties))
    zero_count = np.sum(all_uncertainties == 0)
    negative_count = np.sum(all_uncertainties < 0)

    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    print(f"  Zero values: {zero_count}")
    print(f"  Negative values: {negative_count}")
    
    if nan_count > 0 or inf_count > 0:
        print(f"\nWarning: Found {nan_count} NaN values and {inf_count} Inf values in uncertainties.")
        print("These predictions will be excluded from the histogram.")
        
        # Show some examples of problematic cases
        problematic_indices = np.where(np.logical_or(np.isnan(all_uncertainties), np.isinf(all_uncertainties)))[0][:5]
        print(f"First few problematic indices: {problematic_indices}")
        for idx in problematic_indices:
            print(f"  Index {idx}: uncertainty = {all_uncertainties[idx]}, prediction = {all_predictions[idx]}, label = {all_labels[idx]}")
    else:
        print("No NaN or Inf values found in uncertainties.")
    
    # Filter out invalid uncertainty values before plotting
    valid_mask = np.logical_and(~np.isnan(all_uncertainties), ~np.isinf(all_uncertainties))
    valid_uncertainties = all_uncertainties[valid_mask]
    valid_predictions = all_predictions[valid_mask]
    valid_labels = all_labels[valid_mask]
    valid_corrects = (valid_predictions == valid_labels)
    
    print(f"Valid predictions after filtering: {len(valid_predictions)}")
    
    # Calculate threshold statistics using valid data
    corrects = (all_predictions == all_labels)
    threshold = 1.33  # or any threshold you want

    above_threshold = all_uncertainties > threshold
    correct_above_threshold = np.sum(corrects[above_threshold])
    incorrect_above_threshold = np.sum(~corrects[above_threshold])

    print(f"\nAt uncertainty > {threshold}:")
    print(f"  Number of correct predictions with high uncertainty: {correct_above_threshold}")
    print(f"  Number of incorrect predictions with high uncertainty: {incorrect_above_threshold}")

    # Define bins for the histogram
    max_entropy = np.log2(dknn_classifier.k) # Max entropy is log2(k) for DkNN
    bins = np.linspace(0, max_entropy, num_bins + 1)
    
    correct_counts = np.zeros(num_bins)
    incorrect_counts = np.zeros(num_bins)

    for i in range(num_bins):
        in_bin = (valid_uncertainties >= bins[i]) & (valid_uncertainties < bins[i+1])
        correct_counts[i] = np.sum(valid_corrects[in_bin])
        incorrect_counts[i] = np.sum(~valid_corrects[in_bin])
    
    total_correct = np.sum(correct_counts)
    total_incorrect = np.sum(incorrect_counts)
    total_predictions = total_correct + total_incorrect

    print(f"\nTotal predictions included in bar chart: {total_predictions}")

    # Plot the bar chart
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.35
    
    plt.figure(figsize=(15, 8))
    plt.bar(bin_centers - width/2, correct_counts, width=width, color='green', label='Correct Predictions')
    plt.bar(bin_centers + width/2, incorrect_counts, width=width, color='red', label='Incorrect Predictions')
    
    plt.xlabel('DkNN Uncertainty (Entropy of Neighbors)', fontsize=14)
    plt.ylabel('Number of Predictions', fontsize=14)
    plt.title('Correct vs. Incorrect Predictions by DkNN Uncertainty', fontsize=16)
    plt.xticks(bin_centers, [f'{b:.2f}' for b in bins[:-1]], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('dknn_uncertainty_bar_chart.png')
    print("DkNN uncertainty bar chart saved as 'dknn_uncertainty_bar_chart.png'")

# Call the function to generate the plots
plot_dknn_bar_chart(dknn, val_loader)
