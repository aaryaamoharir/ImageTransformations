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

# --- 2. FEATURE DISTANCE UNCERTAINTY CLASS ---

class FeatureDistanceUncertainty:
    def __init__(self, model, reference_loader, device):
        self.model = model
        self.reference_loader = reference_loader
        self.device = device
        self.reference_features, self.reference_labels = self._extract_reference_data()
        self.knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        self._build_knn_index()

    def _extract_reference_data(self):
        """Extracts features and labels from the reference dataset."""
        features_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.reference_loader, desc="Extracting reference features"):
                inputs = inputs.to(self.device)
                features = self.model.forward_features(inputs).cpu().numpy()
                features_list.append(features)
                labels_list.append(labels.numpy())
        return np.concatenate(features_list), np.concatenate(labels_list)

    def _build_knn_index(self):
        """Builds the k-NN index using the extracted features."""
        print("Building nearest neighbor index...")
        self.knn.fit(self.reference_features)
        print("Index built successfully.")

    def get_uncertainty_and_prediction(self, input_data):
        """
        Calculates uncertainty (distance to nearest neighbor) and prediction.
        """
        self.model.eval()
        with torch.no_grad():
            features = self.model.forward_features(input_data).cpu().numpy()

        distances, indices = self.knn.kneighbors(features)
        
        uncertainties = distances.flatten()
        
        # Get the label of the nearest neighbor for prediction
        predictions = self.reference_labels[indices.flatten()]
        
        return predictions, uncertainties

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

# Initialize the Feature Distance Uncertainty class
distance_uncertainty_model = FeatureDistanceUncertainty(model, train_loader, device)

# --- 3. PLOTTING UNCERTAINTY ---

def plot_distance_bar_chart(uncertainty_model, val_loader, num_bins=20):
    """
    Plots the number of correct and incorrect predictions vs. feature distance.
    """
    all_predictions = []
    all_uncertainties = []
    all_labels = []

    print("Evaluating model on the validation set...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            preds, uncertainties = uncertainty_model.get_uncertainty_and_prediction(inputs)
            
            all_predictions.extend(preds)
            all_uncertainties.extend(uncertainties)
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    all_labels = np.array(all_labels)
    
    corrects = (all_predictions == all_labels)
    threshold = 1.90  # or any threshold you want

    above_threshold = all_uncertainties > threshold
    correct_above_threshold = np.sum(corrects[above_threshold])
    incorrect_above_threshold = np.sum(~corrects[above_threshold])

    print(f"\nAt uncertainty > {threshold}:")
    print(f"  Number of correct predictions with high uncertainty: {correct_above_threshold}")
    print(f"  Number of incorrect predictions with high uncertainty: {incorrect_above_threshold}")
 
    # Define bins for the histogram
    max_distance = np.max(all_uncertainties)
    bins = np.linspace(0, max_distance, num_bins + 1)
    
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
    
    plt.xlabel('Feature Distance to Nearest Neighbor', fontsize=14)
    plt.ylabel('Number of Predictions', fontsize=14)
    plt.title('Correct vs. Incorrect Predictions by Feature Distance', fontsize=16)
    plt.xticks(bin_centers, [f'{b:.2f}' for b in bins[:-1]], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('feature_distance_bar_chart.png')
    print("Feature distance bar chart saved as 'feature_distance_bar_chart.png'")

# Call the function to generate the plots
plot_distance_bar_chart(distance_uncertainty_model, val_loader)
