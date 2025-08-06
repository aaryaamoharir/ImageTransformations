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

# --- 2. TEMPERATURE SCALING IMPLEMENTATION ---

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        # Initialize temperature with a value greater than 1 to start
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Scales the logits by the temperature parameter.
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, val_loader):
        """
        Calibrates the model by finding the optimal temperature T on the validation set.
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
                logits = self.model(input.to(device))
                logits_list.append(logits)
                labels_list.append(label)
        
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        # Find the optimal temperature using L-BFGS optimizer
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        print(f'Optimal temperature found: {self.temperature.item():.4f}')

# Create a temperature-scaled version of the model and calibrate it
scaled_model = ModelWithTemperature(model).to(device)
print("Calibrating the model to find optimal temperature...")
scaled_model.set_temperature(val_loader)
print("Calibration complete.")

# --- 3. UNCERTAINTY PLOTTING ---

def plot_uncertainty_bar_chart(model, val_loader, device, num_bins=20):
    """
    Calculates and plots the number of correct and incorrect predictions for
    different uncertainty levels.
    """
    model.eval()
    uncertainties = []
    corrects = []
    threshold = 0.55
    under_correct = 0
    under_incorrect = 0 

    print("Calculating uncertainties and predictions...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model.model(inputs)
            calibrated_logits = model.temperature_scale(logits)
            probabilities = torch.softmax(calibrated_logits, dim=1)
            
            # Uncertainty is 1 - max probability
            max_probs, preds = torch.max(probabilities, dim=1)
            uncertainty = 1 - max_probs
            
            # Check for correct predictions
            is_correct = (preds == labels).cpu().numpy()
            
            uncertainty = uncertainty.cpu().numpy()
            #is_correct = is_correct.cpu().numpy()

            for i in range(len(uncertainty)):
                if uncertainty[i] > 0.45:
                    if is_correct[i]:
                        under_correct += 1
                    else:
                        under_incorrect += 1

            uncertainties.extend(uncertainty)
            corrects.extend(is_correct)

    uncertainties = np.array(uncertainties)
    corrects = np.array(corrects)
    print("Number of correct predictions with high uncertainty:", under_correct)
    print("Number of incorrect predictions with high uncertainty:", under_incorrect)

    # Define bins for the histogram
    bins = np.linspace(0, 1, num_bins + 1)
    
    # Calculate counts for correct and incorrect predictions in each bin
    correct_counts = np.zeros(num_bins)
    incorrect_counts = np.zeros(num_bins)

    for i in range(num_bins):
        bin_start = bins[i]
        bin_end = bins[i+1]
        
        # Filter data for the current bin
        in_bin = (uncertainties >= bin_start) & (uncertainties < bin_end)
        
        correct_counts[i] = np.sum(corrects[in_bin])
        incorrect_counts[i] = np.sum(~corrects[in_bin])
    
    # Plot the bar chart with two separate bars per bin
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.35  # Adjust width for two bars
    
    plt.figure(figsize=(15, 8))
    plt.bar(bin_centers - width/2, correct_counts, width=width, color='green', label='Correct Predictions')
    plt.bar(bin_centers + width/2, incorrect_counts, width=width, color='red', label='Incorrect Predictions')
    
    plt.xlabel('Uncertainty (1 - Max Probability)', fontsize=14)
    plt.ylabel('Number of Predictions', fontsize=14)
    plt.title('Correct vs. Incorrect Predictions by Uncertainty Level', fontsize=16)
    plt.xticks(bin_centers, [f'{b:.2f}' for b in bins[:-1]], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('uncertainty_bar_chart.png')
    print("Uncertainty bar chart saved as 'uncertainty_bar_chart.png'")

# Call the function to generate the plot
plot_uncertainty_bar_chart(scaled_model, val_loader, device)
