import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import math

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load and download CIFAR-10 data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class SGLD(optim.Optimizer):
    def __init__(self, params, lr=1e-2, weight_decay=0, temperature=1.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay, temperature=temperature)
        super(SGLD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # SGLD update: subtract gradient and add Gaussian noise
                # The noise variance is scaled by 2 * learning_rate * temperature
                noise = torch.randn_like(p.data) * math.sqrt(2 * group['lr'] * group['temperature'])
                p.data.add_(grad, alpha=-group['lr'])
                p.data.add_(noise)

        return loss

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = SGLD(model.parameters(), lr=1e-4, weight_decay=1e-5, temperature=1.0) # SGLD optimizer

num_epochs = 20 # Increase epochs for better sampling
sample_frequency = 200 # Collect a sample every N batches
collected_weights = []

print("\nStarting SGMCMC Training...")
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Collect weight samples
        if (i + 1) % sample_frequency == 0:
            collected_weights.append(model.state_dict())

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')

print(f"Finished SGMCMC Training. Collected {len(collected_weights)} weight samples.")

def shannon_entropy(probs):
    """Calculates Shannon entropy for a probability distribution."""
    # Clip probabilities to avoid log(0) errors
    probs = np.clip(probs, a_min=1e-12, a_max=1.0)
    return -np.sum(probs * np.log2(probs), axis=1)

all_uncertainties = []
all_correct = []

print("\nQuantifying uncertainty on the test set...")
# Iterate through the test set
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        # For each image in the batch, get predictions from all collected weight samples
        for img_idx in range(images.size(0)):
            img = images[img_idx].unsqueeze(0) # Add batch dimension
            true_label = labels[img_idx].item()

            individual_predictions = []
            for state_dict in collected_weights:
                model.load_state_dict(state_dict) # Load sampled weights
                model.eval() # Set to eval mode for consistent forward pass
                output = model(img)
                individual_predictions.append(torch.softmax(output, dim=1).cpu().numpy().flatten())

            individual_predictions = np.array(individual_predictions)

            # Calculate mean predictive probability distribution
            mean_predictive_probs = np.mean(individual_predictions, axis=0)

            # Calculate Shannon entropy as uncertainty
            uncertainty_value = shannon_entropy(mean_predictive_probs.reshape(1, -1))[0]

            predicted_class = np.argmax(mean_predictive_probs)
            is_correct = (predicted_class == true_label)

            all_uncertainties.append(uncertainty_value)
            all_correct.append(is_correct)

all_uncertainties = np.array(all_uncertainties)
all_correct = np.array(all_correct)

# Create bins for the uncertainty values
num_bins = 20
# Use min/max uncertainty to define bins
uncertainty_bins = np.linspace(all_uncertainties.min(), all_uncertainties.max(), num_bins + 1)
bin_indices = np.digitize(all_uncertainties, uncertainty_bins)

correct_counts = np.zeros(num_bins)
incorrect_counts = np.zeros(num_bins)

# Populate the bins with correct and incorrect counts
for i in range(1, num_bins + 1):
    bin_mask = (bin_indices == i)
    correct_counts[i-1] = np.sum(all_correct[bin_mask])
    incorrect_counts[i-1] = np.sum(~all_correct[bin_mask])

# Plot the bar graph
fig, ax = plt.subplots(figsize=(12, 6))
bin_centers = (uncertainty_bins[:-1] + uncertainty_bins[1:]) / 2

width = (uncertainty_bins[1] - uncertainty_bins[0]) * 0.4 # Adjust width for separate bars

ax.bar(bin_centers - width/2, correct_counts, width, label='Correct Images', color='skyblue')
ax.bar(bin_centers + width/2, incorrect_counts, width, label='Incorrect Images', color='salmon')

ax.set_xlabel('Uncertainty (Shannon Entropy)', fontsize=12)
ax.set_ylabel('Number of Images', fontsize=12)
ax.set_title('Correct vs. Incorrect Predictions per Uncertainty Bin (SGMCMC)', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig("sgsc")
from sklearn.metrics import roc_auc_score

# Prepare binary labels: incorrect=1, correct=0
labels = (~all_correct).astype(int)

# Compute AUC
auc_score = roc_auc_score(labels, all_uncertainties)

print(f"AUC score for Shannon entropy uncertainty detecting errors: {auc_score:.4f}")

# === Threshold logic to print counts above a specified uncertainty threshold ===

threshold = 0.3  # Set your uncertainty threshold here (example: 3.0)

above_threshold_mask = all_uncertainties > threshold
num_above = np.sum(above_threshold_mask)
correct_above = np.sum(all_correct[above_threshold_mask])
incorrect_above = np.sum(~all_correct[above_threshold_mask])

print(f"\nFor uncertainty threshold > {threshold}:")
print(f"  Number of images above threshold: {num_above}")
print(f"  Number of correct predictions above threshold: {correct_above}")
print(f"  Number of incorrect predictions above threshold: {incorrect_above}")

threshold = 0.5  # Set your uncertainty threshold here (example: 3.0)

above_threshold_mask = all_uncertainties > threshold
num_above = np.sum(above_threshold_mask)
correct_above = np.sum(all_correct[above_threshold_mask])
incorrect_above = np.sum(~all_correct[above_threshold_mask])

print(f"\nFor uncertainty threshold > {threshold}:")
print(f"  Number of images above threshold: {num_above}")
print(f"  Number of correct predictions above threshold: {correct_above}")
print(f"  Number of incorrect predictions above threshold: {incorrect_above}")

threshold = 0.2  # Set your uncertainty threshold here (example: 3.0)

above_threshold_mask = all_uncertainties > threshold
num_above = np.sum(above_threshold_mask)
correct_above = np.sum(all_correct[above_threshold_mask])
incorrect_above = np.sum(~all_correct[above_threshold_mask])

print(f"\nFor uncertainty threshold > {threshold}:")
print(f"  Number of images above threshold: {num_above}")
print(f"  Number of correct predictions above threshold: {correct_above}")
print(f"  Number of incorrect predictions above threshold: {incorrect_above}")
(.venv) diversity_project@csr-88307:~/aaryaa/attacks/pipeline/metrics$ 
