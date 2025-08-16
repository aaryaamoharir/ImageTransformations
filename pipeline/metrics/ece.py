import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

# Setup (same as before)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])

val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
model.eval()

# Step 1: Gather confidences, correctness, uncertainties
confidences = []
correctness = []
uncertainties = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

        confidences.extend(max_probs.cpu().numpy())
        correctness.extend((preds == targets).cpu().numpy())
        uncertainties.extend(entropy.cpu().numpy())

confidences = np.array(confidences)
correctness = np.array(correctness, dtype=bool)
uncertainties = np.array(uncertainties)
labels = (~correctness).astype(int)  # incorrect = 1, correct = 0

auc_score = roc_auc_score(labels, uncertainties)
print(f"AUC score (uncertainty detecting errors): {auc_score:.4f}")
# Step 2: Compute ECE
def compute_ece(confidences, correctness, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        if np.any(mask):
            bin_acc = np.mean(correctness[mask])
            bin_conf = np.mean(confidences[mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / len(confidences)
    return ece

ece_val = compute_ece(confidences, correctness, n_bins=10)
print(f"ECE: {ece_val:.4f}")

# Step 3: Bin by uncertainty and count correct/incorrect
n_bins = 20
bins = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

correct_counts = np.zeros(n_bins, dtype=int)
incorrect_counts = np.zeros(n_bins, dtype=int)

for u, corr in zip(uncertainties, correctness):
    idx = np.searchsorted(bins, u, side='right') - 1
    if 0 <= idx < n_bins:
        if corr:
            correct_counts[idx] += 1
        else:
            incorrect_counts[idx] += 1

# Step 4: Plot and save figure
plt.figure(figsize=(10, 6))
width = (bin_centers[1] - bin_centers[0]) * 0.4

plt.bar(bin_centers - width/2, correct_counts, width=width, label='Correct', color='g', alpha=0.7)
plt.bar(bin_centers + width/2, incorrect_counts, width=width, label='Incorrect', color='r', alpha=0.7)

plt.xlabel('Predictive Entropy (Uncertainty)')
plt.ylabel('Number of Samples')
plt.title('Correct vs Incorrect Predictions by Uncertainty Bin')
plt.legend()
plt.tight_layout()
plt.savefig('uncertainty_correct_incorrect_histogram.png')
print("Saved plot to uncertainty_correct_incorrect_histogram.png")
plt.show()

# Step 5: Print counts above a given uncertainty threshold
threshold = 0.80  # set your desired threshold here
above_threshold = uncertainties > threshold

correct_above = np.sum(correctness[above_threshold])
incorrect_above = np.sum(~correctness[above_threshold])

print(f"\nAt uncertainty > {threshold}:")
print(f"  Number of correct predictions with high uncertainty: {correct_above}")
print(f"  Number of incorrect predictions with high uncertainty: {incorrect_above}")

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

# Setup (same as before)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])

val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
model.eval()

# Step 1: Gather confidences, correctness, uncertainties
confidences = []
correctness = []
uncertainties = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

        confidences.extend(max_probs.cpu().numpy())
        correctness.extend((preds == targets).cpu().numpy())
        uncertainties.extend(entropy.cpu().numpy())

confidences = np.array(confidences)
correctness = np.array(correctness, dtype=bool)
uncertainties = np.array(uncertainties)
labels = (~correctness).astype(int)  # incorrect = 1, correct = 0

auc_score = roc_auc_score(labels, uncertainties)
print(f"AUC score (uncertainty detecting errors): {auc_score:.4f}")
# Step 2: Compute ECE
def compute_ece(confidences, correctness, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        if np.any(mask):
            bin_acc = np.mean(correctness[mask])
            bin_conf = np.mean(confidences[mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / len(confidences)
    return ece

ece_val = compute_ece(confidences, correctness, n_bins=10)
print(f"ECE: {ece_val:.4f}")

# Step 3: Bin by uncertainty and count correct/incorrect
n_bins = 20
bins = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

correct_counts = np.zeros(n_bins, dtype=int)
incorrect_counts = np.zeros(n_bins, dtype=int)

for u, corr in zip(uncertainties, correctness):
    idx = np.searchsorted(bins, u, side='right') - 1
    if 0 <= idx < n_bins:
        if corr:
            correct_counts[idx] += 1
        else:
            incorrect_counts[idx] += 1

# Step 4: Plot and save figure
plt.figure(figsize=(10, 6))
width = (bin_centers[1] - bin_centers[0]) * 0.4

plt.bar(bin_centers - width/2, correct_counts, width=width, label='Correct', color='g', alpha=0.7)
plt.bar(bin_centers + width/2, incorrect_counts, width=width, label='Incorrect', color='r', alpha=0.7)

plt.xlabel('Predictive Entropy (Uncertainty)')
plt.ylabel('Number of Samples')
plt.title('Correct vs Incorrect Predictions by Uncertainty Bin')
plt.legend()
plt.tight_layout()
plt.savefig('uncertainty_correct_incorrect_histogram.png')
print("Saved plot to uncertainty_correct_incorrect_histogram.png")
plt.show()

# Step 5: Print counts above a given uncertainty threshold
threshold = 0.80  # set your desired threshold here
above_threshold = uncertainties > threshold

correct_above = np.sum(correctness[above_threshold])
incorrect_above = np.sum(~correctness[above_threshold])

print(f"\nAt uncertainty > {threshold}:")
print(f"  Number of correct predictions with high uncertainty: {correct_above}")
print(f"  Number of incorrect predictions with high uncertainty: {incorrect_above}")

