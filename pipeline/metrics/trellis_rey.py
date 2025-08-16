import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# --- Setup ---
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

# --- Entropy functions ---
def tsallis_entropy(probs, q=1.5):
    sum_pq = torch.sum(probs ** q, dim=1)
    return (1 - sum_pq) / (q - 1)

def renyi_entropy(probs, alpha=1.5):
    sum_pa = torch.sum(probs ** alpha, dim=1)
    return (1 / (1 - alpha)) * torch.log(sum_pa + 1e-12)

# --- Data collection ---
confidences = []
correctness = []
entropy_uncertainty = []
tsallis_uncertainty = []
renyi_uncertainty = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)

        # Compute entropies
        shannon_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
        tsallis_ent = tsallis_entropy(probs, q=1.5)
        renyi_ent = renyi_entropy(probs, alpha=1.5)

        confidences.extend(max_probs.cpu().numpy())
        correctness.extend((preds == targets).cpu().numpy())
        entropy_uncertainty.extend(shannon_entropy.cpu().numpy())
        tsallis_uncertainty.extend(tsallis_ent.cpu().numpy())
        renyi_uncertainty.extend(renyi_ent.cpu().numpy())

confidences = np.array(confidences)
correctness = np.array(correctness, dtype=bool)
entropy_uncertainty = np.array(entropy_uncertainty)
tsallis_uncertainty = np.array(tsallis_uncertainty)
renyi_uncertainty = np.array(renyi_uncertainty)

# --- ECE computation ---
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
print(f"ECE (using max softmax confidence): {ece_val:.4f}")

# --- AUC computation ---
def compute_auc(uncertainty, correctness):
    labels = (~correctness).astype(int)  # incorrect=1, correct=0
    return roc_auc_score(labels, uncertainty)

print(f"AUC Shannon entropy: {compute_auc(entropy_uncertainty, correctness):.4f}")
print(f"AUC Tsallis entropy: {compute_auc(tsallis_uncertainty, correctness):.4f}")
print(f"AUC Renyi entropy: {compute_auc(renyi_uncertainty, correctness):.4f}")

# --- Print counts above uncertainty threshold ---
def print_counts_above_threshold(uncertainty, correctness, threshold, name):
    above = uncertainty > threshold
    correct_above = np.sum(correctness[above])
    incorrect_above = np.sum(~correctness[above])
    print(f"\n{name} uncertainty > {threshold}:")
    print(f"  Correct predictions with high uncertainty: {correct_above}")
    print(f"  Incorrect predictions with high uncertainty: {incorrect_above}")

threshold = 0.3  # Adjust as needed for your data
print_counts_above_threshold(entropy_uncertainty, correctness, threshold, "Shannon")
print_counts_above_threshold(tsallis_uncertainty, correctness, threshold, "Tsallis")
print_counts_above_threshold(renyi_uncertainty, correctness, threshold, "Renyi")

# --- Plotting function ---
def plot_correct_incorrect_histogram(uncertainty_values, correctness, entropy_name, n_bins=20):
    bins = np.linspace(uncertainty_values.min(), uncertainty_values.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    correct_counts = np.zeros(n_bins, dtype=int)
    incorrect_counts = np.zeros(n_bins, dtype=int)

    for u, corr in zip(uncertainty_values, correctness):
        idx = np.searchsorted(bins, u, side='right') - 1
        if 0 <= idx < n_bins:
            if corr:
                correct_counts[idx] += 1
            else:
                incorrect_counts[idx] += 1

    plt.figure(figsize=(10, 6))
    width = (bin_centers[1] - bin_centers[0]) * 0.4

    plt.bar(bin_centers - width/2, correct_counts, width=width, label='Correct', color='g', alpha=0.7)
    plt.bar(bin_centers + width/2, incorrect_counts, width=width, label='Incorrect', color='r', alpha=0.7)

    plt.xlabel(f'{entropy_name} Entropy (Uncertainty)')
    plt.ylabel('Number of Samples')
    plt.title(f'Correct vs Incorrect Predictions by {entropy_name} Uncertainty Bin')
    plt.legend()
    plt.tight_layout()
    filename = f'uncertainty_correct_incorrect_histogram_{entropy_name.lower()}.png'
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.show()

# Plot Tsallis and Renyi histograms
plot_correct_incorrect_histogram(tsallis_uncertainty, correctness, 'Tsallis')
plot_correct_incorrect_histogram(renyi_uncertainty, correctness, 'Renyi')
