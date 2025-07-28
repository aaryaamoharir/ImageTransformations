import tensorflow_datasets as tfds
import torchvision.models as models
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ds = tfds.load('imagenet_a', split='test', shuffle_files=False)
print("loaded imagenet")

class ImageNetADataset(Dataset):
    def __init__(self, tfds_dataset, transform=None):
        self.data = list(tfds.as_numpy(tfds_dataset))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image = example['image']
        label = example['label']
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to(device)
model.eval()

dataset = ImageNetADataset(ds, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

def calculate_least_confidence(probabilities_batch):
    if probabilities_batch.numel() == 0:
        return torch.tensor([])
    most_confident_probability = torch.max(probabilities_batch, dim=1).values
    least_confidence = 1 - most_confident_probability
    return least_confidence

def calculate_margin_confidence(probabilities_batch):
    if probabilities_batch.shape[1] < 2:
        return torch.tensor([])
    top_two_probabilities = torch.topk(probabilities_batch, 2, dim=1).values
    margin_confidence = top_two_probabilities[:, 0] - top_two_probabilities[:, 1]
    return margin_confidence

def calculate_ratio_confidence(probabilities_batch):
    if probabilities_batch.shape[1] < 2:
        return torch.tensor([])
    top_two_probabilities = torch.topk(probabilities_batch, 2, dim=1).values
    ratio_confidence = top_two_probabilities[:, 0] / (top_two_probabilities[:, 1] + 1e-9)
    return ratio_confidence

def calculate_msp(probabilities_batch):
    return torch.max(probabilities_batch, dim=1).values

def calculate_doctor(probabilities_batch, type_val):
    g_hat = torch.sum(probabilities_batch**2, dim=1)
    pred_error_prob = 1.0 - torch.max(probabilities_batch, dim=1).values
    if type_val == 'alpha':
        doctor_score = torch.where(g_hat != 0, (1.0 - g_hat) / g_hat, torch.tensor(float('inf')).to(g_hat.device))
    else:
        denominator = (1.0 - pred_error_prob)
        doctor_score = torch.where(denominator != 0, pred_error_prob / denominator, torch.tensor(float('inf')).to(denominator.device))
    return doctor_score

def calculate_max_logit(logits_batch):
    max_logits, _ = torch.max(logits_batch, dim=1)
    return max_logits

def calculate_energy(logits_batch, temperature=1.0):
    energy_score = -temperature * torch.logsumexp(logits_batch / temperature, dim=-1)
    return energy_score

def plot_uncertainty_vs_correct_counts(uncertainty_scores, is_correct, title, x_label, num_bins=20, save_dir="plots_imagenet_a"):
    os.makedirs(save_dir, exist_ok=True)
    finite_mask = np.isfinite(uncertainty_scores)
    finite_uncertainty_scores = uncertainty_scores[finite_mask]
    correct_predictions_mask = is_correct[finite_mask]
    if len(finite_uncertainty_scores) == 0:
        print(f"No finite uncertainty scores to plot for '{title}'. Skipping plot.")
        return
    min_uncertainty = np.min(finite_uncertainty_scores)
    max_uncertainty = np.max(finite_uncertainty_scores)
    if min_uncertainty == max_uncertainty:
        bins = np.array([min_uncertainty, min_uncertainty + 1e-6])
    else:
        bins = np.linspace(min_uncertainty, max_uncertainty, num_bins + 1)
    bin_indices = np.digitize(finite_uncertainty_scores, bins)
    correct_counts_per_bin = np.zeros(num_bins)
    incorrect_counts_per_bin = np.zeros(num_bins)
    for i in range(len(finite_uncertainty_scores)):
        bin_idx = bin_indices[i] - 1
        if 0 <= bin_idx < num_bins:
            if correct_predictions_mask[i]:
                correct_counts_per_bin[bin_idx] += 1
            else:
                incorrect_counts_per_bin[bin_idx] += 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.figure(figsize=(12, 7))
    bar_width = (bins[1]-bins[0]) * 0.4
    plt.bar(bin_centers - bar_width/2, correct_counts_per_bin, width=bar_width, color='skyblue', label='Correct Images')
    plt.bar(bin_centers + bar_width/2, incorrect_counts_per_bin, width=bar_width, color='salmon', label='Incorrect Images')
    plt.xlabel(x_label)
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(save_dir, f"{title.replace(' ', '_').replace('.', '').replace('/', '_').replace(':', '')}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved: {filename}")

all_preds = []
all_labels = []
all_least_confidences = []
all_margin_confidences = []
all_ratio_confidences = []
all_msps = []
all_doctor_alpha = []
all_doctor_beta = []
all_energies = []
all_maxlogit_uncertainties = []

top5_correct = 0
total_samples = 0
print("about to go into for loop")
for img_batch, label_batch in dataloader:
    img_batch = img_batch.to(device)
    label_batch = label_batch.to(device)

    with torch.no_grad():
        logits_batch = model(img_batch)
        probs_batch = F.softmax(logits_batch, dim=1)

        lc_batch = calculate_least_confidence(probs_batch).cpu().numpy()
        mc_batch = calculate_margin_confidence(probs_batch).cpu().numpy()
        rc_batch = calculate_ratio_confidence(probs_batch).cpu().numpy()
        msp_batch = calculate_msp(probs_batch).cpu().numpy()
        alpha_batch = calculate_doctor(probs_batch, 'alpha').cpu().numpy()
        beta_batch = calculate_doctor(probs_batch, 'beta').cpu().numpy()
        energy_batch = calculate_energy(logits_batch).cpu().numpy()
        max_logit_batch = calculate_max_logit(logits_batch).cpu().numpy()

        all_least_confidences.extend(lc_batch)
        all_margin_confidences.extend(mc_batch)
        all_ratio_confidences.extend(rc_batch)
        all_msps.extend(msp_batch)
        all_doctor_alpha.extend(alpha_batch)
        all_doctor_beta.extend(beta_batch)
        all_energies.extend(energy_batch)
        all_maxlogit_uncertainties.extend(max_logit_batch)

        top1_preds = torch.argmax(probs_batch, dim=1)
        all_preds.extend(top1_preds.cpu().numpy())
        all_labels.extend(label_batch.cpu().numpy())

        top5_preds = torch.topk(probs_batch, k=5, dim=1).indices
        for i in range(label_batch.size(0)):
            if label_batch[i] in top5_preds[i]:
                top5_correct += 1
        total_samples += label_batch.size(0)

print("computing metrics")
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
top5_acc = top5_correct / total_samples

print(f'Top-1 Accuracy: {acc:.4f}')
print(f'Top-5 Accuracy: {top5_acc:.4f}')
print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')

all_least_confidences = np.array(all_least_confidences)
all_margin_confidences = np.array(all_margin_confidences)
all_ratio_confidences = np.array(all_ratio_confidences)
all_msps = np.array(all_msps)
all_doctor_alpha = np.array(all_doctor_alpha)
all_doctor_beta = np.array(all_doctor_beta)
all_energies = np.array(all_energies)
all_maxlogit_uncertainties = np.array(all_maxlogit_uncertainties)

is_correct_overall = (np.array(all_preds) == np.array(all_labels)).astype(int)

auc_scores = {}
all_uncertainty_metrics = {
    'LC': all_least_confidences,
    'MC': all_margin_confidences,
    'RC': all_ratio_confidences,
    'MSP': 1 - all_msps,
    'Doctor-α': all_doctor_alpha,
    'Doctor-β': all_doctor_beta,
    'Energy': all_energies,
    'MaxLogit': -all_maxlogit_uncertainties
}

for metric_name, uncertainty_values in all_uncertainty_metrics.items():
    try:
        auc = roc_auc_score(is_correct_overall, -uncertainty_values)
        auc_scores[metric_name] = auc
        print(f"{metric_name} AUC = {auc:.4f}")
    except ValueError as e:
        print(f"Skipping {metric_name} due to error: {e}")
        
metric_order = ['LC', 'MC', 'RC', 'MSP', 'Doctor-α', 'Doctor-β', 'Energy', 'MaxLogit']
auc_values = [auc_scores[m] for m in metric_order if m in auc_scores]

plt.figure(figsize=(10, 5))
plt.plot(metric_order, auc_values, marker='o', label='ResNet-50 on ImageNet-A')
plt.title("AUC of Uncertainty Metrics for Correctness Prediction on ImageNet-A")
plt.xlabel("Uncertainty Metric")
plt.ylabel("AUC (Higher is Better)")
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("auc_scores_uncertainty_imagenet_a.png")

print("\n--- Generating and Saving Plots ---")
plot_uncertainty_vs_correct_counts(
    1 - all_msps,
    is_correct_overall,
    'Max Softmax Probability (MSP) Uncertainty vs. Correct/Incorrect Predictions (ImageNet-A)',
    'MSP Uncertainty (1 - Max Probability)'
)
plot_uncertainty_vs_correct_counts(
    all_least_confidences,
    is_correct_overall,
    'Least Confidence Uncertainty vs. Correct/Incorrect Predictions (ImageNet-A)',
    'Least Confidence (1 - Max Probability)'
)
plot_uncertainty_vs_correct_counts(
    all_margin_confidences,
    is_correct_overall,
    'Margin Confidence vs. Correct/Incorrect Predictions (ImageNet-A)',
    'Margin Confidence (Prob1 - Prob2)'
)
plot_uncertainty_vs_correct_counts(
    all_ratio_confidences,
    is_correct_overall,
    'Ratio Confidence vs. Correct/Incorrect Predictions (ImageNet-A)',
    'Ratio Confidence (Prob2 / Prob1)'
)
plot_uncertainty_vs_correct_counts(
    all_doctor_alpha,
    is_correct_overall,
    'Doctor Alpha Uncertainty vs. Correct/Incorrect Predictions (ImageNet-A)',
    'Doctor Alpha'
)
plot_uncertainty_vs_correct_counts(
    all_doctor_beta,
    is_correct_overall,
    'Doctor Beta Uncertainty vs. Correct/Incorrect Predictions (ImageNet-A)',
    'Doctor Beta'
)
plot_uncertainty_vs_correct_counts(
    all_energies,
    is_correct_overall,
    'Energy Uncertainty vs. Correct/Incorrect Predictions (ImageNet-A)',
    'Energy Score'
)
plot_uncertainty_vs_correct_counts(
    all_maxlogit_uncertainties,
    is_correct_overall,
    'Max Logit vs. Correct/Incorrect Predictions (ImageNet-A)',
    'Max Logit'
)

print("\nLeast Confidence Metrics:")
print(f"  → Average Least Confidence: {np.mean(all_least_confidences):.4f}")
print(f"  → Min Least Confidence: {np.min(all_least_confidences):.4f}")
print(f"  → Max Least Confidence: {np.max(all_least_confidences):.4f}")
print(f"  → Std Dev of Least Confidence: {np.std(all_least_confidences):.4f}")

print("\nMargin Confidence Metrics:")
print(f"  → Average Margin Confidence: {np.mean(all_margin_confidences):.4f}")
print(f"  → Min Margin Confidence: {np.min(all_margin_confidences):.4f}")
print(f"  → Max Margin Confidence: {np.max(all_margin_confidences):.4f}")
print(f"  → Std Dev of Margin Confidence: {np.std(all_margin_confidences):.4f}")

print("\nRatio Confidence Metrics:")
print(f"  → Average Ratio Confidence: {np.mean(all_ratio_confidences):.4f}")
print(f"  → Min Ratio Confidence: {np.min(all_ratio_confidences):.4f}")
print(f"  → Max Ratio Confidence: {np.max(all_ratio_confidences):.4f}")
print(f"  → Std Dev of Ratio Confidence: {np.std(all_ratio_confidences):.4f}")

print(f"\nAverage MSP: {np.mean(all_msps):.4f}")

print("\nDoctor Alpha Metrics:")
print(f"  → Average Doctor Alpha: {np.mean(all_doctor_alpha):.4f}")
print(f"  → Min Doctor Alpha: {np.min(all_doctor_alpha):.4f}")
print(f"  → Max Doctor Alpha: {np.max(all_doctor_alpha):.4f}")
print(f"  → Std Dev of Doctor Alpha: {np.std(all_doctor_alpha):.4f}")

print("\nDoctor Beta Metrics:")
print(f"  → Average Doctor Beta: {np.mean(all_doctor_beta):.4f}")
print(f"  → Min Doctor Beta: {np.min(all_doctor_beta):.4f}")
print(f"  → Max Doctor Beta: {np.max(all_doctor_beta):.4f}")
print(f"  → Std Dev of Doctor Beta: {np.std(all_doctor_beta):.4f}")

print(f"\nEnergy Uncertainty: Avg={np.mean(all_energies):.4f}")

print(f"\nMax Logit: Avg={np.mean(all_maxlogit_uncertainties):.4f}")
