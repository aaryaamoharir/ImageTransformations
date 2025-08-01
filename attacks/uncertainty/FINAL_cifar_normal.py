import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# Removed tensorflow_datasets as it's no longer needed
# import tensorflow_datasets as tfds

# No need for load_cifar10c function as we'll use torchvision
# def load_cifar10c():
#     ds = tfds.load(
#         'cifar10',
#         split='test',
#         shuffle_files=False,
#         as_supervised=True,
#     )
#     return ds

# Modified preprocess function to work with PIL Images from torchvision
def preprocess_torch_dataset(image, label):
    # Normalize with CIFAR-10 mean and std after converting to tensor
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform(image), int(label)

def calculate_least_confidence(probabilities):
    if probabilities.numel() == 0:
        raise ValueError("Probabilities tensor cannot be empty.")
    most_confident_probability = torch.max(probabilities).item()
    least_confidence = 1 - most_confident_probability
    return least_confidence

def calculate_margin_confidence(probabilities):
    if probabilities.numel() < 2:
        raise ValueError("Probabilities tensor must have at least two elements for margin confidence.")
    top_two_probabilities = torch.topk(probabilities, 2).values
    margin_confidence = top_two_probabilities[0].item() - top_two_probabilities[1].item()
    return margin_confidence

def calculate_ratio_confidence(probabilities):
    if probabilities.numel() < 2:
        raise ValueError("Probabilities tensor must have at least two elements for ratio confidence.")
    top_two_probabilities = torch.topk(probabilities, 2).values
    if top_two_probabilities[1].item() == 0:
        return float('inf')
    ratio_confidence = top_two_probabilities[1].item() / top_two_probabilities[0].item()
    return ratio_confidence

def calculate_msp(probabilities):
    return torch.max(probabilities).item()

def calculate_doctor(probabilities, type):
    g_hat = torch.sum(probabilities**2).item()
    pred_error_prob = 1.0 - torch.max(probabilities).item()
    if g_hat == 0:
        return float('inf')
    if (1 - pred_error_prob) == 0:
        return float ('inf')
    if (type == 'alpha'):
        return (1.0 - g_hat) / g_hat
    else:
        return pred_error_prob / (1.0 - pred_error_prob)

def calculate_max_logit(logits):
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    max_logits, _ = torch.max(logits, dim=1)
    return max_logits.cpu().numpy()

def calculate_energy(logits, temperature=1.0):
    if logits.numel() == 0:
        raise ValueError("Logits tensor cannot be empty.")
    energy_score = -temperature * torch.logsumexp(logits / temperature, dim=-1)
    return energy_score.item()

def plot_uncertainty_vs_correct_counts(uncertainty_scores, is_correct, title, x_label, num_bins=20, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    finite_mask = np.isfinite(uncertainty_scores)
    finite_uncertainty_scores = uncertainty_scores[finite_mask]
    correct_predictions = is_correct[finite_mask]

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
    total_counts_per_bin = np.zeros(num_bins)

    for i in range(len(finite_uncertainty_scores)):
        bin_idx = bin_indices[i] - 1
        if 0 <= bin_idx < num_bins:
            total_counts_per_bin[bin_idx] += 1
            if correct_predictions[i]:
                correct_counts_per_bin[bin_idx] += 1

    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(12, 7))

    plt.bar(bin_centers, correct_counts_per_bin, width=(bins[1]-bins[0])*0.4, color='skyblue', label='Correct Images')

    plt.plot(bin_centers, total_counts_per_bin, color='red', linestyle='--', marker='o', markersize=5, label='Total Images in Bin', alpha=0.7)

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

def main():
    # Load CIFAR-10 test dataset from torchvision
    # No need for shuffle=False here as we take a fixed number of samples.
    # download=True will download the dataset if not already present.
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    all_preds, all_labels = [], []
    all_least_confidences = []
    all_margin_confidences = []
    all_ratio_confidences = []
    all_msps = []
    all_doctor_alpha = []
    all_doctor_beta = []
    all_odin_uncertainties = []
    all_energies = []
    all_maxlogit_uncertainties = []

    temper = 1000
    noiseMagnitude1 = 0.0014

    mean_np = np.array([125.3, 123.0, 113.9]) / 255.0
    std_np = np.array([63.0, 62.1, 66.7]) / 255.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean_torch = torch.tensor(mean_np, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std_torch = torch.tensor(std_np, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    criterion = nn.CrossEntropyLoss()

    # tested over the first 10 images of the PyTorch dataset
    # cifar10_test is directly iterable
    #for i in tqdm(range(min(10, len(cifar10_test))), desc="Processing images"):
    for i in tqdm(range(len(cifar10_test)), desc="Processing images"):
        img_pil, label = cifar10_test[i] # Get PIL Image and label directly

        # Use the modified preprocess_torch_dataset
        img_tensor, lbl = preprocess_torch_dataset(img_pil, label)
        img_no_grad = img_tensor.unsqueeze(0).to(next(model.parameters()).device)

        with torch.no_grad():
            logits_no_grad = model(img_no_grad)
            probabilities_no_grad = F.softmax(logits_no_grad, dim=1)
            pred = probabilities_no_grad.argmax(dim=1).cpu().item()
            lc = calculate_least_confidence(probabilities_no_grad.squeeze(0))
            mc = calculate_margin_confidence(probabilities_no_grad.squeeze(0))
            rc = calculate_ratio_confidence(probabilities_no_grad.squeeze(0))
            msp = calculate_msp(probabilities_no_grad.squeeze(0))
            alpha = calculate_doctor(probabilities_no_grad.squeeze(0), 'alpha')
            beta = calculate_doctor(probabilities_no_grad.squeeze(0), 'beta')
            energy = calculate_energy(logits_no_grad.squeeze(0))
            max_logit = calculate_max_logit(logits_no_grad)
            max_logit = max_logit[0]

        img_odin_pil, _ = cifar10_test[i] # Get PIL Image again for ODIN
        img_odin, _ = preprocess_torch_dataset(img_odin_pil, label) # Preprocess for ODIN
        img_odin = img_odin.unsqueeze(0).to(next(model.parameters()).device).requires_grad_(True)

        logits_odin_initial = model(img_odin)

        outputs_scaled_by_temper = logits_odin_initial / temper

        probabilities_for_label = F.softmax(logits_odin_initial.detach(), dim=1).cpu().numpy()[0]
        maxIndexTemp = np.argmax(probabilities_for_label)
        labels_odin = torch.LongTensor([maxIndexTemp]).to(img_odin.device) # Renamed to avoid conflict

        loss = criterion(outputs_scaled_by_temper, labels_odin)

        model.zero_grad()
        if img_odin.grad is not None:
            img_odin.grad.zero_()

        loss.backward()

        gradient = torch.sign(img_odin.grad.data)
        gradient_scaled = gradient / std_torch

        tempInputs = torch.add(img_odin.data, -noiseMagnitude1, gradient_scaled)

        with torch.no_grad():
            outputs_odin_final = model(tempInputs)
            outputs_odin_final_scaled = outputs_odin_final / temper
            odin_probabilities = F.softmax(outputs_odin_final_scaled, dim=1)
            odin_uncertainty = torch.max(odin_probabilities).cpu().item()

        print(f"LC: {lc:.4f}, MC: {mc:.4f}, RC: {rc:.4f}, MSP: {msp:.4f}, Doctor Alpha: {alpha:.4f}, Doctor Beta: {beta:.4f}, ODIN Uncertainty: {odin_uncertainty:.4f}, Energy: {energy:.4f}, Max Logit: {max_logit:.4f}")

        all_preds.append(pred)
        all_labels.append(int(lbl))
        all_least_confidences.append(lc)
        all_margin_confidences.append(mc)
        all_ratio_confidences.append(rc)
        all_msps.append(msp)
        all_doctor_alpha.append(alpha)
        all_doctor_beta.append(beta)
        all_odin_uncertainties.append(odin_uncertainty)
        all_energies.append(energy)
        all_maxlogit_uncertainties.append(max_logit)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_least_confidences = np.array(all_least_confidences)
    all_margin_confidences = np.array(all_margin_confidences)
    all_ratio_confidences = np.array(all_ratio_confidences)
    all_msps = np.array(all_msps)
    all_doctor_alpha = np.array(all_doctor_alpha)
    all_doctor_beta = np.array(all_doctor_beta)
    all_odin_uncertainties = np.array(all_odin_uncertainties)
    all_energies = np.array(all_energies)
    all_maxlogit_uncertainties = np.array(all_maxlogit_uncertainties)

    is_correct = (all_preds == all_labels)

    print("\n--- Generating and Saving Plots ---")
    plot_uncertainty_vs_correct_counts(
        1 - all_msps,
        is_correct,
        'Max Softmax Probability (MSP) Uncertainty vs. Correct Predictions',
        'MSP Uncertainty (1 - Max Probability)'
    )
    plot_uncertainty_vs_correct_counts(
        all_least_confidences,
        is_correct,
        'Least Confidence Uncertainty vs. Correct Predictions',
        'Least Confidence (1 - Max Probability)'
    )
    plot_uncertainty_vs_correct_counts(
        all_margin_confidences,
        is_correct,
        'Margin Confidence Uncertainty vs. Correct Predictions',
        'Margin Confidence (Prob1 - Prob2)'
    )
    plot_uncertainty_vs_correct_counts(
        all_ratio_confidences,
        is_correct,
        'Ratio Confidence Uncertainty vs. Correct Predictions',
        'Ratio Confidence (Prob2 / Prob1)'
    )
    plot_uncertainty_vs_correct_counts(
        all_doctor_alpha,
        is_correct,
        'Doctor Alpha Uncertainty vs. Correct Predictions',
        'Doctor Alpha'
    )
    plot_uncertainty_vs_correct_counts(
        all_doctor_beta,
        is_correct,
        'Doctor Beta Uncertainty vs. Correct Predictions',
        'Doctor Beta'
    )
    plot_uncertainty_vs_correct_counts(
        1 - all_odin_uncertainties,
        is_correct,
        'ODIN Uncertainty vs. Correct Predictions',
        'ODIN Uncertainty (1 - ODIN Score)'
    )
    plot_uncertainty_vs_correct_counts(
        all_energies,
        is_correct,
        'Energy Uncertainty vs. Correct Predictions',
        'Energy Score'
    )
    plot_uncertainty_vs_correct_counts(
        all_maxlogit_uncertainties,
        is_correct,
        'Max Logit Uncertainty vs. Correct Predictions',
        'Max Logit'
    )

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print("\nCIFAR‑10‑C Evaluation with ResNet56:")
    print(f"  → Accuracy : {acc:.4f}")
    print(f"  → Precision: {prec:.4f}")
    print(f"  → Recall   : {rec:.4f}")
    print(f"  → F1‑Score : {f1:.4f}")

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

    print("\nODIN Uncertainty Metrics:")
    print(f"  → Average ODIN Uncertainty: {np.mean(all_odin_uncertainties):.4f}")
    print(f"  → Min ODIN Uncertainty: {np.min(all_odin_uncertainties):.4f}")
    print(f"  → Max ODIN Uncertainty: {np.max(all_odin_uncertainties):.4f}")
    print(f"  → Std Dev of ODIN Uncertainty: {np.std(all_odin_uncertainties):.4f}")

    print("\nEnergy uncertainty Metrics:")
    print(f" -> Average Energy Unvertainty: {np.mean(all_energies):.4f}")

    print("\nMaxLogits Uncertainty Metrics")
    print(f" -> Average Max-logits Uncertainty: {np.mean(all_maxlogit_uncertainties):.4f}")

    print("\n--- AUC Scores per Uncertainty Metric (Lower AUC is Better for Uncertainty) ---")

    try:
        auc_msp = roc_auc_score(1 - is_correct, 1 - all_msps)
        print(f"AUC for MSP: {auc_msp:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC for MSP: {e}. This might happen if there's only one class present in 'is_correct' for the first 10 images.")

    try:
        auc_lc = roc_auc_score(1 - is_correct, all_least_confidences)
        print(f"AUC for Least Confidence: {auc_lc:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC for Least Confidence: {e}")

    try:
        auc_mc = roc_auc_score(1 - is_correct, -all_margin_confidences)
        print(f"AUC for Margin Confidence: {auc_mc:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC for Margin Confidence: {e}")

    finite_ratio_confidences = all_ratio_confidences[np.isfinite(all_ratio_confidences)]
    finite_is_correct_for_ratio = is_correct[np.isfinite(all_ratio_confidences)]
    try:
        auc_rc = roc_auc_score(1 - finite_is_correct_for_ratio, -finite_ratio_confidences)
        print(f"AUC for Ratio Confidence: {auc_rc:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC for Ratio Confidence: {e}")

    finite_doctor_alpha = all_doctor_alpha[np.isfinite(all_doctor_alpha)]
    finite_is_correct_for_alpha = is_correct[np.isfinite(all_doctor_alpha)]
    try:
        auc_doctor_alpha = roc_auc_score(1 - finite_is_correct_for_alpha, finite_doctor_alpha)
        print(f"AUC for Doctor Alpha: {auc_doctor_alpha:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC for Doctor Alpha: {e}")

    finite_doctor_beta = all_doctor_beta[np.isfinite(all_doctor_beta)]
    finite_is_correct_for_beta = is_correct[np.isfinite(all_doctor_beta)]
    try:
        auc_doctor_beta = roc_auc_score(1 - finite_is_correct_for_beta, finite_doctor_beta)
        print(f"AUC for Doctor Beta: {auc_doctor_beta:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC for Doctor Beta: {e}")

    try:
        auc_odin = roc_auc_score(1 - is_correct, 1 - all_odin_uncertainties)
        print(f"AUC for ODIN Uncertainty: {auc_odin:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC for ODIN Uncertainty: {e}")

    try:
        auc_energy = roc_auc_score(1 - is_correct, all_energies)
        print(f"AUC for Energy: {auc_energy:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC for Energy: {e}")

    try:
        auc_max_logit = roc_auc_score(1 - is_correct, -all_maxlogit_uncertainties)
        print(f"AUC for Max Logit: {auc_max_logit:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC for Max Logit: {e}")


if __name__ == "__main__":

