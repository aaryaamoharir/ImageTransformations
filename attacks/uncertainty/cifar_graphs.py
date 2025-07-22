import tensorflow_datasets as tfds
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn # For CrossEntropyLoss
import matplotlib.pyplot as plt # Import for plotting
import os # Import os module for path manipulation

def load_cifar10c():
    ds = tfds.load(
        'cifar10',
        split='test',
        shuffle_files=False,
        as_supervised=True,
    )
    return ds

def preprocess_tf_to_torch(image, label):
    img = T.ToTensor()(image)
    img = T.Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))(img)
    return img, int(label)

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
    if top_two_probabilities[0].item() == 0: # Check if the highest probability is zero
        return float('inf') # Avoid division by zero if both are zero or very small
    ratio_confidence = top_two_probabilities[1].item() / top_two_probabilities[0].item()
    return ratio_confidence

def calculate_msp(probabilities):
    return torch.max(probabilities).item()

def calculate_doctor(probabilities, type):
    g_hat = torch.sum(probabilities**2).item()
    pred_error_prob = 1.0 - torch.max(probabilities).item()
    
    # Handle edge cases to prevent division by zero or invalid values
    if g_hat == 0:
        return float('inf')
    if (1 - pred_error_prob) == 0:
        return float('inf')
        
    if (type == 'alpha'):
        return (1.0 - g_hat) / g_hat
    else: # type == 'beta'
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
    return energy_score.item() # Return the scalar value

def plot_uncertainty_vs_correct_counts(uncertainty_scores, is_correct, title, x_label, num_bins=20, save_dir="plots"):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Filter out infinite and NaN values from uncertainty_scores before binning
    finite_mask = np.isfinite(uncertainty_scores)
    finite_uncertainty_scores = uncertainty_scores[finite_mask]
    correct_predictions_mask = is_correct[finite_mask] # Boolean mask for correct predictions

    if len(finite_uncertainty_scores) == 0:
        print(f"No finite uncertainty scores to plot for '{title}'. Skipping plot.")
        return

    # Create bins for uncertainty scores
    min_uncertainty = np.min(finite_uncertainty_scores)
    max_uncertainty = np.max(finite_uncertainty_scores)

    # Handle cases where all values are the same or range is very small
    if min_uncertainty == max_uncertainty:
        bins = np.array([min_uncertainty, min_uncertainty + 1e-6]) # Create a tiny bin
    else:
        bins = np.linspace(min_uncertainty, max_uncertainty, num_bins + 1)

    # Digitize uncertainty scores into bins
    bin_indices = np.digitize(finite_uncertainty_scores, bins)

    # Initialize counts for correct, incorrect, and total in each bin
    correct_counts_per_bin = np.zeros(num_bins)
    incorrect_counts_per_bin = np.zeros(num_bins)
    total_counts_per_bin = np.zeros(num_bins) # For verification/debugging

    # Populate counts
    for i in range(len(finite_uncertainty_scores)):
        bin_idx = bin_indices[i] - 1 # Adjust to 0-indexed
        if 0 <= bin_idx < num_bins:
            total_counts_per_bin[bin_idx] += 1
            if correct_predictions_mask[i]:
                correct_counts_per_bin[bin_idx] += 1
            else:
                incorrect_counts_per_bin[bin_idx] += 1

    # Calculate bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(12, 7)) # Increased figure size for better readability

    bar_width = (bins[1]-bins[0]) * 0.4 # Adjust bar width for side-by-side

    # Plotting correct counts
    plt.bar(bin_centers - bar_width/2, correct_counts_per_bin, width=bar_width, color='skyblue', label='Correct Images')

    # Plotting incorrect counts
    plt.bar(bin_centers + bar_width/2, incorrect_counts_per_bin, width=bar_width, color='salmon', label='Incorrect Images')

    plt.xlabel(x_label)
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend() # Show legend for multiple plots
    plt.tight_layout()

    # Generate a clean filename from the title
    filename = os.path.join(save_dir, f"{title.replace(' ', '_').replace('.', '').replace('/', '_').replace(':', '')}.png")
    plt.savefig(filename)
    plt.close() # Close the plot to free memory
    print(f"Plot saved: {filename}")


def main():
    ds = load_cifar10c()
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

    # ODIN Hyperparameters
    temper = 1000
    noiseMagnitude1 = 0.0014

    mean_np = np.array([125.3, 123.0, 113.9]) / 255.0
    std_np = np.array([63.0, 62.1, 66.7]) / 255.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean_torch = torch.tensor(mean_np, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std_torch = torch.tensor(std_np, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    criterion = nn.CrossEntropyLoss()

    print("Starting model evaluation and uncertainty calculation...")
    for img_tf, label in tqdm(tfds.as_numpy(ds), desc="Processing images"):
        img_no_grad, lbl = preprocess_tf_to_torch(img_tf, label)
        img_no_grad = img_no_grad.unsqueeze(0).to(device) # Ensure img is on the correct device

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
            max_logit = max_logit[0] # max_logit is an array from numpy, get the scalar

        # ODIN calculation requires gradients
        img_odin, _ = preprocess_tf_to_torch(img_tf, label)
        img_odin = img_odin.unsqueeze(0).to(device).requires_grad_(True)

        logits_odin_initial = model(img_odin)
        outputs_scaled_by_temper = logits_odin_initial / temper

        # Detach and move to CPU for numpy conversion
        probabilities_for_label = F.softmax(logits_odin_initial.detach(), dim=1).cpu().numpy()[0]
        maxIndexTemp = np.argmax(probabilities_for_label)
        labels = torch.LongTensor([maxIndexTemp]).to(img_odin.device)

        loss = criterion(outputs_scaled_by_temper, labels)

        model.zero_grad()
        if img_odin.grad is not None:
            img_odin.grad.zero_()

        loss.backward()

        gradient = torch.sign(img_odin.grad.data)
        gradient_scaled = gradient / std_torch # Apply normalization scaling to gradient

        tempInputs = torch.add(img_odin.data, -noiseMagnitude1, gradient_scaled)

        with torch.no_grad():
            outputs_odin_final = model(tempInputs)
            outputs_odin_final_scaled = outputs_odin_final / temper
            odin_uncertainty = torch.max(F.softmax(outputs_odin_final_scaled, dim=1)).cpu().item() # ODIN score is max softmax prob

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

    all_least_confidences = np.array(all_least_confidences)
    all_margin_confidences = np.array(all_margin_confidences)
    all_ratio_confidences = np.array(all_ratio_confidences)
    all_msps = np.array(all_msps)
    all_doctor_alpha = np.array(all_doctor_alpha)
    all_doctor_beta = np.array(all_doctor_beta)
    all_odin_uncertainties = np.array(all_odin_uncertainties)
    all_energies = np.array(all_energies)
    all_maxlogit_uncertainties = np.array(all_maxlogit_uncertainties)

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    is_correct = (all_preds_np == all_labels_np)

    # --- Plotting and Saving ---
    print("\n--- Generating and Saving Plots ---")
    plot_uncertainty_vs_correct_counts(
        1 - all_msps, # MSP is confidence, so 1-MSP is uncertainty
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
        1 - all_odin_uncertainties, # ODIN provides a confidence score, so 1-score for uncertainty
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

    # Original evaluation metrics
    print("\n--- Evaluation Metrics ---")
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print("CIFAR‑10‑C Evaluation with ResNet56:")
    print(f"  → Accuracy : {acc:.4f}")
    print(f"  → Precision: {prec:.4f}")
    print(f"  → Recall   : {rec:.4f}")
    print(f"  → F1‑Score : {f1:.4f}")

    print("\n--- Uncertainty Metrics ---")
    print(f"Least Confidence: Avg={np.mean(all_least_confidences):.4f}, Min={np.min(all_least_confidences):.4f}, Max={np.max(all_least_confidences):.4f}, Std={np.std(all_least_confidences):.4f}")
    print(f"Margin Confidence: Avg={np.mean(all_margin_confidences):.4f}, Min={np.min(all_margin_confidences):.4f}, Max={np.max(all_margin_confidences):.4f}, Std={np.std(all_margin_confidences):.4f}")
    print(f"Ratio Confidence: Avg={np.mean(all_ratio_confidences):.4f}, Min={np.min(all_ratio_confidences):.4f}, Max={np.max(all_ratio_confidences):.4f}, Std={np.std(all_ratio_confidences):.4f}")
    print(f"MSP (Confidence): Avg={np.mean(all_msps):.4f}")
    print(f"Doctor Alpha: Avg={np.mean(all_doctor_alpha):.4f}, Min={np.min(all_doctor_alpha):.4f}, Max={np.max(all_doctor_alpha):.4f}, Std={np.std(all_doctor_alpha):.4f}")
    print(f"Doctor Beta: Avg={np.mean(all_doctor_beta):.4f}, Min={np.min(all_doctor_beta):.4f}, Max={np.max(all_doctor_beta):.4f}, Std={np.std(all_doctor_beta):.4f}")
    print(f"ODIN Uncertainty: Avg={np.mean(1 - all_odin_uncertainties):.4f}, Min={np.min(1 - all_odin_uncertainties):.4f}, Max={np.max(1 - all_odin_uncertainties):.4f}, Std={np.std(1 - all_odin_uncertainties):.4f}")
    print(f"Energy Uncertainty: Avg={np.mean(all_energies):.4f}")
    print(f"Max Logit Uncertainty: Avg={np.mean(all_maxlogit_uncertainties):.4f}")


if __name__ == "__main__":
    main()

