import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur',
    'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]
CIFAR10C_BASE_PATH = '/home/diversity_project/aaryaa/attacks/Cifar-10/cifar_npy_files'

class CIFAR10CDataset(Dataset):
    def __init__(self, images_path, labels_path, corruption_type, severity_level, transform=None):
        self.transform = transform
        
        try:
            full_images_path = os.path.join(images_path, f'{corruption_type}.npy')
            self.images = np.load(full_images_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {full_images_path} not found. Ensure CIFAR-10-C data is in the specified path.")
        
        try:
            self.labels = np.load(labels_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {labels_path} not found. Ensure CIFAR-10-C labels are in the specified path.")

        start_idx = severity_level * 10000
        end_idx = (severity_level + 1) * 10000
        self.images_subset = self.images[start_idx:end_idx]
        self.labels_subset = self.labels[start_idx:end_idx]

    def __len__(self):
        return len(self.images_subset)

    def __getitem__(self, idx):
        image = self.images_subset[idx]
        label = self.labels_subset[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    'cifar10_resnet56',
    pretrained=True
).eval().to(device)

#calculats all the uncertainty values 
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
    
    ratio_confidence = top_two_probabilities[:, 1] / (top_two_probabilities[:, 0] + 1e-9)
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

def plot_uncertainty_vs_correct_counts(uncertainty_scores, is_correct, title, x_label, num_bins=20, save_dir="plots_corrupted"):
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

def main():
    BATCH_SIZE = 96

    #stores and prints all the predictions 
    all_preds_overall_list = []
    all_labels_overall_list = []
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

    mean_np_odin = np.array(CIFAR10_MEAN)
    std_np_odin = np.array(CIFAR10_STD)
    mean_torch_odin = torch.tensor(mean_np_odin, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std_torch_odin = torch.tensor(std_np_odin, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    criterion = nn.CrossEntropyLoss()

    print("Starting uncertainty metrics calculation for CIFAR-10-C...")

    for corruption in corruptions:
        for severity in range(5):
            print(f"Processing {corruption} severity {severity+1}...")
            
            try:
                dataset = CIFAR10CDataset(
                    images_path=CIFAR10C_BASE_PATH,
                    labels_path=os.path.join(CIFAR10C_BASE_PATH, 'labels.npy'),
                    corruption_type=corruption,
                    severity_level=severity,
                    transform=transform
                )
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            except FileNotFoundError as e:
                print(f"Skipping {corruption} severity {severity+1}: {e}")
                continue

            preds_batch_current_severity = []
            labels_batch_current_severity = []

            for imgs_batch, lbls_batch in tqdm(dataloader, desc=f'{corruption} severity {severity+1}'):
                imgs_batch = imgs_batch.to(device)
                lbls_batch = lbls_batch.to(device)

                with torch.no_grad():
                    logits_no_grad = model(imgs_batch)
                    probabilities_no_grad = F.softmax(logits_no_grad, dim=1)

                    preds = probabilities_no_grad.argmax(dim=1).cpu().numpy()
                    
                    lc_batch = calculate_least_confidence(probabilities_no_grad).cpu().numpy()
                    mc_batch = calculate_margin_confidence(probabilities_no_grad).cpu().numpy()
                    rc_batch = calculate_ratio_confidence(probabilities_no_grad).cpu().numpy()
                    msp_batch = calculate_msp(probabilities_no_grad).cpu().numpy()
                    alpha_batch = calculate_doctor(probabilities_no_grad, 'alpha').cpu().numpy()
                    beta_batch = calculate_doctor(probabilities_no_grad, 'beta').cpu().numpy()
                    energy_batch = calculate_energy(logits_no_grad).cpu().numpy()
                    max_logit_batch = calculate_max_logit(logits_no_grad).cpu().numpy()

                imgs_odin = imgs_batch.clone().detach().requires_grad_(True)
                
                logits_odin_initial = model(imgs_odin)
                outputs_scaled_by_temper = logits_odin_initial / temper

                probabilities_for_label = F.softmax(logits_odin_initial.detach(), dim=1)
                max_indices_temp = torch.argmax(probabilities_for_label, dim=1)
                labels_odin_batch = max_indices_temp.to(imgs_odin.device)

                loss = criterion(outputs_scaled_by_temper, labels_odin_batch)

                model.zero_grad()
                if imgs_odin.grad is not None:
                    imgs_odin.grad.zero_()

                loss.backward()

                gradient = torch.sign(imgs_odin.grad.data)
                gradient_scaled = gradient / std_torch_odin 

                tempInputs = torch.add(imgs_odin.data, -noiseMagnitude1, gradient_scaled)

                with torch.no_grad():
                    outputs_odin_final = model(tempInputs)
                    outputs_odin_final_scaled = outputs_odin_final / temper
                    odin_uncertainty_batch = torch.max(F.softmax(outputs_odin_final_scaled, dim=1), dim=1).values.cpu().numpy()

                preds_batch_current_severity.extend(preds)
                labels_batch_current_severity.extend(lbls_batch.cpu().numpy())

                all_least_confidences.extend(lc_batch)
                all_margin_confidences.extend(mc_batch)
                all_ratio_confidences.extend(rc_batch)
                all_msps.extend(msp_batch)
                all_doctor_alpha.extend(alpha_batch)
                all_doctor_beta.extend(beta_batch)
                all_odin_uncertainties.extend(1 - odin_uncertainty_batch)
                all_energies.extend(energy_batch)
                all_maxlogit_uncertainties.extend(max_logit_batch)

            acc_severity = accuracy_score(labels_batch_current_severity, preds_batch_current_severity)
            prec_severity = precision_score(labels_batch_current_severity, preds_batch_current_severity, average='weighted', zero_division=0)
            rec_severity = recall_score(labels_batch_current_severity, preds_batch_current_severity, average='weighted', zero_division=0)
            f1_severity = f1_score(labels_batch_current_severity, preds_batch_current_severity, average='weighted', zero_division=0)
            print(f'{corruption} severity {severity+1}: Accuracy={acc_severity:.4f}, Precision={prec_severity:.4f}, Recall={rec_severity:.4f}, F1={f1_severity:.4f}')

            all_preds_overall.extend(preds_batch_current_severity)
            all_labels_overall.extend(labels_batch_current_severity)

    all_least_confidences = np.array(all_least_confidences)
    all_margin_confidences = np.array(all_margin_confidences)
    all_ratio_confidences = np.array(all_ratio_confidences)
    all_msps = np.array(all_msps)
    all_doctor_alpha = np.array(all_doctor_alpha)
    all_doctor_beta = np.array(all_doctor_beta)
    all_odin_uncertainties = np.array(all_odin_uncertainties)
    all_energies = np.array(all_energies)
    all_maxlogit_uncertainties = np.array(all_maxlogit_uncertainties)

    is_correct_overall = (np.array(all_preds_overall) == np.array(all_labels_overall)).astype(int)

    auc_scores = {}
    
    all_uncertainty_metrics = {
        'LC': all_least_confidences,
        'MC': all_margin_confidences,
        'RC': all_ratio_confidences,
        'MSP': 1 - all_msps,
        'Doctor-α': all_doctor_alpha,
        'Doctor-β': all_doctor_beta,
        'Energy': all_energies,
        'MaxLogit': -all_maxlogit_uncertainties,
        'ODIN': all_odin_uncertainties
    }

    for metric_name, uncertainty_values in all_uncertainty_metrics.items():
        try:
            auc = roc_auc_score(is_correct_overall, -uncertainty_values)
            auc_scores[metric_name] = auc
            print(f"{metric_name} AUC = {auc:.4f}")
        except ValueError as e:
            print(f"Skipping {metric_name} due to error: {e}")
            
    metric_order = ['LC', 'MC', 'RC', 'MSP', 'Doctor-α', 'Doctor-β', 'Energy', 'MaxLogit', 'ODIN']
    auc_values = [auc_scores[m] for m in metric_order if m in auc_scores]

    plt.figure(figsize=(10, 5))
    plt.plot(metric_order, auc_values, marker='o', label='ResNet-50 on CIFAR-10-C')
    plt.title("AUC of Uncertainty Metrics for Correctness Prediction on CIFAR-10-C")
    plt.xlabel("Uncertainty Metric")
    plt.ylabel("AUC (Higher is Better)")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("auc_scores_uncertainty_cifar-c.png")

    print("\n--- Generating and Saving Plots ---")
    plot_uncertainty_vs_correct_counts(
        1 - all_msps,
        is_correct_overall,
        'Max Softmax Probability (MSP) Uncertainty vs. Correct/Incorrect Predictions (CIFAR-10-C)',
        'MSP Uncertainty (1 - Max Probability)'
    )
    plot_uncertainty_vs_correct_counts(
        all_least_confidences,
        is_correct_overall,
        'Least Confidence Uncertainty vs. Correct/Incorrect Predictions (CIFAR-10-C)',
        'Least Confidence (1 - Max Probability)'
    )
    plot_uncertainty_vs_correct_counts(
        all_margin_confidences,
        is_correct_overall,
        'Margin Confidence vs. Correct/Incorrect Predictions (CIFAR-10-C)',
        'Margin Confidence (Prob1 - Prob2)'
    )
    plot_uncertainty_vs_correct_counts(
        all_ratio_confidences,
        is_correct_overall,
        'Ratio Confidence vs. Correct/Incorrect Predictions (CIFAR-10-C)',
        'Ratio Confidence (Prob2 / Prob1)'
    )
    plot_uncertainty_vs_correct_counts(
        all_doctor_alpha,
        is_correct_overall,
        'Doctor Alpha Uncertainty vs. Correct/Incorrect Predictions (CIFAR-10-C)',
        'Doctor Alpha'
    )
    plot_uncertainty_vs_correct_counts(
        all_doctor_beta,
        is_correct_overall,
        'Doctor Beta Uncertainty vs. Correct/Incorrect Predictions (CIFAR-10-C)',
        'Doctor Beta'
    )
    plot_uncertainty_vs_correct_counts(
        all_odin_uncertainties,
        is_correct_overall,
        'ODIN Uncertainty vs. Correct/Incorrect Predictions (CIFAR-10-C)',
        'ODIN Uncertainty (1 - ODIN Score)'
    )
    plot_uncertainty_vs_correct_counts(
        all_energies,
        is_correct_overall,
        'Energy Uncertainty vs. Correct/Incorrect Predictions (CIFAR-10-C)',
        'Energy Score'
    )
    plot_uncertainty_vs_correct_counts(
        all_maxlogit_uncertainties,
        is_correct_overall,
        'Max Logit vs. Correct/Incorrect Predictions (CIFAR-10-C)',
        'Max Logit'
    )

    overall_acc = accuracy_score(all_labels_overall, all_preds_overall)
    overall_prec = precision_score(all_labels_overall, all_preds_overall, average='weighted', zero_division=0)
    overall_rec = recall_score(all_labels_overall, all_preds_overall, average='weighted', zero_division=0)
    overall_f1 = f1_score(all_labels_overall, all_preds_overall, average='weighted', zero_division=0)

    print(f'\n--- Overall Metrics (across all corruptions and severities) ---')
    print(f'Accuracy={overall_acc:.4f}, Precision={overall_prec:.4f}, Recall={overall_rec:.4f}, F1={overall_f1:.4f}')

    print("\n--- Uncertainty Metrics (Overall) ---")
    print(f"Least Confidence: Avg={np.mean(all_least_confidences):.4f}, Min={np.min(all_least_confidences):.4f}, Max={np.max(all_least_confidences):.4f}, Std={np.std(all_least_confidences):.4f}")
    print(f"Margin Confidence: Avg={np.mean(all_margin_confidences):.4f}, Min={np.min(all_margin_confidences):.4f}, Max={np.max(all_margin_confidences):.4f}, Std={np.std(all_margin_confidences):.4f}")
    print(f"Ratio Confidence: Avg={np.mean(all_ratio_confidences):.4f}, Min={np.min(all_ratio_confidences):.4f}, Max={np.max(all_ratio_confidences):.4f}, Std={np.std(all_ratio_confidences):.4f}")
    print(f"MSP (Confidence): Avg={np.mean(all_msps):.4f}")
    print(f"Doctor Alpha: Avg={np.mean(all_doctor_alpha):.4f}, Min={np.min(all_doctor_alpha):.4f}, Max={np.max(all_doctor_alpha):.4f}, Std={np.std(all_doctor_alpha):.4f}")
    print(f"Doctor Beta: Avg={np.mean(all_doctor_beta):.4f}, Min={np.min(all_doctor_beta):.4f}, Max={np.max(all_doctor_beta):.4f}, Std={np.std(all_doctor_beta):.4f}")
    print(f"ODIN Uncertainty: Avg={np.mean(all_odin_uncertainties):.4f}, Min={np.min(all_odin_uncertainties):.4f}, Max={np.max(all_odin_uncertainties):.4f}, Std={np.std(all_odin_uncertainties):.4f}")
    print(f"Energy Uncertainty: Avg={np.mean(all_energies):.4f}")
    print(f"Max Logit: Avg={np.mean(all_maxlogit_uncertainties):.4f}")


if __name__ == "__main__":
    main()
