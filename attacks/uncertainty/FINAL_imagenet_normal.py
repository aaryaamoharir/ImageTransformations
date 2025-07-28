import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

val_dir = '/home/diversity_project/aaryaa/attacks/imagenet_data'
gt_file = '/home/diversity_project/aaryaa/attacks/imagenet_caffe_2012/val.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

with open(gt_file, 'r') as f:
    lines = f.read().splitlines()
    gt_info = [line.strip().split() for line in lines]
    val_images_raw = [info[0] for info in gt_info]
    gt_labels_map = {info[0]: int(info[1]) for info in gt_info}

class ImageNetValDataset(Dataset):
    def __init__(self, val_dir, val_images, gt_labels_map, transform=None):
        self.val_dir = val_dir
        self.val_images = val_images
        self.gt_labels_map = gt_labels_map
        self.transform = transform

    def __len__(self):
        return len(self.val_images)

    def __getitem__(self, idx):
        img_name = self.val_images[idx]
        img_path = os.path.join(self.val_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        label = self.gt_labels_map[img_name]

        if self.transform:
            img = self.transform(img)

        return img, label, img_name

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

existing_val_images = []
existing_gt_labels = []
for img_name in val_images_raw:
    if os.path.exists(os.path.join(val_dir, img_name)):
        existing_val_images.append(img_name)
        existing_gt_labels.append(gt_labels_map[img_name])
val_images = existing_val_images
gt_labels = existing_gt_labels

imagenet_dataset = ImageNetValDataset(val_dir, val_images, gt_labels_map, preprocess)
val_loader = DataLoader(imagenet_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = models.resnet50(pretrained=True)
model.eval()
model.to(device)

def calculate_least_confidence(probabilities_batch):
    most_confident_probability = torch.max(probabilities_batch, dim=1).values
    least_confidence = 1 - most_confident_probability
    return least_confidence

def calculate_margin_confidence(probabilities_batch):
    if probabilities_batch.shape[1] < 2:
        raise ValueError("Probabilities tensor must have at least two classes for margin confidence.")
    
    top_two_probabilities = torch.topk(probabilities_batch, 2, dim=1).values
    margin_confidence = top_two_probabilities[:, 0] - top_two_probabilities[:, 1]
    return margin_confidence

def calculate_ratio_confidence(probabilities_batch):
    if probabilities_batch.shape[1] < 2:
        raise ValueError("Probabilities tensor must have at least two classes for ratio confidence.")

    top_two_probabilities = torch.topk(probabilities_batch, 2, dim=1).values
    
    ratio_confidence = torch.where(
        top_two_probabilities[:, 1] != 0,
        top_two_probabilities[:, 0] / top_two_probabilities[:, 1],
        torch.tensor(float('inf')).to(probabilities_batch.device)
    )
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

top5_correct = 0
total_samples = 0
all_preds = []
all_targets = []

all_least_confidences = []
all_margin_confidences = []
all_ratio_confidences = []
all_msps = []
all_doctor_alpha = []
all_doctor_beta = []
all_energies = []
all_maxlogit_uncertainties = []

print("Starting evaluation and uncertainty metrics calculation...")

for images, targets, img_names in tqdm(val_loader, desc="Processing ImageNet Validation"):
    images, targets = images.to(device), targets.to(device)

    with torch.no_grad():
        logits = model(images)
        probabilities = F.softmax(logits, dim=1)

    preds = torch.argmax(probabilities, dim=1).cpu().numpy()
    all_preds.extend(preds)
    all_targets.extend(targets.cpu().numpy())

    top5_probs, top5_preds = torch.topk(logits, k=5, dim=1)
    for i in range(targets.size(0)):
        if targets[i].item() in top5_preds[i].tolist():
            top5_correct += 1
    total_samples += targets.size(0)

    msp_scores = calculate_msp(probabilities)
    least_conf_scores = calculate_least_confidence(probabilities)
    margin_conf_scores = calculate_margin_confidence(probabilities)
    ratio_conf_scores = calculate_ratio_confidence(probabilities)
    
    doctor_alpha_scores = calculate_doctor(probabilities, 'alpha')
    doctor_beta_scores = calculate_doctor(probabilities, 'beta') 
    energy_scores = calculate_energy(logits)
    max_logit_scores = calculate_max_logit(logits)

    all_msps.extend(msp_scores.cpu().numpy())
    all_least_confidences.extend(least_conf_scores.cpu().numpy())
    all_margin_confidences.extend(margin_conf_scores.cpu().numpy())
    all_ratio_confidences.extend(ratio_conf_scores.cpu().numpy())
    all_doctor_alpha.extend(doctor_alpha_scores.cpu().numpy())
    all_doctor_beta.extend(doctor_beta_scores.cpu().numpy())
    all_energies.extend(energy_scores.cpu().numpy())
    all_maxlogit_uncertainties.extend(max_logit_scores.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

all_least_confidences = np.array(all_least_confidences)
all_margin_confidences = np.array(all_margin_confidences)
all_ratio_confidences = np.array(all_ratio_confidences)
all_msps = np.array(all_msps)
all_doctor_alpha = np.array(all_doctor_alpha)
all_doctor_beta = np.array(all_doctor_beta)
all_energies = np.array(all_energies)
all_maxlogit_uncertainties = np.array(all_maxlogit_uncertainties)

all_odin_uncertainties = np.array([]) 



#made a map to make the code more readable since it was really messy originally 
uncertainty_metric_scores = {
    'LC': np.array(all_least_confidences),
    'MC': np.array(all_margin_confidences),
    'RC': np.array(all_ratio_confidences),
    'MSP': np.array(all_msps),
    'Doctor-α': np.array(all_doctor_alpha),
    'Doctor-β': np.array(all_doctor_beta),
    'Energy': np.array(all_energies),
    'MaxLogit': np.array(all_maxlogit_uncertainties),
}

# Ground truth for AUC: 1 if correct prediction, 0 if incorrect
correctness_labels = (all_preds == all_targets).astype(int)

#calculates auc scores 
auc_scores = {}
for metric_name, uncertainty_values in uncertainty_metric_scores.items():
    try:
        auc = roc_auc_score(correctness_labels, -uncertainty_values)  # negate so high uncertainty = more likely incorrect
        auc_scores[metric_name] = auc
        print(f"{metric_name} AUC = {auc:.4f}")
    except ValueError as e:
        print(f"Skipping {metric_name} due to error: {e}")

#sort metrics alphabetically 
metric_order = ['LC', 'MC', 'RC', 'MSP', 'Doctor-α', 'Doctor-β', 'Energy', 'MaxLogit']
auc_values = [auc_scores[m] for m in metric_order if m in auc_scores]

#plot aucs but decided i liked the other aucs better 
plt.figure(figsize=(10, 5))
plt.plot(metric_order, auc_values, marker='o', label='ResNet-50')
plt.title("AUC on Input Validation for Vulnerability Detection")
plt.xlabel("Uncertainty Metric")
plt.ylabel("AUC")
plt.ylim(0.3, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("auc_per_metric_cifar")


top5_accuracy = top5_correct / total_samples
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

#prints everything out 

print(f'\nResults on ImageNet Validation Set:')
print(f' Top 5 Accuracy: {top5_accuracy * 100:.2f}%')
print(f' Accuracy:       {accuracy * 100:.2f}%')
print(f' Precision:      {precision * 100:.2f}%')
print(f' Recall:         {recall * 100:.2f}%')
print(f' F1 Score:       {f1 * 100:.2f}%')

print("\nLeast Confidence Metrics (Uncertainty):")
print(f"  → Average Least Confidence: {np.mean(all_least_confidences):.4f}")
print(f"  → Min Least Confidence: {np.min(all_least_confidences):.4f}")
print(f"  → Max Least Confidence: {np.max(all_least_confidences):.4f}")
print(f"  → Std Dev of Least Confidence: {np.std(all_least_confidences):.4f}")

print("\nMargin Confidence Metrics (Confidence):")
print(f"  → Average Margin Confidence: {np.mean(all_margin_confidences):.4f}")
print(f"  → Min Margin Confidence: {np.min(all_margin_confidences):.4f}")
print(f"  → Max Margin Confidence: {np.max(all_margin_confidences):.4f}")
print(f"  → Std Dev of Margin Confidence: {np.std(all_margin_confidences):.4f}")

print("\nRatio Confidence Metrics (Confidence):")
print(f"  → Average Ratio Confidence: {np.mean(all_ratio_confidences):.4f}")
print(f"  → Min Ratio Confidence: {np.min(all_ratio_confidences):.4f}")
print(f"  → Max Ratio Confidence: {np.max(all_ratio_confidences):.4f}")
print(f"  → Std Dev of Ratio Confidence: {np.std(all_ratio_confidences):.4f}")

print("\nMaximum Softmax Probability (MSP) Metrics (Confidence):")
print(f"  → Average MSP: {np.mean(all_msps):.4f}")
print(f"  → Min MSP: {np.min(all_msps):.4f}")
print(f"  → Max MSP: {np.max(all_msps):.4f}")
print(f"  → Std Dev of MSP: {np.std(all_msps):.4f}")

print("\nDoctor Alpha Metrics (Uncertainty):")
print(f"  → Average Doctor Alpha: {np.mean(all_doctor_alpha):.4f}")
print(f"  → Min Doctor Alpha: {np.min(all_doctor_alpha):.4f}")
print(f"  → Max Doctor Alpha: {np.max(all_doctor_alpha):.4f}")
print(f"  → Std Dev of Doctor Alpha: {np.std(all_doctor_alpha):.4f}")

print("\nDoctor Beta Metrics (Uncertainty):")
print(f"  → Average Doctor Beta: {np.mean(all_doctor_beta):.4f}")
print(f"  → Min Doctor Beta: {np.min(all_doctor_beta):.4f}")
print(f"  → Max Doctor Beta: {np.max(all_doctor_beta):.4f}")
print(f"  → Std Dev of Doctor Beta: {np.std(all_doctor_beta):.4f}")

if all_odin_uncertainties.size > 0:
    print("\nODIN Uncertainty Metrics (Uncertainty):")
    print(f"  → Average ODIN Uncertainty: {np.mean(all_odin_uncertainties):.4f}")
    print(f"  → Min ODIN Uncertainty: {np.min(all_odin_uncertainties):.4f}")
    print(f"  → Max ODIN Uncertainty: {np.max(all_odin_uncertainties):.4f}")
    print(f"  → Std Dev of ODIN Uncertainty: {np.std(all_odin_uncertainties):.4f}")
else:
    print("\nODIN Uncertainty: Not calculated in this run.")

print("\nEnergy Score Metrics (Uncertainty):")
print(f"  → Average Energy Score: {np.mean(all_energies):.4f}")
print(f"  → Min Energy Score: {np.min(all_energies):.4f}")
print(f"  → Max Energy Score: {np.max(all_energies):.4f}")
print(f"  → Std Dev of Energy Score: {np.std(all_energies):.4f}")

print("\nMax Logit Metrics (Confidence):")
print(f"  → Average Max Logit: {np.mean(all_maxlogit_uncertainties):.4f}")
print(f"  → Min Max Logit: {np.min(all_maxlogit_uncertainties):.4f}")
print(f"  → Max Max Logit: {np.max(all_maxlogit_uncertainties):.4f}")
print(f"  → Std Dev of Max Logit: {np.std(all_maxlogit_uncertainties):.4f}")
