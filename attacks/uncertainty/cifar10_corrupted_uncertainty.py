import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader


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

labels = np.load('/home/diversity_project/aaryaa/attacks/Cifar-10/cifar_npy_files/labels.npy')

model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    'cifar10_resnet56',
    pretrained=True
).eval().to(device)

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
    ratio_confidence = top_two_probabilities[0].item() / top_two_probabilities[1].item()
    return ratio_confidence

all_preds = []
all_true_labels = []
all_least_confidences = []
all_margin_confidences = []
all_ratio_confidences = []

print("Starting uncertainty metrics calculation for CIFAR-10-C...")

for corruption in corruptions:
    data = np.load(f'/home/diversity_project/aaryaa/attacks/Cifar-10/cifar_npy_files/{corruption}.npy')
    for severity in range(5):
        start = severity * 10000
        end = (severity + 1) * 10000
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
        imgs_batch = data[start:end]
        lbls_batch = labels[start:end]

        preds_batch = []

        for img_np, lbl_single in tqdm(zip(imgs_batch, lbls_batch), total=len(imgs_batch), desc=f'{corruption} severity {severity+1}'):
            img = transform(img_np).unsqueeze(0).to(device)
            lbl = torch.tensor([lbl_single]).to(device)

            with torch.no_grad():
                outputs = model(img)
                probabilities = torch.softmax(outputs, dim=1).squeeze(0) # Remove batch dim

                pred = probabilities.argmax(dim=0).cpu().item()

                lc = calculate_least_confidence(probabilities)
                mc = calculate_margin_confidence(probabilities)
                rc = calculate_ratio_confidence(probabilities)

            preds_batch.append(pred)
            all_least_confidences.append(lc)
            all_margin_confidences.append(mc)
            all_ratio_confidences.append(rc)

        acc = accuracy_score(lbls_batch, preds_batch)
        prec = precision_score(lbls_batch, preds_batch, average='weighted', zero_division=0)
        rec = recall_score(lbls_batch, preds_batch, average='weighted', zero_division=0)
        f1 = f1_score(lbls_batch, preds_batch, average='weighted', zero_division=0)

        print(f'{corruption} severity {severity+1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}')

        all_preds.extend(preds_batch)
        all_true_labels.extend(lbls_batch)

overall_acc = accuracy_score(all_true_labels, all_preds)
overall_prec = precision_score(all_true_labels, all_preds, average='weighted', zero_division=0)
overall_rec = recall_score(all_true_labels, all_preds, average='weighted', zero_division=0)
overall_f1 = f1_score(all_true_labels, all_preds, average='weighted', zero_division=0)

print(f'\nOverall Metrics (across all corruptions and severities): Accuracy={overall_acc:.4f}, Precision={overall_prec:.4f}, Recall={overall_rec:.4f}, F1={overall_f1:.4f}')

all_least_confidences = np.array(all_least_confidences)
all_margin_confidences = np.array(all_margin_confidences)
all_ratio_confidences = np.array(all_ratio_confidences)

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
