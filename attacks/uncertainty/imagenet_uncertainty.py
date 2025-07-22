import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import numpy as np

# Paths
val_dir = '/home/diversity_project/aaryaa/attacks/imagenet_data'
gt_file = '/home/diversity_project/aaryaa/attacks/imagenet_caffe_2012/val.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ground truth labels
with open(gt_file, 'r') as f:
    gt_labels = [int(line.strip().split()[-1]) for line in f]

# Preprocessing (ResNet ImageNet defaults)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Generate image names (for ImageNet val set)
val_images = [f'ILSVRC2012_val_{i:08d}.JPEG' for i in range(1, len(gt_labels)+1)]

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)
model.eval()
model.to(device)

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

top5_correct = 0
total = 0
all_preds = []
all_targets = []
all_least_confidences = []
all_margin_confidences = []
all_ratio_confidences = []

print("Starting uncertainty metrics calculation...")

for idx, img_name in enumerate(val_images):
    img_path = os.path.join(val_dir, img_name)

    if not os.path.exists(img_path):
        print(f"Warning: {img_path} does not exist.")
        continue

    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).squeeze(0)

        pred = torch.argmax(probabilities).item()
        gt = gt_labels[idx]

        lc = calculate_least_confidence(probabilities)
        mc = calculate_margin_confidence(probabilities)
        rc = calculate_ratio_confidence(probabilities)

    all_preds.append(pred)
    all_targets.append(gt)
    all_least_confidences.append(lc)
    all_margin_confidences.append(mc)
    all_ratio_confidences.append(rc)

    top5_probs, top5_preds = torch.topk(output, k=5, dim=1)
    top5_preds = top5_preds.squeeze(0).tolist()
    if gt in top5_preds:
        top5_correct += 1

    total += 1

    if total % 1000 == 0:
        print(f'Processed {total}/{len(val_images)} images...')

top5_accuracy = top5_correct / total
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

print(f'\nResults on ImageNet Validation Set:')
print(f' Top 5 Accuracy: {top5_accuracy * 100:.2f}%')
print(f' Accuracy:       {accuracy * 100:.2f}%')
print(f' Precision:      {precision * 100:.2f}%')
print(f' Recall:         {recall * 100:.2f}%')
print(f' F1 Score:       {f1 * 100:.2f}%')

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
