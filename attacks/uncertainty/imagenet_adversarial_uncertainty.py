import tensorflow_datasets as tfds
import torchvision.models as models
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ImageNet-A
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
dataloader = DataLoader(dataset, batch_size=16, shuffle=False) # Changed to False for consistent results

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
all_labels = []
all_least_confidences = []
all_margin_confidences = []
all_ratio_confidences = []

top5_correct = 0
total_samples = 0
print("about to go into for loop")
for img, label in dataloader:
    img = img.to(device)
    label = label.to(device)

    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=1)

        # Calculate uncertainty metrics for each image in the batch
        for i in range(probs.size(0)):
            probabilities_single = probs[i]
            lc = calculate_least_confidence(probabilities_single)
            mc = calculate_margin_confidence(probabilities_single)
            rc = calculate_ratio_confidence(probabilities_single)

            all_least_confidences.append(lc)
            all_margin_confidences.append(mc)
            all_ratio_confidences.append(rc)

        top1_preds = torch.argmax(probs, dim=1)
        all_preds.extend(top1_preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        top5_preds = torch.topk(probs, k=5, dim=1).indices
        for i in range(label.size(0)):
            if label[i] in top5_preds[i]:
                top5_correct += 1

        total_samples += label.size(0)

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
