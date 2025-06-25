import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

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

# FGSM params
epsilon = 0.03  # Small perturbation

top5_correct = 0
total = 0
all_preds = []
all_targets = []

print("Starting FGSM attack evaluation...")

for idx, img_name in enumerate(val_images):
    img_path = os.path.join(val_dir, img_name)
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} does not exist.")
        continue

    # Load and preprocess
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True  # FGSM requires this

    # Forward pass
    output = model(input_tensor)
    gt_label = torch.tensor([gt_labels[idx]]).to(device)

    # Calculate loss & backward to get gradients
    loss = F.cross_entropy(output, gt_label)
    model.zero_grad()
    loss.backward()

    # FGSM perturbation
    data_grad = input_tensor.grad.data
    perturbed_data = input_tensor + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)  # Keep pixel values valid

    # Inference on perturbed image
    output_adv = model(perturbed_data)
    top5_probs, top5_preds = torch.topk(output_adv, k=5, dim=1)
    top5_preds = top5_preds.squeeze(0).tolist()

    # Evaluate
    gt = gt_labels[idx]
    all_preds.append(top5_preds[0])  # For top-1 stats
    all_targets.append(gt)
    if gt in top5_preds:
        top5_correct += 1

    total += 1

    if total % 1000 == 0:
        print(f'Processed {total}/{len(val_images)} images...')

# Metrics
top5_accuracy = top5_correct / total
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

print(f'\nResults after FGSM Attack on ImageNet Validation Set:')
print(f' Top 5 Accuracy: {top5_accuracy * 100:.2f}%')
print(f' Accuracy:      {accuracy * 100:.2f}%')
print(f' Precision:     {precision * 100:.2f}%')
print(f' Recall:        {recall * 100:.2f}%')
print(f' F1 Score:      {f1 * 100:.2f}%')

