import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

val_dir = '/home/diversity_project/aaryaa/attacks/imagenet_data'
gt_file = '/home/diversity_project/aaryaa/attacks/imagenet_caffe_2012/val.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(gt_file, 'r') as f:
    gt_labels = [int(line.strip().split()[-1]) for line in f]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_images = [f'ILSVRC2012_val_{i:08d}.JPEG' for i in range(1, len(gt_labels)+1)]

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()
model.to(device)

mean_vals = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std_vals = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

min_pixel_vals = ((0 - mean_vals) / std_vals).to(device)
max_pixel_vals = ((1 - mean_vals) / std_vals).to(device)

def pgd_attack(model, image, label, epsilon, alpha, num_iter, min_val, max_val):
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()
    random_noise = torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
    perturbed_image = original_image + random_noise
    
    perturbed_image = torch.max(perturbed_image, min_val)
    perturbed_image = torch.min(perturbed_image, max_val)

    for i in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = F.cross_entropy(output, label)

        model.zero_grad()
        loss.backward()
        
        data_grad = perturbed_image.grad.data
        
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        
        eta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
        perturbed_image = original_image + eta
        
        perturbed_image = torch.max(perturbed_image, min_val)
        perturbed_image = torch.min(perturbed_image, max_val)
        
        perturbed_image = perturbed_image.detach()

    return perturbed_image

epsilon = 8/255.0
alpha = 2/255.0
num_iter = 10

top5_correct = 0
total = 0
all_preds = []
all_targets = []

print("Starting PGD attack evaluation...")

for idx, img_name in enumerate(val_images):
    img_path = os.path.join(val_dir, img_name)
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} does not exist. Skipping.")
        continue

    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    gt_label = torch.tensor([gt_labels[idx]]).to(device)

    perturbed_data = pgd_attack(model, input_tensor, gt_label, epsilon, alpha, num_iter, min_pixel_vals, max_pixel_vals)

    with torch.no_grad():
        output_adv = model(perturbed_data)
        top5_probs, top5_preds = torch.topk(output_adv, k=5, dim=1)
        top5_preds = top5_preds.squeeze(0).tolist()

    gt = gt_labels[idx]
    all_preds.append(top5_preds[0])
    all_targets.append(gt)
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

print(f'\nResults after PGD Attack on ImageNet Validation Set:')
print(f' Top 5 Accuracy: {top5_accuracy * 100:.2f}%')
print(f' Accuracy:       {accuracy * 100:.2f}%')
print(f' Precision:      {precision * 100:.2f}%')
print(f' Recall:         {recall * 100:.2f}%')
print(f' F1 Score:       {f1 * 100:.2f}%')
