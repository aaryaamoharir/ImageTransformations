import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

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

criterion = nn.CrossEntropyLoss()

min_vals = torch.tensor([(0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)
max_vals = torch.tensor([(1 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)

def reverse_fgsm_attack(image, epsilon, data_grad, min_pixel_val, max_pixel_val):
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.max(perturbed_image, min_pixel_val)
    perturbed_image = torch.min(perturbed_image, max_pixel_val)
    return perturbed_image

epsilon = 0.03

all_perturbed_preds = []
all_true_labels = []

for corruption in corruptions:
    data = np.load(f'/home/diversity_project/aaryaa/attacks/Cifar-10/cifar_npy_files/{corruption}.npy')
    for severity in range(5):
        start = severity * 10000
        end = (severity + 1) * 10000
        imgs_batch = data[start:end]
        lbls_batch = labels[start:end]
        
        perturbed_preds_batch = []
        
        for img_np, lbl_single in tqdm(zip(imgs_batch, lbls_batch), total=len(imgs_batch), desc=f'{corruption} severity {severity+1}'):
            img = transform(img_np).unsqueeze(0).to(device)
            lbl = torch.tensor([lbl_single]).to(device)
            
            img.requires_grad = True
            
            outputs = model(img)
            loss = criterion(outputs, lbl)
            
            model.zero_grad()
            loss.backward()
            
            data_grad = img.grad.data
            
            perturbed_img = reverse_fgsm_attack(img, epsilon, data_grad, min_vals, max_vals)
            
            with torch.no_grad():
                perturbed_logits = model(perturbed_img)
                perturbed_pred = perturbed_logits.argmax(dim=1).cpu().item()
            
            perturbed_preds_batch.append(perturbed_pred)
            
        acc = accuracy_score(lbls_batch, perturbed_preds_batch)
        prec = precision_score(lbls_batch, perturbed_preds_batch, average='weighted', zero_division=0)
        rec = recall_score(lbls_batch, perturbed_preds_batch, average='weighted', zero_division=0)
        f1 = f1_score(lbls_batch, perturbed_preds_batch, average='weighted', zero_division=0)

        print(f'{corruption} severity {severity+1} (Perturbed): Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}')
        
        all_perturbed_preds.extend(perturbed_preds_batch)
        all_true_labels.extend(lbls_batch)

overall_acc = accuracy_score(all_true_labels, all_perturbed_preds)
overall_prec = precision_score(all_true_labels, all_perturbed_preds, average='weighted', zero_division=0)
overall_rec = recall_score(all_true_labels, all_perturbed_preds, average='weighted', zero_division=0)
overall_f1 = f1_score(all_true_labels, all_perturbed_preds, average='weighted', zero_division=0)

print(f'\nOverall Metrics (Perturbed across all corruptions and severities): Accuracy={overall_acc:.4f}, Precision={overall_prec:.4f}, Recall={overall_rec:.4f}, F1={overall_f1:.4f}')
