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
#list of cifar-10c corruptions 
corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur',
    'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]
corruptions = ['gaussian_noise']
labels = np.load('/home/diversity_project/aaryaa/attacks/Cifar-10/cifar_npy_files/labels.npy')

#load the pretrained model 
model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    'cifar10_resnet56',
    pretrained=True
).eval().to(device)

criterion = nn.CrossEntropyLoss()

min_vals = torch.tensor([(0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)
max_vals = torch.tensor([(1 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)

#execute the reverse fgsm attack which attempts to improve the image classification accuracy by reverting the image to its original 
def reverse_fgsm_attack(image, epsilon, data_grad, min_pixel_val, max_pixel_val):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.max(perturbed_image, min_pixel_val)
    perturbed_image = torch.min(perturbed_image, max_pixel_val)
    return perturbed_image

epsilon = 0.03

all_perturbed_preds = []
all_true_labels = []
all_perturbed_images = []
original_labels_for_reverse = []

#corrupt the images and for each corruption + severity predict the labels using the model and get metrics
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
            all_perturbed_images.append(perturbed_img.cpu().squeeze(0))
            original_labels_for_reverse.append(lbl)

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
for idx, (perturbed_img_cpu, original_lbl) in tqdm(enumerate(zip(all_perturbed_images, original_labels_for_reverse)), total=len(all_perturbed_images), desc="Processing perturbed images for reverse FGSM"):
    img_for_reverse_fgsm = perturbed_img_cpu.unsqueeze(0).to(device).detach().requires_grad_()

    output_on_perturbed = model(img_for_reverse_fgsm)
        
    loss_for_reverse = torch.nn.functional.cross_entropy(output_on_perturbed, torch.tensor([original_lbl]).to(device))
        
    model.zero_grad()
    loss_for_reverse.backward()
    data_grad_reverse = img_for_reverse_fgsm.grad.data

    reverse_perturbed_img = reverse_fgsm_attack(img_for_reverse_fgsm, -epsilon, data_grad_reverse)

    output_reverse_adv = model(reverse_perturbed_img)
    final_pred_reverse = output_reverse_adv.max(1, keepdim=True)[1]
        
    reverse_adv_preds.append(final_pred_reverse.item())

    # Metrics for the reverse FGSM images
acc_reverse_adv = accuracy_score(original_labels_for_reverse, reverse_adv_preds)
prec_reverse_adv = precision_score(original_labels_for_reverse, reverse_adv_preds, average='macro', zero_division=0)
rec_reverse_adv = recall_score(original_labels_for_reverse, reverse_adv_preds, average='macro', zero_division=0)
f1_reverse_adv = f1_score(original_labels_for_reverse, reverse_adv_preds, average='macro', zero_division=0)
print("\nReverse FGSM Adversarial CIFAR-10 Evaluation:")
print(f"  → Accuracy : {acc_reverse_adv:.4f}")
print(f"  → Precision: {prec_reverse_adv:.4f}")
print(f"  → Recall   : {rec_reverse_adv:.4f}")
print(f"  → F1-Score : {f1_reverse_adv:.4f}")


