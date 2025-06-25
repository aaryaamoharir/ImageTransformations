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

corruptions = [
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

# These min/max values are for the normalized image space
min_vals = torch.tensor([(0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)
max_vals = torch.tensor([(1 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)

def pgd_attack(model, image, label, epsilon, alpha, num_iter, min_pixel_val, max_pixel_val):
    original_image = image.clone().detach()
    
    # Initialize with a random perturbation within the epsilon ball
    perturbed_image = image.clone().detach()
    random_noise = torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
    perturbed_image = original_image + random_noise
    
    # Clip to valid pixel range after initial random noise (before normalization considerations)
    # This initial clamp is crucial to ensure the random noise doesn't push pixels beyond [0,1] raw range.
    # We then apply normalization-aware clamping in the loop.
    perturbed_image = torch.max(perturbed_image, min_pixel_val)
    perturbed_image = torch.min(perturbed_image, max_pixel_val)

    for i in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = criterion(output, label) # Use your defined criterion

        model.zero_grad()
        loss.backward()
        
        data_grad = perturbed_image.grad.data
        
        # PGD step: Add perturbation
        # If your goal is to reduce accuracy, you typically *add* epsilon * sign(grad)
        # If reverse_fgsm_attack was intended to make it *more* robust, then keep subtraction.
        # For a standard PGD attack aiming to cause misclassification, use addition.
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        
        # Project back to epsilon-ball around the original image
        eta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
        perturbed_image = original_image + eta
        
        # Clamp to the valid normalized pixel range [min_vals, max_vals]
        perturbed_image = torch.max(perturbed_image, min_pixel_val)
        perturbed_image = torch.min(perturbed_image, max_pixel_val)
        
        perturbed_image = perturbed_image.detach() # Detach to prevent gradients from accumulating in the next iteration

    return perturbed_image

# PGD Parameters
epsilon = 0.3 # Total perturbation budget (e.g., 8/255 for L_inf)
alpha = 2/255.0   # Step size for each iteration (should be smaller than epsilon)
num_iter = 10     # Number of PGD iterations

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
            
            # --- PGD Attack instead of FGSM ---
            perturbed_img = pgd_attack(model, img, lbl, epsilon, alpha, num_iter, min_vals, max_vals)
            
            with torch.no_grad():
                perturbed_logits = model(perturbed_img)
                perturbed_pred = perturbed_logits.argmax(dim=1).cpu().item()
            
            perturbed_preds_batch.append(perturbed_pred)
            
        acc = accuracy_score(lbls_batch, perturbed_preds_batch)
        prec = precision_score(lbls_batch, perturbed_preds_batch, average='weighted', zero_division=0)
        rec = recall_score(lbls_batch, perturbed_preds_batch, average='weighted', zero_division=0)
        f1 = f1_score(lbls_batch, perturbed_preds_batch, average='weighted', zero_division=0)

        print(f'{corruption} severity {severity+1} (PGD Perturbed): Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}')
        
        all_perturbed_preds.extend(perturbed_preds_batch)
        all_true_labels.extend(lbls_batch)

overall_acc = accuracy_score(all_true_labels, all_perturbed_preds)
overall_prec = precision_score(all_true_labels, all_perturbed_preds, average='weighted', zero_division=0)
overall_rec = recall_score(all_true_labels, all_perturbed_preds, average='weighted', zero_division=0)
overall_f1 = f1_score(all_true_labels, all_perturbed_preds, average='weighted', zero_division=0)

print(f'\nOverall Metrics (PGD Perturbed across all corruptions and severities): Accuracy={overall_acc:.4f}, Precision={overall_prec:.4f}, Recall={overall_rec:.4f}, F1={overall_f1:.4f}')
