import tensorflow_datasets as tfds
import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

def denormalize_tensor(image_tensor, mean, std):
    mean_t = torch.tensor(mean, device=image_tensor.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=image_tensor.device).view(1, -1, 1, 1)
    denormalized_image = image_tensor * std_t + mean_t
    return denormalized_image

def normalize_tensor(image_tensor, mean, std):
    mean_t = torch.tensor(mean, device=image_tensor.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=image_tensor.device).view(1, -1, 1, 1)
    normalized_image = (image_tensor - mean_t) / std_t
    return normalized_image

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def main():
    ds = tfds.load('imagenet_a', split='test', shuffle_files=False)
    print("Loaded ImageNet-A dataset.")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = model.to(device)
    model.eval()

    dataset = ImageNetADataset(ds, transform=IMAGENET_TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    epsilon = 0.03

    clean_preds_top1, clean_preds_top5_correct = [], 0
    adv_preds_top1, adv_preds_top5_correct = [], 0
    reverse_adv_preds_top1, reverse_adv_preds_top5_correct = [], 0
    all_labels_clean, all_labels_adv, all_labels_reverse_adv = [], [], []

    all_perturbed_images_normalized = []

    print("--- Generating initial FGSM adversarial examples ---")
    for img, label in tqdm(dataloader, desc="Processing original images"):
        img = img.to(device)
        label = label.to(device)

        img.requires_grad = True
        
        logits_clean = model(img)
        loss_clean = F.cross_entropy(logits_clean, label)
        
        model.zero_grad()
        loss_clean.backward()
        data_grad = img.grad.data

        denorm_img = denormalize_tensor(img, IMAGENET_MEAN, IMAGENET_STD)

        unnorm_perturbed_img = fgsm_attack(denorm_img, epsilon, data_grad)
        
        perturbed_img = normalize_tensor(unnorm_perturbed_img, IMAGENET_MEAN, IMAGENET_STD)
        
        # Store individual images and labels from the batch
        # This modification ensures each image is stored as [C, H, W]
        for i in range(perturbed_img.size(0)):
            all_perturbed_images_normalized.append(perturbed_img[i].cpu().detach())
            all_labels_clean.append(label[i].item())
            all_labels_adv.append(label[i].item())
            all_labels_reverse_adv.append(label[i].item())

        top1_preds_clean = torch.argmax(F.softmax(logits_clean, dim=1), dim=1)
        clean_preds_top1.extend(top1_preds_clean.cpu().numpy())
        top5_preds_clean = torch.topk(F.softmax(logits_clean, dim=1), k=5, dim=1).indices
        for i in range(label.size(0)):
            if label[i] in top5_preds_clean[i]:
                clean_preds_top5_correct += 1

        logits_adv = model(perturbed_img)
        top1_preds_adv = torch.argmax(F.softmax(logits_adv, dim=1), dim=1)
        adv_preds_top1.extend(top1_preds_adv.cpu().numpy())
        top5_preds_adv = torch.topk(F.softmax(logits_adv, dim=1), k=5, dim=1).indices
        for i in range(label.size(0)):
            if label[i] in top5_preds_adv[i]:
                adv_preds_top5_correct += 1

    total_samples = len(all_labels_clean)

    acc_clean_top1 = accuracy_score(all_labels_clean, clean_preds_top1)
    prec_clean_top1 = precision_score(all_labels_clean, clean_preds_top1, average='macro', zero_division=0)
    rec_clean_top1 = recall_score(all_labels_clean, clean_preds_top1, average='macro', zero_division=0)
    f1_clean_top1 = f1_score(all_labels_clean, clean_preds_top1, average='macro', zero_division=0)
    acc_clean_top5 = clean_preds_top5_correct / total_samples

    print("\nClean ImageNet-A Evaluation:")
    print(f"  → Top-1 Accuracy : {acc_clean_top1:.4f}")
    print(f"  → Top-5 Accuracy : {acc_clean_top5:.4f}")
    print(f"  → Top-1 Precision: {prec_clean_top1:.4f}")
    print(f"  → Top-1 Recall   : {rec_clean_top1:.4f}")
    print(f"  → Top-1 F1-Score : {f1_clean_top1:.4f}")

    acc_adv_top1 = accuracy_score(all_labels_adv, adv_preds_top1)
    prec_adv_top1 = precision_score(all_labels_adv, adv_preds_top1, average='macro', zero_division=0)
    rec_adv_top1 = recall_score(all_labels_adv, adv_preds_top1, average='macro', zero_division=0)
    f1_adv_top1 = f1_score(all_labels_adv, adv_preds_top1, average='macro', zero_division=0)
    acc_adv_top5 = adv_preds_top5_correct / total_samples

    print("\nFGSM Adversarial ImageNet-A Evaluation (initial attack):")
    print(f"  → Top-1 Accuracy : {acc_adv_top1:.4f}")
    print(f"  → Top-5 Accuracy : {acc_adv_top5:.4f}")
    print(f"  → Top-1 Precision: {prec_adv_top1:.4f}")
    print(f"  → Top-1 Recall   : {rec_adv_top1:.4f}")
    print(f"  → Top-1 F1-Score : {f1_adv_top1:.4f}")

    print("\n--- Applying Reverse FGSM on perturbed images ---")
    for idx, (perturbed_img_normalized_cpu, original_lbl) in tqdm(enumerate(zip(all_perturbed_images_normalized, all_labels_reverse_adv)), total=len(all_perturbed_images_normalized), desc="Processing perturbed images for reverse FGSM"):
        img_for_reverse_fgsm = perturbed_img_normalized_cpu.unsqueeze(0).to(device).requires_grad_()

        logits_on_perturbed = model(img_for_reverse_fgsm)
        
        loss_for_reverse = F.cross_entropy(logits_on_perturbed, torch.tensor([original_lbl]).to(device))
        
        model.zero_grad()
        loss_for_reverse.backward()
        data_grad_reverse = img_for_reverse_fgsm.grad.data

        denorm_img_for_reverse = denormalize_tensor(img_for_reverse_fgsm, IMAGENET_MEAN, IMAGENET_STD)

        unnorm_reverse_perturbed_img = fgsm_attack(denorm_img_for_reverse, -epsilon, data_grad_reverse)

        reverse_perturbed_img = normalize_tensor(unnorm_reverse_perturbed_img, IMAGENET_MEAN, IMAGENET_STD)

        logits_reverse_adv = model(reverse_perturbed_img)
        top1_preds_reverse = torch.argmax(F.softmax(logits_reverse_adv, dim=1), dim=1)
        reverse_adv_preds_top1.extend(top1_preds_reverse.cpu().numpy())
        top5_preds_reverse = torch.topk(F.softmax(logits_reverse_adv, dim=1), k=5, dim=1).indices
        # Ensure label comparison is done correctly for the single item
        if torch.tensor([original_lbl]).item() in top5_preds_reverse.cpu().numpy():
            reverse_adv_preds_top5_correct += 1

    acc_reverse_adv_top1 = accuracy_score(all_labels_reverse_adv, reverse_adv_preds_top1)
    prec_reverse_adv_top1 = precision_score(all_labels_reverse_adv, reverse_adv_preds_top1, average='macro', zero_division=0)
    rec_reverse_adv_top1 = recall_score(all_labels_reverse_adv, reverse_adv_preds_top1, average='macro', zero_division=0)
    f1_reverse_adv_top1 = f1_score(all_labels_reverse_adv, reverse_adv_preds_top1, average='macro', zero_division=0)
    acc_reverse_adv_top5 = reverse_adv_preds_top5_correct / total_samples

    print("\nReverse FGSM Adversarial ImageNet-A Evaluation:")
    print(f"  → Top-1 Accuracy : {acc_reverse_adv_top1:.4f}")
    print(f"  → Top-5 Accuracy : {acc_reverse_adv_top5:.4f}")
    print(f"  → Top-1 Precision: {prec_reverse_adv_top1:.4f}")
    print(f"  → Top-1 Recall   : {rec_reverse_adv_top1:.4f}")
    print(f"  → Top-1 F1-Score : {f1_reverse_adv_top1:.4f}")

if __name__ == "__main__":
    main()

