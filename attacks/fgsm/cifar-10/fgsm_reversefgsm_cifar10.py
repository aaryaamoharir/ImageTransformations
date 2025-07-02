import tensorflow_datasets as tfds
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def load_cifar10c():
    ds = tfds.load(
        'cifar10',
        split='test',
        shuffle_files=False,
        as_supervised=True,
    )
    return ds

def preprocess_tf_to_torch(image, label):
    img = T.ToTensor()(image)
    img = T.Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))(img)
    return img, int(label)

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = load_cifar10c()

    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to(device)

    epsilon = 0.03

    clean_preds, adv_preds, labels = [], [], []
    all_perturbed_images = []
    reverse_adv_preds = []
    original_labels_for_reverse = []

    print("--- Generating initial FGSM adversarial examples ---")
    for img_tf, label in tqdm(tfds.as_numpy(ds), desc="Processing original images"):
        img, lbl = preprocess_tf_to_torch(img_tf, label)
        img = img.unsqueeze(0).to(device)
        img.requires_grad = True

        output = model(img)
        init_pred = output.max(1, keepdim=True)[1]
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([lbl]).to(device))
        
        model.zero_grad()
        loss.backward()
        data_grad = img.grad.data

        perturbed_img = fgsm_attack(img, epsilon, data_grad)
        
        all_perturbed_images.append(perturbed_img.cpu().squeeze(0))
        original_labels_for_reverse.append(lbl)

        output_adv = model(perturbed_img)
        final_pred = output_adv.max(1, keepdim=True)[1]

        clean_preds.append(init_pred.item())
        adv_preds.append(final_pred.item())
        labels.append(lbl)

    # Metrics for the clean images
    acc_clean = accuracy_score(labels, clean_preds)
    prec_clean = precision_score(labels, clean_preds, average='macro', zero_division=0)
    rec_clean = recall_score(labels, clean_preds, average='macro', zero_division=0)
    f1_clean = f1_score(labels, clean_preds, average='macro', zero_division=0)

    # Metrics for the adv images (after initial FGSM)
    acc_adv = accuracy_score(labels, adv_preds)
    prec_adv = precision_score(labels, adv_preds, average='macro', zero_division=0)
    rec_adv = recall_score(labels, adv_preds, average='macro', zero_division=0)
    f1_adv = f1_score(labels, adv_preds, average='macro', zero_division=0)

    print("\nClean CIFAR-10 Evaluation:")
    print(f"  → Accuracy : {acc_clean:.4f}")
    print(f"  → Precision: {prec_clean:.4f}")
    print(f"  → Recall   : {rec_clean:.4f}")
    print(f"  → F1-Score : {f1_clean:.4f}")

    print("\nFGSM Adversarial CIFAR-10 Evaluation (initial attack):")
    print(f"  → Accuracy : {acc_adv:.4f}")
    print(f"  → Precision: {prec_adv:.4f}")
    print(f"  → Recall   : {rec_adv:.4f}")
    print(f"  → F1-Score : {f1_adv:.4f}")

    print("\n--- Applying Reverse FGSM on perturbed images ---")
    for idx, (perturbed_img_cpu, original_lbl) in tqdm(enumerate(zip(all_perturbed_images, original_labels_for_reverse)), total=len(all_perturbed_images), desc="Processing perturbed images for reverse FGSM"):
        img_for_reverse_fgsm = perturbed_img_cpu.unsqueeze(0).to(device).detach().requires_grad_()

        output_on_perturbed = model(img_for_reverse_fgsm)
        
        loss_for_reverse = torch.nn.functional.cross_entropy(output_on_perturbed, torch.tensor([original_lbl]).to(device))
        
        model.zero_grad()
        loss_for_reverse.backward()
        data_grad_reverse = img_for_reverse_fgsm.grad.data

        reverse_perturbed_img = fgsm_attack(img_for_reverse_fgsm, -epsilon, data_grad_reverse)

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

if __name__ == "__main__":
    main()


