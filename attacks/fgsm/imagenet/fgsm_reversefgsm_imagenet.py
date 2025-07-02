import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from tqdm import tqdm

val_dir = '/home/diversity_project/aaryaa/attacks/imagenet_data'
gt_file = '/home/diversity_project/aaryaa/attacks/imagenet_caffe_2012/val.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

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
    with open(gt_file, 'r') as f:
        gt_labels_all = [int(line.strip().split()[-1]) for line in f]

    val_images_filenames = [f'ILSVRC2012_val_{i:08d}.JPEG' for i in range(1, len(gt_labels_all)+1)]

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    model.to(device)

    epsilon = 0.03

    clean_preds_top1, clean_preds_top5_correct = [], 0
    adv_preds_top1, adv_preds_top5_correct = [], 0
    reverse_adv_preds_top1, reverse_adv_preds_top5_correct = [], 0
    
    all_labels_clean_eval = [] 
    all_labels_adv_eval = []
    all_labels_reverse_adv_eval = []

    all_perturbed_images_normalized = []

    print("--- Starting initial FGSM attack generation and evaluation ---")

    for idx, img_name in enumerate(tqdm(val_images_filenames, desc="Processing ImageNet images")):
        img_path = os.path.join(val_dir, img_name)
        gt_label_idx = gt_labels_all[idx]

        if not os.path.exists(img_path):
            print(f"Skipping: Image file not found at {img_path}")
            continue

        try:
            img_pil = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
            input_tensor.requires_grad = True
            gt_label_tensor = torch.tensor([gt_label_idx]).to(device)

            with torch.no_grad():
                logits_clean = model(input_tensor)
                top1_preds_clean = torch.argmax(F.softmax(logits_clean, dim=1), dim=1)
                clean_preds_top1.extend(top1_preds_clean.cpu().numpy())
                top5_preds_clean = torch.topk(F.softmax(logits_clean, dim=1), k=5, dim=1).indices
                if gt_label_idx in top5_preds_clean.squeeze(0).tolist():
                    clean_preds_top5_correct += 1
                all_labels_clean_eval.append(gt_label_idx)

            loss_for_attack = F.cross_entropy(logits_clean, gt_label_tensor)
            model.zero_grad()
            loss_for_attack.backward()
            data_grad = input_tensor.grad.data

            denorm_input_tensor = denormalize_tensor(input_tensor, IMAGENET_MEAN, IMAGENET_STD)

            unnorm_perturbed_img = fgsm_attack(denorm_input_tensor, epsilon, data_grad)
            
            perturbed_img_normalized = normalize_tensor(unnorm_perturbed_img, IMAGENET_MEAN, IMAGENET_STD)
            
            all_perturbed_images_normalized.append(perturbed_img_normalized.cpu().squeeze(0).detach())
            all_labels_adv_eval.append(gt_label_idx)

            with torch.no_grad():
                logits_adv = model(perturbed_img_normalized)
                top1_preds_adv = torch.argmax(F.softmax(logits_adv, dim=1), dim=1)
                adv_preds_top1.extend(top1_preds_adv.cpu().numpy())
                top5_preds_adv = torch.topk(F.softmax(logits_adv, dim=1), k=5, dim=1).indices
                if gt_label_idx in top5_preds_adv.squeeze(0).tolist():
                    adv_preds_top5_correct += 1

        except Exception as e:
            print(f"Skipping: Error processing {img_name} - {e}")
            continue

    total_samples = len(all_labels_clean_eval)

    acc_clean_top1 = accuracy_score(all_labels_clean_eval, clean_preds_top1)
    prec_clean_top1 = precision_score(all_labels_clean_eval, clean_preds_top1, average='macro', zero_division=0)
    rec_clean_top1 = recall_score(all_labels_clean_eval, clean_preds_top1, average='macro', zero_division=0)
    f1_clean_top1 = f1_score(all_labels_clean_eval, clean_preds_top1, average='macro', zero_division=0)
    acc_clean_top5 = clean_preds_top5_correct / total_samples

    print("\nClean ImageNet Evaluation:")
    print(f"  → Top-1 Accuracy : {acc_clean_top1:.4f}")
    print(f"  → Top-5 Accuracy : {acc_clean_top5:.4f}")
    print(f"  → Top-1 Precision: {prec_clean_top1:.4f}")
    print(f"  → Top-1 Recall   : {rec_clean_top1:.4f}")
    print(f"  → Top-1 F1-Score : {f1_clean_top1:.4f}")

    # Check if adversarial labels are empty before calculating metrics
    if len(all_labels_adv_eval) == 0:
        print("\nFGSM Adversarial ImageNet Evaluation (initial attack): No adversarial samples processed.")
        acc_adv_top1, prec_adv_top1, rec_adv_top1, f1_adv_top1, acc_adv_top5 = float('nan'), float('nan'), float('nan'), float('nan'), 0.0
    else:
        acc_adv_top1 = accuracy_score(all_labels_adv_eval, adv_preds_top1)
        prec_adv_top1 = precision_score(all_labels_adv_eval, adv_preds_top1, average='macro', zero_division=0)
        rec_adv_top1 = recall_score(all_labels_adv_eval, adv_preds_top1, average='macro', zero_division=0)
        f1_adv_top1 = f1_score(all_labels_adv_eval, adv_preds_top1, average='macro', zero_division=0)
        acc_adv_top5 = adv_preds_top5_correct / len(all_labels_adv_eval) # Use actual number of adv samples

    print("\nFGSM Adversarial ImageNet Evaluation (initial attack):")
    print(f"  → Top-1 Accuracy : {acc_adv_top1:.4f}")
    print(f"  → Top-5 Accuracy : {acc_adv_top5:.4f}")
    print(f"  → Top-1 Precision: {prec_adv_top1:.4f}")
    print(f"  → Top-1 Recall   : {rec_adv_top1:.4f}")
    print(f"  → Top-1 F1-Score : {f1_adv_top1:.4f}")

    print("\n--- Applying Reverse FGSM on previously perturbed images ---")
    # It's crucial that all_perturbed_images_normalized is populated
    if len(all_perturbed_images_normalized) == 0:
        print("No perturbed images available for reverse FGSM attack. Skipping evaluation.")
        acc_reverse_adv_top1, prec_reverse_adv_top1, rec_reverse_adv_top1, f1_reverse_adv_top1, acc_reverse_adv_top5 = float('nan'), float('nan'), float('nan'), float('nan'), 0.0
    else:
        for idx, (perturbed_img_normalized_cpu, original_lbl) in enumerate(tqdm(zip(all_perturbed_images_normalized, all_labels_reverse_adv_eval), total=len(all_perturbed_images_normalized), desc="Processing perturbed images for reverse FGSM")):
            img_for_reverse_fgsm = perturbed_img_normalized_cpu.unsqueeze(0).to(device).requires_grad_()

            logits_on_perturbed = model(img_for_reverse_fgsm)
            
            loss_for_reverse = F.cross_entropy(logits_on_perturbed, torch.tensor([original_lbl]).to(device))
            
            model.zero_grad()
            loss_for_reverse.backward()
            data_grad_reverse = img_for_reverse_fgsm.grad.data

            denorm_img_for_reverse = denormalize_tensor(img_for_reverse_fgsm, IMAGENET_MEAN, IMAGENET_STD)

            unnorm_reverse_perturbed_img = fgsm_attack(denorm_img_for_reverse, -epsilon, data_grad_reverse)

            reverse_perturbed_img_normalized = normalize_tensor(unnorm_reverse_perturbed_img, IMAGENET_MEAN, IMAGENET_STD)

            with torch.no_grad():
                logits_reverse_adv = model(reverse_perturbed_img_normalized)
                top1_preds_reverse = torch.argmax(F.softmax(logits_reverse_adv, dim=1), dim=1)
                reverse_adv_preds_top1.extend(top1_preds_reverse.cpu().numpy())
                top5_preds_reverse = torch.topk(F.softmax(logits_reverse_adv, dim=1), k=5, dim=1).indices
                if original_lbl in top5_preds_reverse.squeeze(0).tolist():
                    reverse_adv_preds_top5_correct += 1
        
        acc_reverse_adv_top1 = accuracy_score(all_labels_reverse_adv_eval, reverse_adv_preds_top1)
        prec_reverse_adv_top1 = precision_score(all_labels_reverse_adv_eval, reverse_adv_preds_top1, average='macro', zero_division=0)
        rec_reverse_adv_top1 = recall_score(all_labels_reverse_adv_eval, reverse_adv_preds_top1, average='macro', zero_division=0)
        f1_reverse_adv_top1 = f1_score(all_labels_reverse_adv_eval, reverse_adv_preds_top1, average='macro', zero_division=0)
        acc_reverse_adv_top5 = reverse_adv_preds_top5_correct / len(all_labels_reverse_adv_eval) # Use actual number of reverse adv samples

    print("\nReverse FGSM Adversarial ImageNet Evaluation:")
    print(f"  → Top-1 Accuracy : {acc_reverse_adv_top1:.4f}")
    print(f"  → Top-5 Accuracy : {acc_reverse_adv_top5:.4f}")
    print(f"  → Top-1 Precision: {prec_reverse_adv_top1:.4f}")
    print(f"  → Top-1 Recall   : {rec_reverse_adv_top1:.4f}")
    print(f"  → Top-1 F1-Score : {f1_reverse_adv_top1:.4f}")

if __name__ == "__main__":
    main()

