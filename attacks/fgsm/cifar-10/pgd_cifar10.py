import tensorflow_datasets as tfds
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def load_cifar10c():
    ds = tfds.load(
        'cifar10',  # using clean CIFAR-10
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

def pgd_attack(model, image, label, epsilon, alpha, num_iter, device):
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()
    perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    for i in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = torch.nn.functional.cross_entropy(output, label)

        model.zero_grad()
        loss.backward()
        
        data_grad = perturbed_image.grad.data
        
        # applys the perturbation 
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        
        # projects the results back 
        eta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
        perturbed_image = torch.clamp(original_image + eta, 0, 1)
        perturbed_image = perturbed_image.detach() # Detach to prevent gradients from accumulating in the next iteration

    return perturbed_image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = load_cifar10c()

    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to(device)

    epsilon = 0.03  # this is the norm bound 
    alpha = 2/255.0    # step size for each iteration (typically smaller than epsilon)
    num_iter = 10      # number of pgd iterations 

    clean_preds, adv_preds, labels = [], [], []

    for img_tf, label in tqdm(tfds.as_numpy(ds)):
        img, lbl = preprocess_tf_to_torch(img_tf, label)
        img = img.unsqueeze(0).to(device)
        lbl_tensor = torch.tensor([lbl]).to(device)

        #clean prediction without any gradients 
        with torch.no_grad():
            output_clean = model(img)
            init_pred = output_clean.max(1, keepdim=True)[1]
        
        #PGD attack
        perturbed_img = pgd_attack(model, img, lbl_tensor, epsilon, alpha, num_iter, device)
        
        #re-classify the perturbed image
        with torch.no_grad():
            output_adv = model(perturbed_img)
            final_pred = output_adv.max(1, keepdim=True)[1]

        clean_preds.append(init_pred.item())
        adv_preds.append(final_pred.item())
        labels.append(lbl)

    #metrics for clean images
    acc_clean = accuracy_score(labels, clean_preds)
    prec_clean = precision_score(labels, clean_preds, average='macro', zero_division=0)
    rec_clean = recall_score(labels, clean_preds, average='macro', zero_division=0)
    f1_clean = f1_score(labels, clean_preds, average='macro', zero_division=0)

    #metrics for adversarial images
    acc_adv = accuracy_score(labels, adv_preds)
    prec_adv = precision_score(labels, adv_preds, average='macro', zero_division=0)
    rec_adv = recall_score(labels, adv_preds, average='macro', zero_division=0)
    f1_adv = f1_score(labels, adv_preds, average='macro', zero_division=0)

    print("\nClean CIFAR-10 Evaluation:")
    print(f"  → Accuracy : {acc_clean:.4f}")
    print(f"  → Precision: {prec_clean:.4f}")
    print(f"  → Recall   : {rec_clean:.4f}")
    print(f"  → F1‑Score : {f1_clean:.4f}")

    print("\nPGD Adversarial CIFAR-10 Evaluation:")
    print(f"  → Accuracy : {acc_adv:.4f}")
    print(f"  → Precision: {prec_adv:.4f}")
    print(f"  → Recall   : {rec_adv:.4f}")
    print(f"  → F1‑Score : {f1_adv:.4f}")

if __name__ == "__main__":
    main()
