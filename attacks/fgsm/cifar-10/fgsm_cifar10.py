import tensorflow_datasets as tfds
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

#load the cifar dataset 
def load_cifar10c():
    ds = tfds.load(
        'cifar10',  # using clean CIFAR-10
        split='test',
        shuffle_files=False,
        as_supervised=True,
    )
    return ds
#preprocess the images 
def preprocess_tf_to_torch(image, label):
    img = T.ToTensor()(image)
    img = T.Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))(img)
    return img, int(label)

#perform the fgsm attack to decrease image accuracy by adding perturbations 
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)  # Keep pixel values in [0,1]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = load_cifar10c()

    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to(device)

    #used 0.03 as my perturbation factor
    epsilon = 0.03  

    clean_preds, adv_preds, labels = [], [], []

    #store the predicted labels and actual labels per image 
    for img_tf, label in tqdm(tfds.as_numpy(ds)):
        img, lbl = preprocess_tf_to_torch(img_tf, label)
        img = img.unsqueeze(0).to(device)
        img.requires_grad = True

        output = model(img)
        init_pred = output.max(1, keepdim=True)[1]  # predicted label
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([lbl]).to(device))
        
        #compute the gradient 
        model.zero_grad()
        loss.backward()
        data_grad = img.grad.data
        perturbed_img = fgsm_attack(img, epsilon, data_grad)

        # get the results from the image 
        output_adv = model(perturbed_img)
        final_pred = output_adv.max(1, keepdim=True)[1]

        clean_preds.append(init_pred.item())
        adv_preds.append(final_pred.item())
        labels.append(lbl)

    # metrics for the clean images 
    acc_clean = accuracy_score(labels, clean_preds)
    prec_clean = precision_score(labels, clean_preds, average='macro', zero_division=0)
    rec_clean = recall_score(labels, clean_preds, average='macro', zero_division=0)
    f1_clean = f1_score(labels, clean_preds, average='macro', zero_division=0)

    # metrics for the adv images 
    acc_adv = accuracy_score(labels, adv_preds)
    prec_adv = precision_score(labels, adv_preds, average='macro', zero_division=0)
    rec_adv = recall_score(labels, adv_preds, average='macro', zero_division=0)
    f1_adv = f1_score(labels, adv_preds, average='macro', zero_division=0)

    print("\nClean CIFAR-10 Evaluation:")
    print(f"  → Accuracy : {acc_clean:.4f}")
    print(f"  → Precision: {prec_clean:.4f}")
    print(f"  → Recall   : {rec_clean:.4f}")
    print(f"  → F1‑Score : {f1_clean:.4f}")

    print("\nFGSM Adversarial CIFAR-10 Evaluation:")
    print(f"  → Accuracy : {acc_adv:.4f}")
    print(f"  → Precision: {prec_adv:.4f}")
    print(f"  → Recall   : {rec_adv:.4f}")
    print(f"  → F1‑Score : {f1_adv:.4f}")

if __name__ == "__main__":
    main()

