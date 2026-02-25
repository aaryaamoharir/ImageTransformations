import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps, ImageEnhance
import random

# -----------------------------
# CIFAR-10 normalization
# -----------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def normalize_tensor(tensor):
    mean = torch.tensor(CIFAR10_MEAN).view(3,1,1)
    std = torch.tensor(CIFAR10_STD).view(3,1,1)
    return (tensor - mean) / std

def denormalize_tensor(tensor):
    mean = torch.tensor(CIFAR10_MEAN).view(3,1,1)
    std = torch.tensor(CIFAR10_STD).view(3,1,1)
    return tensor * std + mean

# -----------------------------
# Augmentation operations
# -----------------------------
def rotate(img, severity): return img.rotate(severity * random.choice([-1,1]))
def posterize(img, severity): return ImageOps.posterize(img, int(severity))
def shear_x(img, severity): return img.transform(img.size, Image.AFFINE, (1, severity*0.3, 0, 0, 1, 0))
def shear_y(img, severity): return img.transform(img.size, Image.AFFINE, (1,0,0,severity*0.3,1,0))
def translate_x(img, severity): return img.transform(img.size, Image.AFFINE, (1,0,severity*2,0,1,0))
def translate_y(img, severity): return img.transform(img.size, Image.AFFINE, (1,0,0,0,1,severity*2))
def equalize(img, _): return ImageOps.equalize(img)
def solarize(img, severity): return ImageOps.solarize(img, int(severity*20))

AUG_OPS = [rotate, posterize, shear_x, shear_y, translate_x, translate_y, equalize, solarize]
ALPHA = 1.0  # Dirichlet parameter for AugMix

# -----------------------------
# AugMix function
# -----------------------------
def augmix(image_tensor, severity=3, width=3, depth=-1):
    ws = np.random.dirichlet([ALPHA]*width)
    m = np.random.beta(ALPHA, ALPHA)

    mix = torch.zeros_like(image_tensor)

    for i in range(width):
        image_aug = image_tensor.clone()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = random.choice(AUG_OPS)
            pil = TF.to_pil_image(image_aug)
            pil = op(pil, severity)
            image_aug = TF.to_tensor(pil).to(image_tensor.device)
        mix += ws[i] * image_aug

    mixed = (1 - m) * image_tensor + m * mix
    return mixed

# -----------------------------
# Evaluation functions
# -----------------------------
def get_accuracy(model, loader, device, use_augmix=False):
    model.eval()
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        if use_augmix:
            # Apply AugMix per image
            images = torch.stack([augmix(img) for img in images]).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total

# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CIFAR-10 test set
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # Load pretrained CIFAR-10 ResNet56
    print("Loading pretrained ResNet56...")
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet56', pretrained=True)
    model.to(device).eval()

    # Evaluate on clean images
    clean_acc = get_accuracy(model, testloader, device, use_augmix=False)
    print(f"\nClean CIFAR-100 Accuracy: {clean_acc:.2f}%")

    # Evaluate on AugMix-processed images
    augmix_acc = get_accuracy(model, testloader, device, use_augmix=True)
    print(f"AugMix CIFAR-100 Accuracy: {augmix_acc:.2f}%")

if __name__ == "__main__":
    main()
