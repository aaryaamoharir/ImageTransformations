import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Normalize CIFAR-10
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))
])

corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur',
    'zoom_blur', 'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

labels = np.load('/home/diversity_project/aaryaa/attacks/Cifar-10/cifar_npy_files/labels.npy')

# Load model
model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    'cifar10_vgg19_bn',
    pretrained=True
).eval().to('cuda' if torch.cuda.is_available() else 'cpu')

device = next(model.parameters()).device
all_accs = []
all_preds = []
all_labels = []

batch_size = 64

for corruption in corruptions:
    data = np.load(f'/home/diversity_project/aaryaa/attacks/Cifar-10/cifar_npy_files/{corruption}.npy')
    corruption_accs = []

    for severity in range(5):
        start = severity * 10000
        end = (severity + 1) * 10000
        imgs = data[start:end]
        lbls = labels[start:end]

        # Apply transform to all images
        imgs_tensor = torch.stack([transform(img) for img in imgs])
        lbls_tensor = torch.tensor(lbls, dtype=torch.long)

        dataset = TensorDataset(imgs_tensor, lbls_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds = []
        true_labels = []

        for batch_imgs, batch_lbls in tqdm(loader, desc=f'{corruption} severity {severity+1}'):
            batch_imgs = batch_imgs.to(device)
            with torch.no_grad():
                logits = model(batch_imgs)
                batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)
            true_labels.extend(batch_lbls.numpy())

        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, average='weighted', zero_division=0)
        rec = recall_score(true_labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)

        print(f'{corruption} severity {severity+1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}')
        corruption_accs.append(acc)

        all_preds.extend(preds)
        all_labels.extend(true_labels)

    mean_acc = np.mean(corruption_accs)
    print(f'{corruption}: Mean Accuracy over 5 severities = {mean_acc:.4f}')
    all_accs.append(mean_acc)

# Final overall metrics
overall_acc = accuracy_score(all_labels, all_preds)
overall_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
overall_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
overall_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f'\nOverall CIFAR-10-C Metrics:')
print(f'Accuracy = {overall_acc:.4f}')
print(f'Precision = {overall_prec:.4f}')
print(f'Recall = {overall_rec:.4f}')
print(f'F1 Score = {overall_f1:.4f}')

print(f'\nMean Accuracy over all corruptions (for comparison): {np.mean(all_accs):.4f}')
