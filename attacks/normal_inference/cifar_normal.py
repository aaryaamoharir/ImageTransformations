import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

def preprocess_tf_to_torch(image, label):
    img = T.ToTensor()(image)
    img = T.Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))(img)
    return img, int(label)

def main():
    BATCH_SIZE = 64
    
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_vgg19_bn',
        pretrained=True
    ).eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    all_preds, all_labels = [], []

    for imgs_batch, labels_batch in tqdm(test_loader, desc="Evaluating CIFAR-10"):
        imgs_batch = imgs_batch.to(next(model.parameters()).device)
        labels_batch = labels_batch.to(next(model.parameters()).device)

        with torch.no_grad():
            logits = model(imgs_batch)
            preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("CIFAR‑10 Evaluation with VGG19_BN:")
    print(f" → Accuracy : {acc:.4f}")
    print(f" → Precision: {prec:.4f}")
    print(f" → Recall   : {rec:.4f}")
    print(f" → F1‑Score : {f1:.4f}")

if __name__ == "__main__":
    main()
