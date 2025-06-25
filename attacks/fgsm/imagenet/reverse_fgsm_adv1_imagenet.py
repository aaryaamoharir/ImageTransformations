import tensorflow_datasets as tfds
import torchvision.models as models
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ds = tfds.load('imagenet_a', split='test', shuffle_files=False)

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

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to(device)
model.eval()

dataset = ImageNetADataset(ds, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  

eps = 0.03  # epsilon parameter 

all_preds = []
all_labels = []

top5_correct = 0
total_samples = 0

for img, label in dataloader:
    img = img.to(device)
    label = label.to(device)

    img.requires_grad = True
    logits = model(img)
    loss = F.cross_entropy(logits, label)
    
    model.zero_grad()
    loss.backward()
    
    grad = img.grad.data
    #actual fgsm attack happens here 
    img_adv = img - eps * grad.sign()
    img_adv = torch.clamp(img_adv, 0, 1)  

    #renormalize to prevent any issues 
    for c in range(3):
        img_adv[:, c, :, :] = (img_adv[:, c, :, :] - transform.transforms[-1].mean[c]) / transform.transforms[-1].std[c]

    with torch.no_grad():
        logits_adv = model(img_adv)
        probs = F.softmax(logits_adv, dim=1)

        #store top-1 predcitions 
        top1_preds = torch.argmax(probs, dim=1)
        all_preds.extend(top1_preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        #store values for top-5 predictions 
        top5_preds = torch.topk(probs, k=5, dim=1).indices
        for i in range(label.size(0)):
            if label[i] in top5_preds[i]:
                top5_correct += 1

        total_samples += label.size(0)

#calcualte and print final metrics 
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
top5_acc = top5_correct / total_samples

print(f'Top-1 Accuracy: {acc:.4f}')
print(f'Top-5 Accuracy: {top5_acc:.4f}')
print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')

