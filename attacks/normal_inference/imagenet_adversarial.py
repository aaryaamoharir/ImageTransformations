import tensorflow_datasets as tfds
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import torchvision.models as models
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#load imagenet_a from the python packages 
ds = tfds.load('imagenet_a', split='test', shuffle_files=False)
class ImageNetADataset(Dataset):
    def __init__(self, tfds_dataset, transform=None):
        self.data = list(tfds.as_numpy(tfds_dataset))  # convert TF dataset to list of dicts
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image = example['image']
        label = example['label']

        if self.transform:
            #converts the numpy into a PIL image 
            from PIL import Image
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label

    #preprocess the image for resNet50
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),               # convert to tensor (C, H, W), float in [0,1]
    transforms.Normalize(                # normalize with set values 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

for example in tfds.as_numpy(ds.take(1)):
    image = example['image']    
    label = example['label']    

#use the pretrained model 
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

dataset = ImageNetADataset(ds, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for img, label in dataloader:
        img = img.to(device)
        logits = model(img)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.array(all_labels)

num_classes = all_probs.shape[1]
all_labels_onehot = np.eye(num_classes)[all_labels]

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
auc = roc_auc_score(all_labels_onehot, all_probs, average='macro', multi_class='ovr')

print(f'Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')
print(f'AUC Score: {auc:.4f}')


