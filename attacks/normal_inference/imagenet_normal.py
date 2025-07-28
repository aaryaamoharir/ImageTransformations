import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

val_dir = '/home/diversity_project/aaryaa/attacks/imagenet_data'
gt_file = '/home/diversity_project/aaryaa/attacks/imagenet_caffe_2012/val.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

with open(gt_file, 'r') as f:
    gt_info = [line.strip().split() for line in f]
    val_images_raw = [info[0] for info in gt_info]
    gt_labels_map = {info[0]: int(info[1]) for info in gt_info}

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ImageNetValDataset(Dataset):
    def __init__(self, val_dir, val_images, gt_labels_map, transform=None):
        self.val_dir = val_dir
        self.val_images = val_images
        self.gt_labels_map = gt_labels_map
        self.transform = transform

    def __len__(self):
        return len(self.val_images)

    def __getitem__(self, idx):
        img_name = self.val_images[idx]
        img_path = os.path.join(self.val_dir, img_name)
        
        img = Image.open(img_path).convert('RGB')
        label = self.gt_labels_map[img_name]

        if self.transform:
            img = self.transform(img)

        return img, label

existing_val_images = []
for img_name in val_images_raw:
    if os.path.exists(os.path.join(val_dir, img_name)):
        existing_val_images.append(img_name)

imagenet_dataset = ImageNetValDataset(val_dir, existing_val_images, gt_labels_map, preprocess)
val_loader = DataLoader(imagenet_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = models.vgg19_bn(pretrained=True)
model.eval()
model.to(device)

top5_correct = 0
total_samples = 0
all_preds = []
all_targets = []

for images_batch, labels_batch in tqdm(val_loader, desc="Processing ImageNet Validation"):
    images_batch = images_batch.to(device)
    labels_batch = labels_batch.to(device)

    with torch.no_grad():
        output_batch = model(images_batch)
    
    top1_preds_batch = torch.argmax(output_batch, dim=1)
    all_preds.extend(top1_preds_batch.cpu().numpy())
    all_targets.extend(labels_batch.cpu().numpy())

    top5_probs_batch, top5_preds_batch = torch.topk(output_batch, k=5, dim=1)
    
    for i in range(labels_batch.size(0)):
        if labels_batch[i].item() in top5_preds_batch[i].tolist():
            top5_correct += 1
    total_samples += labels_batch.size(0)

top5_accuracy = top5_correct / total_samples
accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

print(f'\nResults on ImageNet Validation Set:')
print(f' Top 5 accuracy: {top5_accuracy*100:.2f}%')
print(f'Accuracy:  {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall:    {recall*100:.2f}%')
print(f'F1 Score:  {f1*100:.2f}%')

