import tensorflow_datasets as tfds
import torchvision.models as models
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load the imagenet-a dataset using the library 
ds = tfds.load('imagenet_a', split='test', shuffle_files=False)
print("loaded imagenet-a")

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

#tranform the images so that they can be fed into the pretrained model 
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

mean_vals = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std_vals = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

min_pixel_vals = ((0 - mean_vals) / std_vals).to(device)
max_pixel_vals = ((1 - mean_vals) / std_vals).to(device)

def pgd_attack(model, image, label, epsilon, alpha, num_iter, min_val_norm, max_val_norm):
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()
    
    random_noise = torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
    perturbed_image = original_image + random_noise
    
    perturbed_image = torch.max(perturbed_image, min_val_norm)
    perturbed_image = torch.min(perturbed_image, max_val_norm)

    for i in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = F.cross_entropy(output, label)

        model.zero_grad()
        loss.backward()
        
        data_grad = perturbed_image.grad.data
        
        # PGD step
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        
        eta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
        perturbed_image = original_image + eta
        
        perturbed_image = torch.max(perturbed_image, min_val_norm)
        perturbed_image = torch.min(perturbed_image, max_val_norm)
        
        perturbed_image = perturbed_image.detach()

    return perturbed_image

dataset = ImageNetADataset(ds, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False) # Keep shuffle=False for consistent results

eps = 0.03 #epsilon for image perturbations
alpha = 2/255.0 #step size, usually smaller than epsilon
num_iter = 10 # iterations

all_preds = []
all_labels = []

top5_correct = 0
total_samples = 0
print("about to go into for loop")
for img, label in dataloader:
    img = img.to(device)
    label = label.to(device)

    #carry out the pgd attack 
    img_adv = pgd_attack(model, img, label, eps, alpha, num_iter, min_pixel_vals, max_pixel_vals)

    with torch.no_grad():
        logits_adv = model(img_adv)
        probs = F.softmax(logits_adv, dim=1)

        top1_preds = torch.argmax(probs, dim=1)
        all_preds.extend(top1_preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        top5_preds = torch.topk(probs, k=5, dim=1).indices
        for i in range(label.size(0)):
            if label[i] in top5_preds[i]:
                top5_correct += 1

        total_samples += label.size(0)

#metrics 
print("computing metrics")
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
top5_acc = top5_correct / total_samples

print(f'Top-1 Accuracy: {acc:.4f}')
print(f'Top-5 Accuracy: {top5_acc:.4f}')
print(f'Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')
