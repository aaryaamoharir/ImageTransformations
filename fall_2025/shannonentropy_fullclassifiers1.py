import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models",
    "cifar100_resnet56",
    pretrained=True,
).to(device)

model.eval()
clean_ds = CIFAR100(root="/tmp/cifar100", train=False,
                    download=True, transform=transform)

clean_loader = DataLoader(clean_ds, batch_size=256, shuffle=False)
def extract_outputs(model, x):
    logits = model(x)
    probs = F.softmax(logits, dim=1)

    top2 = torch.topk(probs, 2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    confidence = top2[:, 0]
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

    pred = logits.argmax(1)
    return pred, confidence, margin, entropy

X = []
y = []

with torch.no_grad():
    for imgs, labels in clean_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        pred, conf, margin, ent = extract_outputs(model, imgs)

        correct = (pred == labels).float()

        feats = torch.stack([conf, margin, ent], dim=1)

        X.append(feats.cpu())
        y.append((1 - correct).cpu())  # failure label

X = torch.cat(X)
y = torch.cat(y)

print("Clean training samples:", len(X))

class FailureHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

model_fail = FailureHead().to(device)

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model_fail.parameters(), lr=1e-3)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X, y),
    batch_size=256,
    shuffle=True
)

for epoch in range(10):
    total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        loss = loss_fn(model_fail(xb), yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Epoch {epoch+1}: {total:.4f}")

class FailureHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

model_fail = FailureHead().to(device)

loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model_fail.parameters(), lr=1e-3)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X, y),
    batch_size=256,
    shuffle=True
)

for epoch in range(10):
    total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        loss = loss_fn(model_fail(xb), yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Epoch {epoch+1}: {total:.4f}")

class CIFAR100C:
    def __init__(self, root, corruption, severity, transform):
        self.data = np.load(f"{root}/{corruption}.npy", mmap_mode="r")
        self.labels = np.load(f"{root}/labels.npy", mmap_mode="r")
        self.start = (severity - 1) * 10000
        self.transform = transform

    def __len__(self):
        return 10000

    def __getitem__(self, i):
        idx = self.start + i
        return self.transform(self.data[idx]), int(self.labels[idx])
CORRUPTIONS = [
    "brightness","contrast","defocus_blur","elastic_transform",
    "fog","frost","glass_blur","jpeg_compression",
    "motion_blur","pixelate","snow","zoom_blur",
]

TP=TN=FP=FN=0
threshold = 0.5

with torch.no_grad():
    for corr in CORRUPTIONS:
        for sev in range(1,6):
            CIFAR_C_ROOT = "/home/diversity_project/aaryaa/attacks/Cifar-100/CIFAR-100-C"

            ds = CIFAR100C(CIFAR_C_ROOT, corr, sev, transform)
            dl = DataLoader(ds, batch_size=256)

            for imgs, labels in dl:
                imgs, labels = imgs.to(device), labels.to(device)

                pred, conf, margin, ent = extract_outputs(model, imgs)
                correct = (pred == labels).float()

                feats = torch.stack([conf, margin, ent], dim=1)
                probs = torch.sigmoid(model_fail(feats))
                predicted_fail = (probs > threshold).float()

                TP += ((correct==0)&(predicted_fail==1)).sum().item()
                TN += ((correct==1)&(predicted_fail==0)).sum().item()
                FP += ((correct==1)&(predicted_fail==1)).sum().item()
                FN += ((correct==0)&(predicted_fail==0)).sum().item()

print("\n==== FINAL RESULTS ====")
print("TP (correctly detected failure):", TP)
print("TN (correctly detected success):", TN)
print("FP (false alarm):", FP)
print("FN (missed failure):", FN)

print("False Trust Rate:", FP / (FP + TN + 1e-6))
print("False Reject Rate:", FN / (FN + TP + 1e-6))
