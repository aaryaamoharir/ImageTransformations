import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

# ---------------- CONFIG ----------------
CIFARC_ROOT = "/home/diversity_project/aaryaa/attacks/Cifar-100/CIFAR-100-C"
BATCH_SIZE = 256
SEED = 42

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

CORRUPTIONS = [
    "brightness","contrast","defocus_blur","elastic_transform",
    "fog","frost","glass_blur","jpeg_compression",
    "motion_blur","pixelate","snow","zoom_blur",
]

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- LOAD MODEL ----------------
model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models",
    "cifar100_resnet56",
    pretrained=True,
).to(device)
model.eval()

# ---------------- FEATURE EXTRACTOR ----------------
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = {}

        model.conv1.register_forward_hook(self.save("conv1"))
        model.layer1.register_forward_hook(self.save("layer1"))
        model.layer2.register_forward_hook(self.save("layer2"))
        model.layer3.register_forward_hook(self.save("layer3"))

    def save(self, name):
        def hook(m, i, o):
            pooled = F.adaptive_avg_pool2d(o,1).flatten(1)
            self.features[name] = pooled.detach()
        return hook

    def forward(self, x):
        out = self.model(x)
        feats = torch.cat([
            self.features["conv1"],
            self.features["layer1"],
            self.features["layer2"],
            self.features["layer3"],
        ], dim=1)
        return out, feats

extractor = FeatureExtractor(model).to(device)
extractor.eval()

# ---------------- RELIABILITY CLASSIFIER ----------------
class ReliabilityClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

# ---------------- CIFAR100-C FULL DATASET ----------------
class CIFAR100CFull(Dataset):
    def __init__(self, root, corruption, severity, transform):
        self.data = np.load(f"{root}/{corruption}.npy", mmap_mode="r")
        self.labels = np.load(f"{root}/labels.npy", mmap_mode="r")

        self.start = (severity - 1) * 10000
        self.end   = severity * 10000
        self.transform = transform

    def __len__(self): return 10000

    def __getitem__(self, idx):
        real_idx = self.start + idx
        img = self.data[real_idx]
        label = int(self.labels[real_idx])
        return self.transform(img), label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

# ---------------- BUILD META DATASET FROM ALL 600K ----------------
print("Extracting features from ALL CIFAR100-C images...")

X_feats = []
y_correct = []

with torch.no_grad():
    for corr in CORRUPTIONS:
        print("Corruption:", corr)
        for sev in range(1,6):
            ds = CIFAR100CFull(CIFARC_ROOT, corr, sev, transform)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4)

            for imgs, labels in dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds, feats = extractor(imgs)

                pred_labels = preds.argmax(1)
                correct = (pred_labels == labels).float()

                X_feats.append(feats.cpu())
                y_correct.append(correct.cpu())

X_feats = torch.cat(X_feats)
y_correct = torch.cat(y_correct)

print("Total samples:", len(X_feats))

# -------- FEATURE NORMALIZATION --------
mean = X_feats.mean(0)
std = X_feats.std(0) + 1e-6
X_feats = (X_feats - mean) / std

# ---------------- TRAIN RELIABILITY CLASSIFIER ----------------
rel_clf = ReliabilityClassifier(X_feats.shape[1]).to(device)

pos_weight = (y_correct==0).sum()/(y_correct==1).sum()
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
opt = torch.optim.Adam(rel_clf.parameters(), lr=1e-3)

dataset = TensorDataset(X_feats, y_correct)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

print("Training reliability classifier...")
for epoch in range(10):
    total_loss = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = rel_clf(x)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: {total_loss:.4f}")

# ---------------- FINAL EVALUATION ----------------
print("\nEvaluating reliability on FULL dataset...")

TP=TN=FP=FN=0
threshold = 0.6

with torch.no_grad():
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        probs = torch.sigmoid(rel_clf(x))
        reliable = (probs > threshold).float()

        TP += ((y==1)&(reliable==1)).sum().item()
        TN += ((y==0)&(reliable==0)).sum().item()
        FP += ((y==0)&(reliable==1)).sum().item()
        FN += ((y==1)&(reliable==0)).sum().item()

print("\n==== FINAL FULL RESULTS ====")
print("TP(correct & reliable):", TP)
print("TN:(wrong & unreliable)", TN)
print("FP:(wrong but reliable)", FP)
print("FN:(correct but unreliable)", FN)
print("False Trust Rate:", FP/(FP+TN))
print("False Reject Rate:", FN/(FN+TP))
