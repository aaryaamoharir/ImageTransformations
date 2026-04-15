import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

# -------------------- CONFIG --------------------
CIFARC_ROOT           = "/home/diversity_project/aaryaa/attacks/Cifar-100/CIFAR-100-C"
IMAGES_PER_CORRUPTION = 10000
BATCH_SIZE            = 100
SEED                  = 42

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "glass_blur", "jpeg_compression",
    "motion_blur", "pixelate", "snow", "zoom_blur",
]

# CIFAR-100 Fine-to-Coarse (Superclass) Mapping
COARSE_MAP = {
    0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3,  
    10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
    20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15,
    30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5,
    40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
    50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17,
    60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19,
    70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13,
    80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19,
    90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13
}

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- LOAD MODELS --------------------
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True).to(device)
model.eval()

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = {}
        self.model.conv1.register_forward_hook(self.save("conv1"))
        self.model.layer1.register_forward_hook(self.save("layer1"))
        self.model.layer2.register_forward_hook(self.save("layer2"))
        self.model.layer3.register_forward_hook(self.save("layer3"))

    def save(self, name):
        def hook(module, inp, out):
            pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)
            self.features[name] = pooled.detach()
        return hook

    def forward(self, x):
        out = self.model(x)
        feats = torch.cat([self.features[n] for n in ["conv1", "layer1", "layer2", "layer3"]], dim=1)
        return out, feats

extractor = ResNetFeatureExtractor(model).to(device)

class ReliabilityClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Linear(512, 1))
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze()

# -------------------- SYMBOLIC LOGIC --------------------
def get_symbolic_reliability(logits, k=3):
    """Checks if top-k labels belong to the same coarse superclass."""
    _, top_indices = torch.topk(logits, k, dim=1)
    top_indices = top_indices.cpu().numpy()
    
    symbolic_scores = []
    for row in top_indices:
        coarse_labels = [COARSE_MAP[idx] for idx in row]
        # Logic: If more than 1 superclass is represented, the model is 'confused'
        is_consistent = 1.0 if len(set(coarse_labels)) == 1 else 0.0
        symbolic_scores.append(is_consistent)
    
    return torch.tensor(symbolic_scores).to(device)

# -------------------- DATA LOADER --------------------
class CIFAR100CSubset(Dataset):
    def __init__(self, root, corruption, severity, transform, n=100):
        self.images = np.load(f"{root}/{corruption}.npy")[(severity-1)*10000 : (severity-1)*10000 + n]
        self.labels = np.load(f"{root}/labels.npy")[(severity-1)*10000 : (severity-1)*10000 + n]
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return self.transform(self.images[idx]), int(self.labels[idx])

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

# -------------------- TRAIN META-CLASSIFIER --------------------
# (Running subset collection for training)
print("Collecting features for full training...")
X_train_feats, y_train_labels = [], []
with torch.no_grad():
    # Train on the first 5 corruptions to get a diverse baseline of errors
    for corr in CORRUPTIONS[:5]: 
        # Using severity 3 as a 'middle ground' for training
        ds = CIFAR100CSubset(CIFARC_ROOT, corr, 3, transform, n=10000) 
        train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        
        for imgs, labels in train_loader:
            preds, feats = extractor(imgs.to(device))
            X_train_feats.append(feats.cpu())
            y_train_labels.append((preds.argmax(1) == labels.to(device)).float().cpu())

X_train_feats = torch.cat(X_train_feats)
y_train_labels = torch.cat(y_train_labels)

# Increase epochs slightly since the dataset is now much larger
print(f"Training on {len(X_train_feats)} samples...")
rel_clf = ReliabilityClassifier(X_train_feats.shape[1]).to(device)
optimizer = torch.optim.Adam(rel_clf.parameters(), lr=1e-3)
for _ in range(10):
    for x, y in DataLoader(TensorDataset(X_train_feats, y_train_labels), batch_size=64, shuffle=True):
        p = rel_clf(x.to(device))
        loss = F.binary_cross_entropy(p, y.to(device))
        optimizer.zero_grad(); loss.backward(); optimizer.step()

# -------------------- HYBRID EVALUATION --------------------
print("\nEvaluating Hybrid Neuro-Symbolic Reliability...")
TP, TN, FP, FN = 0, 0, 0, 0

with torch.no_grad():
    for corr in CORRUPTIONS:
        for sev in range(1, 6):
            ds = CIFAR100CSubset(CIFARC_ROOT, corr, sev, transform, n=IMAGES_PER_CORRUPTION)
            for imgs, labels in DataLoader(ds, batch_size=BATCH_SIZE):
                imgs, labels = imgs.to(device), labels.to(device)
                preds, feats = extractor(imgs)
                
                # 1. Neural Component (Statistical Confidence)
                neural_rel = (rel_clf(feats) > 0.5).float()
                
                # 2. Symbolic Component (Logical Consistency)
                symbolic_rel = get_symbolic_reliability(preds, k=3)
                
                # 3. Hybrid Gate (Neuro + Symbolic)
                is_reliable = (neural_rel * symbolic_rel) 
                
                is_correct = (preds.argmax(1) == labels).float()

                TP += ((is_correct == 1) & (is_reliable == 1)).sum().item()
                TN += ((is_correct == 0) & (is_reliable == 0)).sum().item()
                FP += ((is_correct == 0) & (is_reliable == 1)).sum().item()
                FN += ((is_correct == 1) & (is_reliable == 0)).sum().item()

print(f"\n==== Results ====\nCorrect & Reliable (TP): {TP}\nIncorrect & Unreliable (TN): {TN}")
print(f"Incorrect but Reliable (FP): {FP}  <-- Symbolic AI helps lower this!")
print(f"Correct but Unreliable (FN): {FN}")
