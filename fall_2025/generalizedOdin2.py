import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn import metrics

# ------------------ 0. Setup ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained CIFAR-10 ResNet20
model = torch.hub.load('chenyaofo/pytorch-cifar-models',
                       'cifar10_resnet20',
                       pretrained=True).to(device)
model.eval()
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=128, shuffle=False)

# ------------------ 1. Scoring Function (Eq. 1) ------------------
def odin_score(x, model, T=1000):
    """Return max softmax probability with temperature scaling."""
    logits = model(x)
    scaled = logits / T
    probs = F.softmax(scaled, dim=1)
    return probs.max(dim=1).values

# ------------------ 2. ε* Search (Eq. 10) ------------------
eps_candidates = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]

def search_best_epsilon(model, dataloader):
    best_eps, best_sum = None, -1
    for eps in eps_candidates:
        total_score = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device).requires_grad_(True)
            scores = odin_score(imgs, model)
            grads = torch.autograd.grad(scores.sum(), imgs)[0]
            perturbed = imgs - eps * torch.sign(-grads)
            with torch.no_grad():
                new_scores = odin_score(perturbed, model)
            total_score += new_scores.sum().item()
        print(f"ε={eps:.4f}, ΣS(x̂)={total_score:.2f}")
        if total_score > best_sum:
            best_sum, best_eps = total_score, eps
    print(f"→ Selected ε* = {best_eps:.4f}")
    return best_eps

eps_star = search_best_epsilon(model, trainloader)
T = 1000  # fixed per paper

# ------------------ 3. Evaluate Uncertainties on CIFAR-10 ------------------
# ------------------ 3. Evaluate Uncertainties on CIFAR-10 ------------------
all_uncertainties = []
all_correct = []

for imgs, labels in testloader:
    imgs, labels = imgs.to(device), labels.to(device)

    # Enable gradient tracking for perturbation
    imgs.requires_grad_(True)

    # Step 1: compute score and gradient wrt input
    s = odin_score(imgs, model, T=T)
    grads = torch.autograd.grad(s.sum(), imgs, retain_graph=False)[0]

    # Step 2: apply Generalized ODIN perturbation
    x_hat = imgs - eps_star * torch.sign(-grads)

    # Step 3: evaluate model on perturbed input (no grad needed now)
    with torch.no_grad():
        logits = model(x_hat)
        probs = F.softmax(logits / T, dim=1)
        max_probs, preds = probs.max(dim=1)
        uncertainty = 1 - max_probs
        correct = (preds == labels).float()

    all_uncertainties.extend(uncertainty.cpu().numpy())
    all_correct.extend(correct.cpu().numpy())

# ------------------ 4. Threshold Sweep ------------------
print("\n===== Uncertainty Threshold Sweep (Generalized ODIN) =====")
print(f"{'Threshold':>10} | {'Correct>thr':>12} | {'Incorrect>thr':>15}")
print("-" * 45)
# Convert to clean NumPy arrays of floats
all_uncertainties = np.array(all_uncertainties, dtype=float).flatten()
all_correct = np.array(all_correct, dtype=float).flatten()

for thr in np.arange(0.0, 5.1, 0.10):
    mask = all_uncertainties > thr
    correct_above = np.sum(all_correct[mask])
    incorrect_above = np.sum(1 - all_correct[mask])
    print(f"{thr:10.2f} | {int(correct_above):12d} | {int(incorrect_above):15d}")

# ------------------ 5. AUROC & TNR@TPR95 (Paper Metrics) ------------------
# Simulated OoD (Gaussian noise) to compute AUROC / TNR@TPR95
ood_data = torch.randn(len(testset), 3, 32, 32)
ood_loader = torch.utils.data.DataLoader([(ood_data[i], 0) for i in range(len(testset))],
                                         batch_size=128, shuffle=False)

def collect_scores(model, dataloader, eps, T):
    scores = []
    for imgs, _ in dataloader:
        imgs = imgs.to(device).requires_grad_(True)
        s = odin_score(imgs, model, T=T)
        grads = torch.autograd.grad(s.sum(), imgs)[0]
        x_hat = imgs - eps * torch.sign(-grads)
        with torch.no_grad():
            s_hat = odin_score(x_hat, model, T=T)
        scores.extend(s_hat.cpu().numpy())
    return np.array(scores)

id_scores = collect_scores(model, testloader, eps_star, T)
ood_scores = collect_scores(model, ood_loader, eps_star, T)
id_labels = np.ones_like(id_scores)
ood_labels = np.zeros_like(ood_scores)

all_scores = np.concatenate([id_scores, ood_scores])
all_labels = np.concatenate([id_labels, ood_labels])

auroc = metrics.roc_auc_score(all_labels, all_scores)
fpr, tpr, _ = metrics.roc_curve(all_labels, all_scores)
tnr_at_tpr95 = 1 - fpr[np.argmax(tpr >= 0.95)]

print("\n===== Generalized ODIN Paper Metrics =====")
print(f"AUROC: {auroc*100:.2f}%")
print(f"TNR@TPR95: {tnr_at_tpr95*100:.2f}%")

