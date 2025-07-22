import tensorflow_datasets as tfds
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def load_cifar10c():
    ds = tfds.load(
        'cifar10',
        split='test',
        shuffle_files=False,
        as_supervised=True,
    )
    return ds

def preprocess_tf_to_torch(image, label):
    img = T.ToTensor()(image)
    img = T.Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))(img)
    return img, int(label)

def calculate_least_confidence(probabilities):
    if probabilities.numel() == 0:
        raise ValueError("Probabilities tensor cannot be empty.")
    most_confident_probability = torch.max(probabilities).item()
    least_confidence = 1 - most_confident_probability
    return least_confidence

def calculate_margin_confidence(probabilities):
    if probabilities.numel() < 2:
        raise ValueError("Probabilities tensor must have at least two elements for margin confidence.")
    top_two_probabilities = torch.topk(probabilities, 2).values
    margin_confidence = top_two_probabilities[0].item() - top_two_probabilities[1].item()
    return margin_confidence

def calculate_ratio_confidence(probabilities):
    if probabilities.numel() < 2:
        raise ValueError("Probabilities tensor must have at least two elements for ratio confidence.")
    top_two_probabilities = torch.topk(probabilities, 2).values
    if top_two_probabilities[1].item() == 0:
        return float('inf')
    ratio_confidence = top_two_probabilities[1].item() / top_two_probabilities[0].item()
    return ratio_confidence
def calculate_msp(probabilities):
    return torch.max(probabilities).item()
def calculate_doctor(probabilities, type):
    g_hat = torch.sum(probabilities**2).item()
    pred_error_prob = 1.0 - torch.max(probabilities).item()
    if g_hat == 0: # Avoid division by zero, though unlikely with softmax
        return float('inf')
    if (1 - pred_error_prob) == 0:
        return float ('inf')
    if (type == 'alpha'):
        return (1.0 - g_hat) / g_hat
    else: 
        return pred_error_prob / (1.0 - pred_error_prob)
def main():
    ds = load_cifar10c()
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    all_preds, all_labels = [], []
    all_least_confidences = []
    all_margin_confidences = []
    all_ratio_confidences = []
    all_msps = []
    all_doctor_alpha = []
    all_doctor_beta = []

    for img_tf, label in tfds.as_numpy(ds):
        img, lbl = preprocess_tf_to_torch(img_tf, label)
        img = img.unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            logits = model(img)
            probabilities = torch.softmax(logits, dim=1)
            pred = probabilities.argmax(dim=1).cpu().item()
            lc = calculate_least_confidence(probabilities.squeeze(0))
            mc = calculate_margin_confidence(probabilities.squeeze(0))
            rc = calculate_ratio_confidence(probabilities.squeeze(0))
            msp = calculate_msp(probabilities.squeeze(0))
            alpha = calculate_doctor(probabilities.squeeze(0), 'alpha')
            beta = calculate_doctor(probabilities.squeeze(0), 'beta')
            print(f"The least confidence is {lc}, the margin confidence is {mc}, the ratio confidence is {rc}, the msp is {msp}, the doctor alpha is {alpha}, the doctor beta is {beta}")

        all_preds.append(pred)
        all_labels.append(int(lbl))
        all_least_confidences.append(lc)
        all_margin_confidences.append(mc)
        all_ratio_confidences.append(rc)
        all_msps.append(msp)
        all_doctor_alpha.append(alpha)
        all_doctor_beta.append(beta)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print("CIFAR‑10‑C Evaluation with ResNet56:")
    print(f"  → Accuracy : {acc:.4f}")
    print(f"  → Precision: {prec:.4f}")
    print(f"  → Recall   : {rec:.4f}")
    print(f"  → F1‑Score : {f1:.4f}")

    all_least_confidences = np.array(all_least_confidences)
    all_margin_confidences = np.array(all_margin_confidences)
    all_ratio_confidences = np.array(all_ratio_confidences)
    all_msps = np.array(all_msps)

    print("\nLeast Confidence Metrics:")
    print(f"  → Average Least Confidence: {np.mean(all_least_confidences):.4f}")
    print(f"  → Min Least Confidence: {np.min(all_least_confidences):.4f}")
    print(f"  → Max Least Confidence: {np.max(all_least_confidences):.4f}")
    print(f"  → Std Dev of Least Confidence: {np.std(all_least_confidences):.4f}")

    print("\nMargin Confidence Metrics:")
    print(f"  → Average Margin Confidence: {np.mean(all_margin_confidences):.4f}")
    print(f"  → Min Margin Confidence: {np.min(all_margin_confidences):.4f}")
    print(f"  → Max Margin Confidence: {np.max(all_margin_confidences):.4f}")
    print(f"  → Std Dev of Margin Confidence: {np.std(all_margin_confidences):.4f}")

    print("\nRatio Confidence Metrics:")
    print(f"  → Average Ratio Confidence: {np.mean(all_ratio_confidences):.4f}")
    print(f"  → Min Ratio Confidence: {np.min(all_ratio_confidences):.4f}")
    print(f"  → Max Ratio Confidence: {np.max(all_ratio_confidences):.4f}")
    print(f"  → Std Dev of Ratio Confidence: {np.std(all_ratio_confidences):.4f}")

    print(f" the average msp: {np.mean(all_msps):.4f}")
    print(f" the average doctor_alpha: {np.mean(all_doctor_alpha):.4f}")
    print(f" the average msp: {np.mean(all_doctor_beta):.4f}")

if __name__ == "__main__":
    main()
