import tensorflow_datasets as tfds
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def load_cifar10c():
    ds = tfds.load(
        #'cifar10_corrupted',
        'cifar10',
        split='test',
        shuffle_files=False,
        as_supervised=True,
    )
    return ds
def preprocess_tf_to_torch(image, label):
    # image: tf.uint8 Tensor [32,32,3]
    #img = image.numpy()
    img = T.ToTensor()(image)               # scales to [0,1]
    img = T.Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))(img)
    return img, int(label)

def calculate_least_confidence(probabilities):
    if probabilities.numel() == 0:
        raise ValueError("Probabilities tensor cannot be empty.")

    # Find the maximum probability across the classes
    most_confident_probability = torch.max(probabilities).item()

    # Calculate least confidence
    least_confidence = 1 - most_confident_probability
    return least_confidence




def main():
    # load dataset from python package 
    ds = load_cifar10c()

    # load pretrained cifar 10 model 
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    all_preds, all_labels = [], []
    all_least_confidences = []


    #get the prediction of each image + the actual label and store it 
    for img_tf, label in tfds.as_numpy(ds):
        img, lbl = preprocess_tf_to_torch(img_tf, label)
        img = img.unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            logits = model(img)
            pred = logits.argmax(dim=1).cpu().item()
            probabilities = torch.softmax(logits, dim=1) # Convert logits to probabilities
            pred = probabilities.argmax(dim=1).cpu().item()
            lc = calculate_least_confidence(probabilities.squeeze(0)) # Remove batch dim before passing to function

        all_preds.append(pred)
        all_labels.append(int(lbl))
        all_least_confidences.append(lc)
    # final metrics 
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

    print("\nLeast Confidence Metrics:")
    print(f"  → Average Least Confidence: {np.mean(all_least_confidences):.4f}")
    print(f"  → Min Least Confidence: {np.min(all_least_confidences):.4f}")
    print(f"  → Max Least Confidence: {np.max(all_least_confidences):.4f}")
    print(f"  → Std Dev of Least Confidence: {np.std(all_least_confidences):.4f}")


if __name__ == "__main__":
    main()
