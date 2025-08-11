import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def uncertainty_nll(model, dataloader, uncertainty_thresholds=[0.5, 1.0, 1.5, 2.0, 2.5]):
    """
    Calculates Negative Log-Likelihood uncertainty, accuracy, and AUC score for multiple thresholds.
    """
    model.eval()
    
    results = {threshold: {'uncertain_correct': 0, 'uncertain_incorrect': 0, 
                           'certain_correct': 0, 'certain_incorrect': 0, 
                           'total_uncertain': 0} for threshold in uncertainty_thresholds}
    
    all_labels = []
    all_preds_nll = []
    all_probs_nll = []
    total_samples = 0
    
    nll_loss = nn.NLLLoss(reduction='none') 
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Processing with NLL"):
            outputs = model(inputs)
            
            log_probs = torch.log_softmax(outputs, dim=1)
            probs = torch.exp(log_probs)
            
            nll_values = nll_loss(log_probs, labels)
            
            predicted_classes = torch.argmax(outputs, dim=1)
            correct_predictions = (predicted_classes == labels)
            
            # Collect predictions and labels for overall metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds_nll.extend(predicted_classes.cpu().numpy())
            all_probs_nll.extend(probs.cpu().numpy())

            for threshold in uncertainty_thresholds:
                high_uncertainty_indices = nll_values > threshold
                low_uncertainty_indices = nll_values <= threshold
                
                results[threshold]['uncertain_correct'] += torch.sum(correct_predictions[high_uncertainty_indices]).item()
                results[threshold]['uncertain_incorrect'] += torch.sum(~correct_predictions[high_uncertainty_indices]).item()
                
                results[threshold]['certain_correct'] += torch.sum(correct_predictions[low_uncertainty_indices]).item()
                results[threshold]['certain_incorrect'] += torch.sum(~correct_predictions[low_uncertainty_indices]).item()
                
                results[threshold]['total_uncertain'] += torch.sum(high_uncertainty_indices).item()

            total_samples += len(inputs)

    # Calculate and print overall metrics
    overall_accuracy = accuracy_score(all_labels, all_preds_nll)
    try:
        overall_auc = roc_auc_score(all_labels, all_probs_nll, multi_class='ovr')
    except ValueError:
        overall_auc = "N/A (requires multiple classes with at least one instance each)"

    print(f"\n--- Overall Model Metrics ---")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall AUC Score (One-vs-Rest): {overall_auc:.4f}")

    for threshold, data in results.items():
        print(f"\n--- Results for NLL with threshold = {threshold} ---")
        print(f"Number of inputs with uncertainty above threshold: {data['total_uncertain']}")
        print(f"  - Correctly predicted: {data['uncertain_correct']}")
        print(f"  - Incorrectly predicted: {data['uncertain_incorrect']}")
        print(f"Number of inputs with uncertainty below or at threshold: {total_samples - data['total_uncertain']}")
        print(f"  - Correctly predicted: {data['certain_correct']}")
        print(f"  - Incorrectly predicted: {data['certain_incorrect']}")

if __name__ == '__main__':
    # CIFAR-10 data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    # Load the pre-trained ResNet-56 model from the specified repository using torch.hub
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    
    # Define a list of thresholds for each metric
    mc_dropout_thresholds = [0.8, 1.2, 1.5, 1.8, 2.2]
    nll_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]

    # Run uncertainty analysis for MC Dropout with multiple thresholds
    #uncertainty_mc_dropout(model, testloader, uncertainty_thresholds=mc_dropout_thresholds)
    
    print("\n" + "="*50 + "\n") # Separator for clarity

    # Run uncertainty analysis for NLL with multiple thresholds
    uncertainty_nll(model, testloader, uncertainty_thresholds=nll_thresholds)
