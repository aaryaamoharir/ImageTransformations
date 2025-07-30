import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Constants for CIFAR-10 normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def inference(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Running Inference"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    print(f"  → Refined Accuracy : {accuracy:.4f}")
    print(f"  → Refined Precision: {precision:.4f}")
    print(f"  → Refined Recall   : {recall:.4f}")
    print(f"  → Refined F1-Score : {f1:.4f}")

    return accuracy, precision, recall, f1

def calculate_max_logit(logits):
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    max_logits, _ = torch.max(logits, dim=1)
    return max_logits.cpu().item()

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
    if top_two_probabilities[1].item() == 0:  # Fixed: check second highest, not first
        return 0  # Return 0 instead of inf for better handling
    ratio_confidence = top_two_probabilities[1].item() / top_two_probabilities[0].item()
    return ratio_confidence

def calculate_msp(probabilities):
    return torch.max(probabilities).item()
def calculate_energy(logits, temperature=1.0):
    """Calculate energy-based uncertainty"""
    return -temperature * torch.logsumexp(logits / temperature, dim=-1).cpu().item()

def calculate_odin(model, image, temperature=1000.0, epsilon=0.0014):
    """Calculate ODIN uncertainty score"""
    model.eval()
    
    # Create a fresh tensor that can have requires_grad
    odin_image = image.detach().clone()
    odin_image.requires_grad = True
    
    # Temperature scaling
    outputs = model(odin_image)
    outputs = outputs / temperature
    
    # Get max class and calculate gradient
    max_class = outputs.argmax(dim=1)
    loss = torch.nn.functional.cross_entropy(outputs, max_class)
    
    # Clear gradients first
    if odin_image.grad is not None:
        odin_image.grad.zero_()
    
    loss.backward()
    
    # Input preprocessing (gradient-based perturbation)
    gradient = torch.sign(odin_image.grad.data)
    perturbed_image = odin_image.detach() - epsilon * gradient
    
    # Forward pass with perturbed input
    with torch.no_grad():
        perturbed_outputs = model(perturbed_image)
        perturbed_outputs = perturbed_outputs / temperature
        odin_score = torch.max(torch.softmax(perturbed_outputs, dim=1)).cpu().item()
    
    return odin_score

def calculate_doctor_alpha(logits, alpha=1.0):
    """Calculate DOCTOR Alpha uncertainty"""
    softmax_probs = torch.softmax(logits, dim=-1)
    # Alpha-based uncertainty (entropy-like measure)
    if alpha == 1.0:
        # Standard entropy when alpha = 1
        uncertainty = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-8))
    else:
        # Generalized entropy
        uncertainty = (1 / (1 - alpha)) * torch.log(torch.sum(softmax_probs ** alpha))
    return uncertainty.cpu().item()

def calculate_doctor_beta(logits, beta=2.0):
    """Calculate DOCTOR Beta uncertainty (based on Renyi entropy)"""
    softmax_probs = torch.softmax(logits, dim=-1)
    if beta == 1.0:
        # Standard entropy
        uncertainty = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-8))
    else:
        # Renyi entropy
        uncertainty = (1 / (1 - beta)) * torch.log(torch.sum(softmax_probs ** beta))
    return uncertainty.cpu().item()


def reverse_fgsm_attack(image, epsilon, data_grad, min_vals, max_vals):
    """Apply reverse FGSM attack with proper bounds"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    # Use proper min/max values for normalized data
    perturbed_image = torch.max(perturbed_image, min_vals)
    perturbed_image = torch.min(perturbed_image, max_vals)
    return perturbed_image

def initial_inference(model, dataloader, criterion, epsilon, device):
    model.eval()
    count_correctUncertainty = 0
    count_incorrectUncertainty = 0
    # Calculate proper min/max values for normalized CIFAR-10 data
    min_vals = torch.tensor([(0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)
    max_vals = torch.tensor([(1 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)

    original_labels = []
    original_predictions = []
    refined_images_list = []
    refined_labels_list = []

    print("\n--- Processing images for uncertainty and perturbation ---")
    for i, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Analyzing and Perturbing"):
        image = images.to(device)
        label = labels.to(device)

        # Get original prediction
        with torch.no_grad():
            outputs_original = model(image)
            _, predicted_original = torch.max(outputs_original.data, 1)
            original_labels.extend(labels.cpu().numpy())
            original_predictions.extend(predicted_original.cpu().numpy())

        # Calculate gradients for perturbation
        image.requires_grad = True
        output_for_grad = model(image)
        loss = criterion(output_for_grad, label)

        model.zero_grad()
        loss.backward()
        data_grad = image.grad.data

        # Calculate uncertainty metrics
        logits = output_for_grad.squeeze(0)
        probabilities = torch.softmax(logits, dim=-1)

        max_logit_val = calculate_max_logit(logits)
        margin_confidence_val = calculate_margin_confidence(probabilities)
        ratio_confidence_val = calculate_ratio_confidence(probabilities)
        msp_val = calculate_msp(probabilities)
        energy_val = calculate_energy(logits)
        odin_val = calculate_odin(model, image)
        doctor_alpha_val = calculate_doctor_alpha(logits, alpha=2.0)
        doctor_beta_val = calculate_doctor_beta(logits, beta=2.0)

        # Original code had inverted logic
        #max_logit_val < 5.0):
        if (ratio_confidence_val > 0.010 and margin_confidence < 0.9970 and doctor_alpha_val > 0.003): 
            #ratio_confidence_val > 0.3 or   
            #msp_val < 0.8):  
       #     print("applying perturbation")
            
            # Apply perturbation to uncertain predictions
            if ( label.item() != predicted_original.item()):
                count_correctUncertainty += 1
            else: 
                count_incorrectUncertainty += 1
            perturbed_image = reverse_fgsm_attack(image, epsilon, data_grad, min_vals, max_vals)
            refined_images_list.append(perturbed_image.squeeze(0).detach().cpu())
        else:
            # Keep original image for confident predictions
        #    print("keeping original")
            refined_images_list.append(image.squeeze(0).detach().cpu())
        
        # Store the original label
        refined_labels_list.append(label.item())

    #initial model inference 
    accuracy = accuracy_score(original_labels, original_predictions)
    precision = precision_score(original_labels, original_predictions, average='macro', zero_division=0)
    recall = recall_score(original_labels, original_predictions, average='macro', zero_division=0)
    f1 = f1_score(original_labels, original_predictions, average='macro', zero_division=0)

    print(f"  → Initial Accuracy : {accuracy:.4f}")
    print(f"  → Initial Precision: {precision:.4f}")
    print(f"  → Initial Recall   : {recall:.4f}")
    print(f"  → Initial F1-Score : {f1:.4f}")
    print(f" Correctly classified incorrect predictions {count_correctUncertainty}")
    print(f" Incorrectly classified incorrect predictions {count_incorrectUncertainty}")
    # Create refined dataset
    refined_dataset = torch.utils.data.TensorDataset(torch.stack(refined_images_list), torch.tensor(refined_labels_list))
    refined_dataloader = torch.utils.data.DataLoader(refined_dataset, batch_size=128, shuffle=False, num_workers=2)

    return refined_dataloader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use the same transform as the working code
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # Use batch_size=1 for individual image processing
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    print("Loading pre-trained CIFAR-10 ResNet56 model...")
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to(device)

    criterion = nn.CrossEntropyLoss()
    epsilon = 0.03

    # Process images and apply selective perturbation
    refined_dataloader = initial_inference(model, testloader, criterion, epsilon, device)
    
    # Run inference on the refined dataset
    print("running final please don't break here")
    final_accuracy, final_precision, final_recall, final_f1 = inference(model, refined_dataloader, device)

    print(f"Final Accuracy on Refined Dataset: {final_accuracy:.4f}")
    print(f"Final F1-Score on Refined Dataset: {final_f1:.4f}")

if __name__ == "__main__":
    main()

