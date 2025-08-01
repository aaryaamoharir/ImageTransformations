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
import torch.nn.functional as F

def negative_class_loss(logits, predicted_labels, num_classes):
    """
    Calculates CE loss with a randomly chosen non-predicted class as the target.
    This forces the model to move away from its original prediction.
    """
    # Create a tensor of all possible class indices
    all_classes = torch.arange(num_classes).to(logits.device)
    
    # Get the predicted class for each item in the batch
    predicted_classes = predicted_labels.unsqueeze(1)
    
    # Create a mask to filter out the predicted class
    non_predicted_mask = all_classes.unsqueeze(0) != predicted_classes
    
    # Select a random non-predicted class for each sample
    # This requires a loop for the random choice per sample
    batch_size = logits.size(0)
    target_labels = torch.zeros(batch_size, dtype=torch.long).to(logits.device)
    for i in range(batch_size):
        non_predicted_indices = all_classes[all_classes != predicted_labels[i]]
        target_labels[i] = non_predicted_indices[torch.randint(len(non_predicted_indices), (1,))]

    return F.cross_entropy(logits, target_labels)

def logit_consistency_loss(logits_orig, logits_trans):
    """
    Calculates the Mean Squared Error (MSE) between two sets of logits.
    This encourages the model to produce consistent logits for the original
    and transformed versions of the same image.
    """
    return torch.nn.functional.mse_loss(logits_trans, logits_orig)

# The old logit_margin_loss is removed, as it is no longer used.

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
    if top_two_probabilities[1].item() == 0:
        return 0
    ratio_confidence = top_two_probabilities[1].item() / top_two_probabilities[0].item()
    return ratio_confidence

def calculate_msp(probabilities):
    return torch.max(probabilities).item()

def reverse_fgsm_attack(image, epsilon, data_grad, min_vals, max_vals):
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.max(perturbed_image, min_vals)
    perturbed_image = torch.min(perturbed_image, max_vals)
    return perturbed_image

def initial_inference(model, dataloader, epsilon, device):
    model.eval()
    count_correctUncertainty = 0
    count_incorrectUncertainty = 0
    min_vals = torch.tensor([(0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)
    max_vals = torch.tensor([(1 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)

    original_labels = []
    original_predictions = []
    refined_images_list = []
    refined_labels_list = []
    
    print("\n--- Processing images for uncertainty and perturbation ---")
    for images, labels in tqdm(dataloader, total=len(dataloader), desc="Analyzing and Perturbing"):
        image = images.to(device)
        label = labels.to(device)

        # Get original prediction and logits
        with torch.no_grad():
            outputs_original = model(image)
            _, predicted_original = torch.max(outputs_original.data, 1)
            original_labels.extend(labels.cpu().numpy())
            original_predictions.extend(predicted_original.cpu().numpy())
            
        # Get initial MSP
        with torch.no_grad():
            probabilities = torch.softmax(outputs_original.squeeze(0), dim=-1)
            msp_val = calculate_msp(probabilities)

        if msp_val < 0.8:
            if label.item() != predicted_original.item():
                count_correctUncertainty += 1
            else:
                count_incorrectUncertainty += 1

            # --- START OF THE ITERATIVE FIX ---
            current_image = image.clone().detach()
            count = 0
            while msp_val < 0.8 and count < 150:
                current_image = current_image.detach().requires_grad_(True)
                
                output_for_grad = model(current_image)
                
                # Calculate consistency loss between the original logits and the current logits
                # Detaching outputs_original ensures we only backpropagate through current_image
                #loss = logit_consistency_loss(outputs_original.detach(), output_for_grad)
                #probs = torch.softmax(output_for_grad, dim=1)
                #loss = -torch.sum(probs * torch.log(probs + 1e-8)) / probs.shape[0]  # entropy loss
                #loss = -torch.max(probs, dim=1).values.mean()  # maximize confidence
                #logit = output_for_grad
                #orig_pred = torch.argmax(probabilities, dim=-1)
                #logit_orig = logit[:, orig_pred.item()]
                #logit_alt = logit[:, 1 - orig_pred.item()]
                #loss = (logit_orig - logit_alt).mean()
                
                #top2 = torch.topk(probabilities, 2, dim=-1).indices
                #target_class = top2[1].item()  # second-best guess
                #logit_orig = logit[:, top2[0].item()]
                #logit_alt = logit[:, target_class]
                #loss = (logit_orig - logit_alt).mean()
                loss = negative_class_loss(output_for_grad, predicted_original, 10)

                model.zero_grad()
                loss.backward()
                data_grad = current_image.grad.data
                
                current_image = reverse_fgsm_attack(current_image, epsilon, data_grad, min_vals, max_vals)
                
                with torch.no_grad():
                    outputs_new = model(current_image)
                    probabilities_new = torch.softmax(outputs_new.squeeze(0), dim=-1)
                    msp_val = calculate_msp(probabilities_new)
                
                count += 1
            
            refined_images_list.append(current_image.squeeze(0).detach().cpu())
            refined_labels_list.append(label.item())

        else:
            refined_images_list.append(image.squeeze(0).detach().cpu())
            refined_labels_list.append(label.item())
            # --- END OF THE ITERATIVE FIX ---

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

    refined_dataset = torch.utils.data.TensorDataset(torch.stack(refined_images_list), torch.tensor(refined_labels_list))
    refined_dataloader = torch.utils.data.DataLoader(refined_dataset, batch_size=128, shuffle=False, num_workers=2)

    return refined_dataloader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    print("Loading pre-trained CIFAR-10 ResNet56 model...")
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to(device)

    # The criterion parameter is no longer needed in this setup
    epsilon = 0.03

    # The call to initial_inference no longer includes the criterion parameter
    refined_dataloader = initial_inference(model, testloader, epsilon, device)
    
    print("running final please don't break here")
    final_accuracy, final_precision, final_recall, final_f1 = inference(model, refined_dataloader, device)

    print(f"Final Accuracy on Refined Dataset: {final_accuracy:.4f}")
    print(f"Final F1-Score on Refined Dataset: {final_f1:.4f}")

if __name__ == "__main__":
    main()
