import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
import pandas as pd
from typing import Dict
import torch.autograd as autograd
from functools import partial
import os

# --- Configuration and Constants ---
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
NUM_CLASSES = 10
NUM_MC_DROPOUT_SAMPLES = 50

# --- Helper Functions (From previous conversations) ---

def logit_margin_loss(logits, labels, margin=1.0):
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1])
    correct_class_logits = torch.sum(logits * one_hot_labels, dim=1)
    incorrect_logits = logits + (-1e9 * one_hot_labels)
    max_incorrect_logits = torch.max(incorrect_logits, dim=1)[0]
    loss = torch.relu(max_incorrect_logits - correct_class_logits + margin)
    return torch.mean(loss)

def calculate_energy(logits):
    return -torch.logsumexp(logits, dim=1).cpu().numpy()

def calculate_mc_dropout_uncertainty(model, images, num_samples, dropout_layer, device):
    model.train()
    softmax_outputs = []
    features_at_layer = []

    def hook_fn(module, input, output):
        features_at_layer.append(output.detach().cpu().numpy())
    
    hook = dropout_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(images)
            softmax_outputs.append(torch.softmax(outputs, dim=1).cpu().numpy())
    
    hook.remove()
    
    softmax_outputs = np.array(softmax_outputs)
    mean_probabilities = np.mean(softmax_outputs, axis=0)
    epsilon = 1e-10
    predictive_entropy = -np.sum(mean_probabilities * np.log(mean_probabilities + epsilon), axis=1)
    
    model.eval()
    
    return predictive_entropy, np.array(features_at_layer)

def calculate_layer_uncertainty(features_at_layer):
    variance_of_features = np.var(features_at_layer, axis=0)
    layer_uncertainty_score = np.sum(variance_of_features, axis=1)
    
    return layer_uncertainty_score

def get_mahalanobis_parameters(model, trainloader, feature_layer, device):
    model.eval()
    features_list = []
    labels_list = []
    def hook_fn(module, input, output):
        flattened_output = output.view(output.size(0), -1)
        features_list.append(flattened_output.cpu().numpy())
    
    hook = feature_layer.register_forward_hook(hook_fn)
    
    print("--- Calculating Mahalanobis parameters on training set ---")
    with torch.no_grad():
        for images, labels in tqdm(trainloader, desc="Extracting features"):
            images = images.to(device)
            _ = model(images)
            labels_list.append(labels.cpu().numpy())
    
    hook.remove()
    
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    feature_dim = all_features.shape[1]
    mean_per_class = np.zeros((NUM_CLASSES, feature_dim))
    precision_per_class = np.zeros((NUM_CLASSES, feature_dim, feature_dim))
    
    for c in range(NUM_CLASSES):
        class_features = all_features[all_labels == c]
        if class_features.shape[0] > feature_dim:
            mean_per_class[c] = np.mean(class_features, axis=0)
            cov_matrix = np.cov(class_features.T)
            precision_per_class[c] = np.linalg.pinv(cov_matrix)
        else:
            mean_per_class[c] = np.mean(class_features, axis=0)
            precision_per_class[c] = np.eye(feature_dim)
            
    return mean_per_class, precision_per_class

def calculate_mahalanobis_distance(model, dataloader, device, feature_layer, mean_per_class, precision_per_class):
    model.eval()
    features_list = []
    def hook_fn(module, input, output):
        flattened_output = output.view(output.size(0), -1)
        features_list.append(flattened_output.cpu().numpy())
    
    hook = feature_layer.register_forward_hook(hook_fn)
    
    mahalanobis_scores = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Calculating Mahalanobis Distance"):
            images = images.to(device)
            features_list.clear()
            _ = model(images)
            features = features_list[0]
            
            min_distances = []
            for feature in features:
                dists = [mahalanobis(feature, mean_per_class[c], precision_per_class[c]) for c in range(NUM_CLASSES)]
                min_distances.append(min(dists))
            mahalanobis_scores.extend(min_distances)
    
    hook.remove()
    return np.array(mahalanobis_scores)

def calculate_decision_change(model, image, label, epsilon, device):
    model.eval()
    image.requires_grad = True
    
    outputs = model(image)
    _, original_pred = torch.max(outputs, 1)
    
    loss = nn.CrossEntropyLoss()(outputs, original_pred)
    model.zero_grad()
    loss.backward()
    
    perturbed_image = image.detach() + epsilon * image.grad.sign()
    
    with torch.no_grad():
        outputs_perturbed = model(perturbed_image)
        _, perturbed_pred = torch.max(outputs_perturbed, 1)
    
    return int(original_pred.item() != perturbed_pred.item())

def get_laplace_parameters(model, trainloader, feature_layer, device):
    model.eval()
    features_list = []
    labels_list = []
    def hook_fn(module, input, output):
        squeezed_output = output.squeeze().detach().cpu().numpy()
        features_list.append(squeezed_output)
    
    hook = feature_layer.register_forward_hook(hook_fn)
    
    print("--- Gathering features for Laplace Approximation on training set ---")
    with torch.no_grad():
        for images, labels in tqdm(trainloader, desc="Extracting features"):
            images = images.to(device)
            _ = model(images)
            labels_list.append(labels.cpu().numpy())
    
    hook.remove()
    
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    return all_features, all_labels

def calculate_laplace_uncertainty(model, dataloader, feature_layer, device, all_train_features, all_train_labels):
    model.eval()
    
    train_features_mean = np.mean(all_train_features, axis=0)
    train_features_cov = np.cov(all_train_features.T)
    
    precision_matrix_reg = np.linalg.pinv(train_features_cov + np.eye(train_features_cov.shape[0]) * 1e-4)
    
    laplace_uncertainty_scores = []
    features_list = []
    def hook_fn(module, input, output):
        squeezed_output = output.squeeze().detach().cpu().numpy()
        features_list.append(squeezed_output)
    
    hook = feature_layer.register_forward_hook(hook_fn)

    print("\n--- Calculating Laplace Uncertainty on test set ---")
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Laplace Inference"):
            images = images.to(device)
            features_list.clear()
            _ = model(images)
            test_features = features_list[0]
            
            for f in test_features:
                f_centered = f - train_features_mean
                laplace_var = f_centered.T @ precision_matrix_reg @ f_centered
                
                laplace_uncertainty_scores.append(laplace_var)

    hook.remove()
    return np.array(laplace_uncertainty_scores)

def calculate_msp(probabilities):
    return torch.max(probabilities).item()

def reverse_fgsm_attack(image, epsilon, data_grad, min_vals, max_vals):
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.max(perturbed_image, min_vals)
    perturbed_image = torch.min(perturbed_image, max_vals)
    return perturbed_image

def refine_dataset_with_uncertainty_metric(model, dataloader, criterion, epsilon, uncertainty_metric_name, metric_threshold, device):
    model.eval()
    min_vals = torch.tensor([(0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)
    max_vals = torch.tensor([(1 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)

    original_labels = []
    original_predictions = []
    refined_images_list = []
    refined_labels_list = []
    
    count_correct_uncertainty = 0
    count_incorrect_uncertainty = 0
    
    print("\n--- Processing images for uncertainty and perturbation ---")
    
    # Pre-calculate metric values
    all_metrics_values = get_all_uncertainty_metrics(model, dataloader, device)
    metric_values_for_refinement = all_metrics_values.get(uncertainty_metric_name)
    if metric_values_for_refinement is None:
        raise ValueError(f"Uncertainty metric '{uncertainty_metric_name}' not found. Please choose from: {list(all_metrics_values.keys())}")
    
    # We assume 'is_confidence_metric' is defined and passed or derived.
    is_confidence_metric = uncertainty_metric_name in ['msp', 'mahalanobis', 'laplace']
    
    for i, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Analyzing and Perturbing"):
        image = images.to(device)
        label = labels.to(device)
        
        with torch.no_grad():
            outputs_original = model(image)
            probabilities_original = torch.softmax(outputs_original, dim=1)
            _, predicted_original = torch.max(outputs_original.data, 1)

        current_score = metric_values_for_refinement[i]
        
        should_refine = (is_confidence_metric and current_score < metric_threshold) or \
                        (not is_confidence_metric and current_score > metric_threshold)
        
        if should_refine:
            if label.item() != predicted_original.item():
                count_correct_uncertainty += 1
            else:
                count_incorrect_uncertainty += 1

            current_image = image.clone().detach()
            count = 0
            while count < 10:
                with torch.no_grad():
                    outputs_check = model(current_image)
                    probabilities_check = torch.softmax(outputs_check, dim=1)
                    current_msp_for_refinement = torch.max(probabilities_check).item()
                
                if current_msp_for_refinement >= metric_threshold:
                    break
                
                current_image = current_image.detach().requires_grad_(True)
                
                output_for_grad = model(current_image)
                loss = logit_margin_loss(output_for_grad, label)
                
                model.zero_grad()
                loss.backward()
                data_grad = current_image.grad.data
                
                current_image = reverse_fgsm_attack(current_image, epsilon, data_grad, min_vals, max_vals)
                
                count += 1
            
            refined_images_list.append(current_image.squeeze(0).detach().cpu())
            refined_labels_list.append(label.item())
        else:
            refined_images_list.append(image.squeeze(0).detach().cpu())
            refined_labels_list.append(label.item())
            
        original_labels.extend(labels.cpu().numpy())
        original_predictions.extend(predicted_original.cpu().numpy())
    
    accuracy = accuracy_score(original_labels, original_predictions)
    precision = precision_score(original_labels, original_predictions, average='macro', zero_division=0)
    recall = recall_score(original_labels, original_predictions, average='macro', zero_division=0)
    f1 = f1_score(original_labels, original_predictions, average='macro', zero_division=0)
    
    print(f"  → Initial Accuracy : {accuracy:.4f}")
    print(f"  → Initial Precision: {precision:.4f}")
    print(f"  → Initial Recall   : {recall:.4f}")
    print(f"  → Initial F1-Score : {f1:.4f}")
    print(f"Correctly classified incorrect predictions: {count_correct_uncertainty}")
    print(f"Incorrectly classified incorrect predictions: {count_incorrect_uncertainty}")

    refined_dataset = torch.utils.data.TensorDataset(torch.stack(refined_images_list), torch.tensor(refined_labels_list))
    refined_dataloader = torch.utils.data.DataLoader(refined_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    return refined_dataloader

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

def get_all_uncertainty_metrics(model, dataloader, device):
    model.eval()
    all_logits = []
    all_predictions = []
    all_labels = []
    
    print("\n--- Calculating all uncertainty metrics for test set ---")
    
    # 1. Standard Metrics (MSP, Energy)
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Standard Inference for Metrics"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels_np = np.array(all_labels)
    
    uncertainty_data = {}
    
    # MSP and Energy
    probabilities = torch.softmax(torch.from_numpy(all_logits), dim=1)
    uncertainty_data['msp'] = np.max(probabilities.cpu().numpy(), axis=1)
    uncertainty_data['energy'] = calculate_energy(torch.from_numpy(all_logits))
    
    # Mahalanobis
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=T.Compose([
        T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
    feature_layer_mahalanobis = model.layer3[8].conv2
    mean_per_class, precision_per_class = get_mahalanobis_parameters(model, trainloader, feature_layer_mahalanobis, device)
    mahalanobis_scores = calculate_mahalanobis_distance(
        model, dataloader, device, feature_layer_mahalanobis, mean_per_class, precision_per_class
    )
    uncertainty_data['mahalanobis'] = mahalanobis_scores
    
    # Laplace
    feature_layer_laplace = model.avgpool
    train_features_for_laplace, train_labels_for_laplace = get_laplace_parameters(model, trainloader, feature_layer_laplace, device)
    laplace_scores = calculate_laplace_uncertainty(model, dataloader, feature_layer_laplace, device,
                                                all_train_features=train_features_for_laplace, all_train_labels=train_labels_for_laplace)
    uncertainty_data['laplace'] = laplace_scores

    # MC Dropout
    mc_dropout_uncertainty_list = []
    dropout_layer_for_lu = model.layer3[8]
    testloader_batch = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=T.Compose([
        T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])), batch_size=128, shuffle=False, num_workers=2)
    for images, _ in tqdm(testloader_batch, desc="MC Dropout"):
        images = images.to(device)
        mc_dropout_scores, _ = calculate_mc_dropout_uncertainty(model, images, NUM_MC_DROPOUT_SAMPLES, dropout_layer_for_lu, device)
        mc_dropout_uncertainty_list.extend(mc_dropout_scores)
    uncertainty_data['mc_dropout'] = np.array(mc_dropout_uncertainty_list)
    
    # Convert lists to numpy
    for key in uncertainty_data:
        uncertainty_data[key] = np.array(uncertainty_data[key])
        
    return uncertainty_data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    
    print("Loading pre-trained CIFAR-10 ResNet56 model...")
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    epsilon = 0.03
    
    # --- CHOOSE YOUR METRIC AND THRESHOLD HERE ---
    #uncertainty_metric_to_use = 'msp'
    uncertainty_metric_to_use = 'mahalanobis'
    metric_threshold_value = 350
    
    print(f"Refining dataset using '{uncertainty_metric_to_use}' with a threshold of {metric_threshold_value}")
    
    refined_dataloader = refine_dataset_with_uncertainty_metric(
        model,
        testloader,
        criterion,
        epsilon,
        uncertainty_metric_to_use,
        metric_threshold_value,
        device
    )

    print("\n--- Running final inference on refined dataset ---")
    final_accuracy, _, _, final_f1 = inference(model, refined_dataloader, device)

    print(f"Final Accuracy on Refined Dataset: {final_accuracy:.4f}")
    print(f"Final F1-Score on Refined Dataset: {final_f1:.4f}")

if __name__ == "__main__":
    main()
