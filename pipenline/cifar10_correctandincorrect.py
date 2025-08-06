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

# --- Configuration and Constants ---
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
NUM_CLASSES = 10
NUM_MC_DROPOUT_SAMPLES = 50

# --- Helper Functions (From previous conversations) ---

def logit_margin_loss(logits, labels, margin=1.0):
    """
    Calculates a custom logit margin loss.
    This loss encourages the logit of the true class to be at least 'margin'
    greater than the maximum logit of all other classes.
    """
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1])
    correct_class_logits = torch.sum(logits * one_hot_labels, dim=1)
    incorrect_logits = logits + (-1e9 * one_hot_labels)
    max_incorrect_logits = torch.max(incorrect_logits, dim=1)[0]
    loss = torch.relu(max_incorrect_logits - correct_class_logits + margin)
    return torch.mean(loss)

def calculate_energy(logits):
    """Calculates the energy score from logits."""
    return -torch.logsumexp(logits, dim=1).cpu().numpy()

def calculate_mc_dropout_uncertainty(model, image, num_samples, dropout_layer, device):
    """
    Calculates predictive entropy and layer uncertainty using Monte Carlo Dropout.
    Model should have dropout layers and be in train() mode for this to work.
    """
    model.train()
    softmax_outputs = []
    features_at_layer = []

    def hook_fn(module, input, output):
        features_at_layer.append(output.detach().cpu().numpy())
    
    hook = dropout_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(image)
            softmax_outputs.append(torch.softmax(outputs, dim=1).cpu().numpy())
    
    hook.remove()
    
    softmax_outputs = np.array(softmax_outputs)
    mean_probabilities = np.mean(softmax_outputs, axis=0)
    epsilon = 1e-10
    predictive_entropy = -np.sum(mean_probabilities * np.log(mean_probabilities + epsilon), axis=1)
    
    model.eval()
    
    return predictive_entropy, np.array(features_at_layer)

def calculate_layer_uncertainty(features_at_layer):
    """
    Calculates layer uncertainty as the variance of feature vectors from multiple runs.
    """
    variance_of_features = np.var(features_at_layer, axis=0)
    layer_uncertainty_score = np.sum(variance_of_features, axis=1)
    
    return layer_uncertainty_score

def get_mahalanobis_parameters(model, trainloader, feature_layer, device):
    """
    Calculates the mean and precision matrix for each class on the training set.
    """
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
    """
    Calculates the Mahalanobis distance from feature space to each class mean.
    Requires pre-calculated mean and precision matrix.
    """
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
    """
    Calculates Decision Change uncertainty by performing a small adversarial perturbation.
    Returns 1 if the prediction changes, 0 otherwise.
    """
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
    """
    Gathers last-layer features and labels from the training set for Laplace approximation.
    """
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

def calculate_laplace_uncertainty(model, dataloader, feature_layer, device,
                                  all_train_features, all_train_labels):
    """
    Performs last-layer Laplace approximation to get predictive variance uncertainty.
    """
    model.eval()
    
    # Compute feature-space mean and covariance from training data
    train_features_mean = np.mean(all_train_features, axis=0)
    train_features_cov = np.cov(all_train_features.T)
    
    # Add a small regularization term to the precision matrix
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
                # Corrected predictive variance calculation. This is a scalar for a single output.
                laplace_var = f_centered.T @ precision_matrix_reg @ f_centered
                
                laplace_uncertainty_scores.append(laplace_var)

    hook.remove()
    return np.array(laplace_uncertainty_scores)

def analyze_uncertainty_thresholds(uncertainty_data: Dict[str, np.ndarray], correct_predictions: np.ndarray):
    """
    Analyzes uncertainty metrics and returns performance metrics including optimal thresholds.
    """
    results = {}
    
    for metric_name, values in uncertainty_data.items():
        if values is None or len(np.unique(values)) < 2:
            print(f"Skipping {metric_name}: insufficient data or constant values.")
            continue
            
        is_confidence_metric = metric_name in ['msp']
        
        y_true = 1 - correct_predictions
        y_scores = -values if is_confidence_metric else values
        
        try:
            auc_roc = roc_auc_score(y_true, y_scores)
            fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
            
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            auc_pr = average_precision_score(y_true, y_scores)
            
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds_roc[optimal_idx]
            
            if is_confidence_metric:
                optimal_threshold = -optimal_threshold
            
            results[metric_name] = {
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'optimal_threshold': optimal_threshold,
                'fpr': fpr,
                'tpr': tpr,
                'precision': precision,
                'recall': recall,
                'values': values,
                'correct': correct_predictions
            }
        except Exception as e:
            print(f"Error analyzing {metric_name}: {e}")
            continue
            
    return results

def plot_individual_uncertainty_histograms(results):
    """Creates individual histogram plots for each uncertainty metric with uncertainty values on x-axis."""
    metrics = list(results.keys())
    print(f"\nSaving individual histogram plots for {len(metrics)} uncertainty metrics...")
    
    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        values = results[metric]['values']
        correct = results[metric]['correct']
        
        correct_values = values[correct == 1]
        incorrect_values = values[correct == 0]
        
        min_val = np.min(values)
        max_val = np.max(values)
        bins = np.linspace(min_val, max_val, 31)
        
        ax.hist(correct_values, bins=bins, alpha=0.7, label='Correct', density=True, color='skyblue')
        ax.hist(incorrect_values, bins=bins, alpha=0.7, label='Misclassified', density=True, color='lightcoral')
        
        threshold = results[metric]['optimal_threshold']
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal threshold: {threshold:.3f}')
        
        metric_name = metric.replace('_', ' ').title()
        ax.set_title(f'Distribution of {metric_name} Scores vs. Predictions')
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        auc_roc = results[metric]['auc_roc']
        ax.text(0.02, 0.98, f'AUC-ROC: {auc_roc:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        filename = f'uncertainty_histogram_{metric}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()

def plot_uncertainty_analysis(results):
    """Create comprehensive plots for uncertainty analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Uncertainty Metrics Analysis', fontsize=16)
    
    metrics = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
    
    ax1 = axes[0, 0]
    for i, metric in enumerate(metrics):
        ax1.plot(results[metric]['fpr'], results[metric]['tpr'],
                 label=f"{metric} (AUC={results[metric]['auc_roc']:.3f})",
                 color=colors[i])
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves (Misclassification Detection)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for i, metric in enumerate(metrics):
        ax2.plot(results[metric]['recall'], results[metric]['precision'],
                 label=f"{metric} (AUC={results[metric]['auc_pr']:.3f})",
                 color=colors[i])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves (Misclassification Detection)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    auc_roc_scores = [results[m]['auc_roc'] for m in metrics]
    bars = ax3.bar(metrics, auc_roc_scores, color=colors)
    ax3.set_title('AUC-ROC Scores')
    ax3.set_ylabel('AUC-ROC')
    ax3.set_ylim(0, 1)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    for bar, score in zip(bars, auc_roc_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom')

    ax4 = axes[1, 1]
    thresholds = [results[m]['optimal_threshold'] for m in metrics]
    bars2 = ax4.bar(metrics, thresholds, color=colors)
    ax4.set_title('Optimal Thresholds (Youden\'s J)')
    ax4.set_ylabel('Threshold Value')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    for bar, thresh in zip(bars2, thresholds):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{thresh:.3f}', ha='center', va='bottom')
                 
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('uncertainty_analysis_summary.png', dpi=300)
    print("Saved uncertainty analysis summary to 'uncertainty_analysis_summary.png'")
    plt.close()

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

def calculate_msp(probabilities):
    return torch.max(probabilities).item()

def reverse_fgsm_attack(image, epsilon, data_grad, min_vals, max_vals):
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.max(perturbed_image, min_vals)
    perturbed_image = torch.min(perturbed_image, max_vals)
    return perturbed_image

def refine_dataset_with_uncertainty_metric(model, dataloader, criterion, epsilon, metric_values, metric_threshold, is_confidence_metric, device):
    model.eval()
    count_correct_uncertainty = 0
    count_incorrect_uncertainty = 0
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
        
        with torch.no_grad():
            outputs_original = model(image)
            _, predicted_original = torch.max(outputs_original.data, 1)
            original_labels.extend(labels.cpu().numpy())
            original_predictions.extend(predicted_original.cpu().numpy())

        current_score = metric_values[i]
        should_refine = (is_confidence_metric and current_score < metric_threshold) or \
                        (not is_confidence_metric and current_score > metric_threshold)
        
        if should_refine:
            if label.item() != predicted_original.item():
                count_correct_uncertainty += 1
            else:
                count_incorrect_uncertainty += 1

            current_image = image.clone().detach()  
            count = 0
            while should_refine and count < 10:
                current_image = current_image.detach().requires_grad_(True)
                
                output_for_grad = model(current_image)
                loss = logit_margin_loss(output_for_grad, label)
                
                model.zero_grad()
                loss.backward()
                data_grad = current_image.grad.data
                
                current_image = reverse_fgsm_attack(current_image, epsilon, data_grad, min_vals, max_vals)
                
                with torch.no_grad():
                    outputs_new = model(current_image)
                    probabilities_new = torch.softmax(outputs_new.squeeze(0), dim=-1)
                    current_msp_for_refinement = torch.max(probabilities_new).item()
                    
                current_score = current_msp_for_refinement 
                should_refine = current_score < 0.998
                
                count += 1
            
            refined_images_list.append(current_image.squeeze(0).detach().cpu())
            refined_labels_list.append(label.item())

        else:
            refined_images_list.append(image.squeeze(0).detach().cpu())
            refined_labels_list.append(label.item())
            
    accuracy = accuracy_score(original_labels, original_predictions)
    precision = precision_score(original_labels, original_predictions, average='macro', zero_division=0)
    recall = recall_score(original_labels, original_predictions, average='macro', zero_division=0)
    f1 = f1_score(original_labels, original_predictions, average='macro', zero_division=0)

    print(f"  → Initial Accuracy : {accuracy:.4f}")
    print(f"  → Initial Precision: {precision:.4f}")
    print(f"  → Initial Recall   : {recall:.4f}")
    print(f"  → Initial F1-Score : {f1:.4f}")
    print(f"Correctly classified incorrect predictions {count_correct_uncertainty}")
    print(f"Incorrectly classified incorrect predictions {count_incorrect_uncertainty}")

    refined_dataset = torch.utils.data.TensorDataset(torch.stack(refined_images_list), torch.tensor(refined_labels_list))
    refined_dataloader = torch.utils.data.DataLoader(refined_dataset, batch_size=128, shuffle=False, num_workers=2)

    return refined_dataloader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)
    testloader_single = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    testloader_batch = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    print("Loading pre-trained CIFAR-10 ResNet56 model...")
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True).to(device)
    model.eval()

    # --- Pre-processing for Bayesian methods ---
    print("\n--- Pre-processing for Bayesian methods ---")
    feature_layer_laplace = model.avgpool 
    train_features_for_laplace, train_labels_for_laplace = get_laplace_parameters(model, trainloader, feature_layer_laplace, device)
    
    all_labels = []
    all_predictions = []
    
    uncertainty_data = {
        'msp': [],
        'energy': [],
        'mc_dropout': [],
        'mahalanobis': [],
        'decision_change': [],
        'layer_uncertainty': [],
        'laplace': [],
    }

    print("--- Calculating all uncertainty metrics ---")
    with torch.no_grad():
        for images, labels in tqdm(testloader_batch, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            uncertainty_data['msp'].extend(torch.max(probabilities, dim=1).values.cpu().numpy())
            uncertainty_data['energy'].extend(calculate_energy(outputs))

    correct_predictions = (np.array(all_predictions) == np.array(all_labels)).astype(int)

    print("\n--- Calculating Monte Carlo Dropout & Layer Uncertainty ---")
    mc_dropout_uncertainty_list = []
    layer_uncertainty_list = []
    dropout_layer_for_lu = model.layer3[8]
    for images, _ in tqdm(testloader_batch, desc="MC Dropout/LU"):
        images = images.to(device)
        mc_dropout_scores, features = calculate_mc_dropout_uncertainty(model, images, NUM_MC_DROPOUT_SAMPLES, dropout_layer_for_lu, device)
        mc_dropout_uncertainty_list.extend(mc_dropout_scores)
        layer_uncertainty_list.extend(calculate_layer_uncertainty(features))
    
    uncertainty_data['mc_dropout'] = np.array(mc_dropout_uncertainty_list)
    uncertainty_data['layer_uncertainty'] = np.array(layer_uncertainty_list)

    print("\n--- Calculating Mahalanobis Distance-based Uncertainty ---")
    feature_layer_mahalanobis = model.layer3[8].conv2
    mean_per_class, precision_per_class = get_mahalanobis_parameters(model, trainloader, feature_layer_mahalanobis, device)
    uncertainty_data['mahalanobis'] = calculate_mahalanobis_distance(
        model, testloader_batch, device, feature_layer_mahalanobis, mean_per_class, precision_per_class
    )
    
    print("\n--- Calculating Laplace Uncertainty ---")
    laplace_scores = calculate_laplace_uncertainty(model, testloader_batch, feature_layer_laplace, device,
                                                all_train_features=train_features_for_laplace, all_train_labels=train_labels_for_laplace)
    uncertainty_data['laplace'] = laplace_scores

    print("\n--- Calculating Decision Change Uncertainty ---")
    decision_change_scores = []
    epsilon = 0.01
    for images, labels in tqdm(testloader_single, desc="Decision Change"):
        images = images.to(device)
        decision_change_scores.append(calculate_decision_change(model, images, labels, epsilon, device))
    uncertainty_data['decision_change'] = np.array(decision_change_scores)
    
    uncertainty_data['msp'] = np.array(uncertainty_data['msp'])
    uncertainty_data['energy'] = np.array(uncertainty_data['energy'])

    print("\n--- Analyzing Uncertainty Metrics and Plotting ---")
    results = analyze_uncertainty_thresholds(uncertainty_data, correct_predictions)
    
    print("\nUncertainty Metrics Performance:")
    results_df = pd.DataFrame({
        m: { 'auc_roc': r['auc_roc'], 'auc_pr': r['auc_pr'], 'threshold': r['optimal_threshold']}
        for m, r in results.items()
    }).T
    print(results_df.sort_values(by='auc_roc', ascending=False))
    
    plot_uncertainty_analysis(results)
    plot_individual_uncertainty_histograms(results)
    
    print("\n--- Now running dataset refinement process with Mahalanobis Scores ---")
    
    laplace_threshold = results['laplace']['optimal_threshold']
    print(laplace_threshold)
    refined_dataloader = refine_dataset_with_uncertainty_metric(
        model, 
        testloader_single,
        nn.CrossEntropyLoss(), 
        0.03, 
        uncertainty_data['laplace'],
        laplace_threshold,
        is_confidence_metric=False,
        device=device
    )

    print("\n--- Running final inference on refined dataset ---")
    final_accuracy, _, _, final_f1 = inference(model, refined_dataloader, device)

    print(f"Final Accuracy on Refined Dataset: {final_accuracy:.4f}")
    print(f"Final F1-Score on Refined Dataset: {final_f1:.4f}")

if __name__ == "__main__":
    main()
