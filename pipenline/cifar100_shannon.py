import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# ==================== Uncertainty Calculation Functions ====================
def calculate_shannon_entropy(logits):
    """
    Calculate Shannon entropy for a batch of logits.
    A higher value indicates higher uncertainty.
    """
    probabilities = F.softmax(logits, dim=1)
    # Add a small epsilon to probabilities to prevent log(0)
    eps = 1e-12
    entropy = -torch.sum(probabilities * torch.log2(probabilities + eps), dim=1)
    return entropy

def calculate_ratio_confidence(logits):
    """
    Calculate ratio uncertainty: ratio of top-2 to top-1 probabilities (p2 / p1).
    A lower value indicates higher confidence (less uncertainty).
    """
    probabilities = F.softmax(logits, dim=1)
    sorted_probs, _ = torch.topk(probabilities, 2, dim=1)
    p1 = sorted_probs[:, 0]
    p2 = sorted_probs[:, 1]
    ratio_uncertainty = torch.where(p1 > 0, p2 / p1, torch.zeros_like(p1))
    return ratio_uncertainty

def calculate_margin_confidence(logits):
    """
    Calculate margin uncertainty: 1 - (p1 - p2).
    A lower value indicates higher uncertainty.
    """
    probabilities = F.softmax(logits, dim=1)
    sorted_probs, _ = torch.topk(probabilities, 2, dim=1)
    p1 = sorted_probs[:, 0]
    p2 = sorted_probs[:, 1]
    margin = p1 - p2
    uncertainty = 1 - margin
    return uncertainty

# ======================== Model and Attack Functions ========================
def load_model():
    """Load the pretrained CIFAR-100 ResNet56 model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet56', pretrained=True).to(device)
    model.eval()
    return model, device

def reverse_fgsm_attack_entropy(model, x, alpha=0.005):
    """
    Reverse FGSM attack using Shannon Entropy as the loss function.
    The goal is to minimize entropy (increase confidence).
    """
    x_adv = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    outputs = model(x_adv)
    
    # Calculate entropy loss
    entropy_loss = torch.mean(calculate_shannon_entropy(outputs))
    
    # Minimize entropy loss to increase confidence
    entropy_loss.backward()
    
    grad_sign = x_adv.grad.data.sign()
    
    with torch.no_grad():
        # Move in the direction that decreases entropy
        x_adv = x_adv - alpha * grad_sign
        x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()

# ======================== Evaluation Logic (Updated) ========================
def evaluate_model_uncertainty(model, dataloader, device, uncertainty_type='entropy', 
                              entropy_threshold=1.16, ratio_threshold=0.2, margin_threshold=0.35, apply_attack=False):
    """
    Evaluate model and apply reverse FGSM attack based on uncertainty thresholds.
    """
    model.eval()
    correct = 0
    total = 0
    
    high_uncertainty_correct_before = 0
    high_uncertainty_incorrect_before = 0
    high_uncertainty_correct_after = 0
    high_uncertainty_incorrect_after = 0
    
    for data, targets in tqdm(dataloader, desc=f"Evaluating ({'attack' if apply_attack else 'no attack'})", unit="batch"):
        data, targets = data.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(data)
            initial_preds = outputs.argmax(dim=1)
            
            if uncertainty_type == 'entropy':
                uncertainty_scores = calculate_shannon_entropy(outputs)
                high_uncertainty_mask = uncertainty_scores > entropy_threshold
            elif uncertainty_type == 'ratio':
                uncertainty_scores = calculate_ratio_confidence(outputs)
                high_uncertainty_mask = uncertainty_scores > ratio_threshold
            else:  # 'margin'
                uncertainty_scores = calculate_margin_confidence(outputs)
                high_uncertainty_mask = uncertainty_scores > margin_threshold
        
        correct_mask = initial_preds == targets
        high_uncertainty_correct_before += (high_uncertainty_mask & correct_mask).sum().item()
        high_uncertainty_incorrect_before += (high_uncertainty_mask & ~correct_mask).sum().item()
        
        if apply_attack:
            modified_data = data.clone()
            high_uncertainty_indices = torch.where(high_uncertainty_mask)[0]
            
            for idx in high_uncertainty_indices:
                sample_data = data[idx:idx+1].clone()
                current_sample = sample_data
                
                for _ in range(10):
                    with torch.no_grad():
                        current_outputs = model(current_sample)
                        if uncertainty_type == 'entropy':
                            current_uncertainty = calculate_shannon_entropy(current_outputs)
                            threshold = entropy_threshold
                        elif uncertainty_type == 'ratio':
                            current_uncertainty = calculate_ratio_confidence(current_outputs)
                            threshold = ratio_threshold
                        else:
                            current_uncertainty = calculate_margin_confidence(current_outputs)
                            threshold = margin_threshold
                    
                    if current_uncertainty.item() <= threshold:
                        break
                    
                    # Use the entropy-based attack since we no longer have labels
                    current_sample = reverse_fgsm_attack_entropy(model, current_sample)
                
                modified_data[idx] = current_sample.squeeze(0)
            
            with torch.no_grad():
                final_outputs = model(modified_data)
                final_preds = final_outputs.argmax(dim=1)
                
                if uncertainty_type == 'entropy':
                    final_uncertainty_scores = calculate_shannon_entropy(final_outputs)
                    final_high_uncertainty_mask = final_uncertainty_scores > entropy_threshold
                elif uncertainty_type == 'ratio':
                    final_uncertainty_scores = calculate_ratio_confidence(final_outputs)
                    final_high_uncertainty_mask = final_uncertainty_scores > ratio_threshold
                else:
                    final_uncertainty_scores = calculate_margin_confidence(final_outputs)
                    final_high_uncertainty_mask = final_uncertainty_scores > margin_threshold
                
                final_correct_mask = final_preds == targets
                high_uncertainty_correct_after += (final_high_uncertainty_mask & final_correct_mask).sum().item()
                high_uncertainty_incorrect_after += (final_high_uncertainty_mask & ~final_correct_mask).sum().item()
                
                correct += (final_preds == targets).sum().item()
        else:
            correct += (initial_preds == targets).sum().item()
            high_uncertainty_correct_after += (high_uncertainty_mask & correct_mask).sum().item()
            high_uncertainty_incorrect_after += (high_uncertainty_mask & ~correct_mask).sum().item()
        
        total += targets.size(0)
    
    accuracy = 100. * correct / total
    return (accuracy, high_uncertainty_correct_before, high_uncertainty_incorrect_before,
            high_uncertainty_correct_after, high_uncertainty_incorrect_after)

# ======================== Main Execution Logic ========================
def run_experiment(uncertainty_type, threshold):
    """Run experiment for a specific uncertainty type"""
    print(f"Loading CIFAR-100 ResNet56 model for {uncertainty_type} uncertainty...")
    model, device = load_model()
    print(f"Model loaded on device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print(f"Dataset size: {len(test_dataset)} samples")
    
    print(f"\n" + "="*60)
    print(f"EVALUATING ORIGINAL MODEL - {uncertainty_type.upper()} UNCERTAINTY")
    print("="*60)
    (original_acc, orig_high_uncert_correct, orig_high_uncert_incorrect, _, _) = \
        evaluate_model_uncertainty(model, test_loader, device, uncertainty_type=uncertainty_type,
                                   entropy_threshold=threshold if uncertainty_type == 'entropy' else 0,
                                   ratio_threshold=threshold if uncertainty_type == 'ratio' else 0,
                                   margin_threshold=threshold if uncertainty_type == 'margin' else 0,
                                   apply_attack=False)
    
    print(f"\nOriginal accuracy: {original_acc:.2f}%")
    print(f"Original high {uncertainty_type} uncertainty (>{threshold}) correct predictions: {orig_high_uncert_correct}")
    print(f"Original high {uncertainty_type} uncertainty (>{threshold}) incorrect predictions: {orig_high_uncert_incorrect}")
    
    print(f"\n" + "="*60)
    print(f"EVALUATING WITH REVERSE FGSM ATTACK - {uncertainty_type.upper()} UNCERTAINTY")
    print("="*60)
    (attack_acc, _, _, final_high_uncert_correct, final_high_uncert_incorrect) = \
        evaluate_model_uncertainty(model, test_loader, device, uncertainty_type=uncertainty_type,
                                   entropy_threshold=threshold if uncertainty_type == 'entropy' else 0,
                                   ratio_threshold=threshold if uncertainty_type == 'ratio' else 0,
                                   margin_threshold=threshold if uncertainty_type == 'margin' else 0,
                                   apply_attack=True)
    
    print(f"\nAccuracy after reverse FGSM: {attack_acc:.2f}%")
    print(f"Final high {uncertainty_type} uncertainty (>{threshold}) correct predictions: {final_high_uncert_correct}")
    print(f"Final high {uncertainty_type} uncertainty (>{threshold}) incorrect predictions: {final_high_uncert_incorrect}")
    
    print(f"\n" + "="*60)
    print(f"SUMMARY - {uncertainty_type.upper()} UNCERTAINTY")
    print("="*60)
    print(f"Accuracy change: {original_acc:.2f}% -> {attack_acc:.2f}% ({attack_acc - original_acc:+.2f}%)")

def main():
    """Run experiments for all three uncertainty types"""
    print("="*80)
    print("CIFAR-100 UNCERTAINTY-BASED REVERSE FGSM ATTACK EXPERIMENTS")
    print("="*80)
    
    # Run Shannon Entropy experiment
    print("\n" + "#"*80)
    print("EXPERIMENT 1: SHANNON ENTROPY UNCERTAINTY (Threshold = 1.16)")
    print("#"*80)
    run_experiment('entropy', 1.16)
    
    # Run ratio uncertainty experiment
    print("\n" + "#"*80)
    print("EXPERIMENT 2: RATIO UNCERTAINTY (Threshold = 0.2)")
    print("#"*80)
    run_experiment('ratio', 0.2)
    
    # Run margin uncertainty experiment
    print("\n" + "#"*80)
    print("EXPERIMENT 3: MARGIN UNCERTAINTY (Threshold = 0.35)")
    print("#"*80)
    run_experiment('margin', 0.35)

if __name__ == "__main__":
    main()

