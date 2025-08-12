import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

#confidence metrics (ratio and margin)
def calculate_ratio_uncertainty(logits):
    probabilities = F.softmax(logits, dim=1)
    sorted_probs, _ = torch.topk(probabilities, 2, dim=1)
    p1 = sorted_probs[:, 0]
    p2 = sorted_probs[:, 1]
    #can't divide by 0 
    ratio_uncertainty = torch.where(p1 > 0, p2 / p1, torch.zeros_like(p1))
    return ratio_uncertainty

def calculate_margin_confidence(logits):
    #calculate confidence not uncertainty 
    probabilities = F.softmax(logits, dim=1)
    sorted_probs, _ = torch.topk(probabilities, 2, dim=1)
    p1 = sorted_probs[:, 0]
    p2 = sorted_probs[:, 1]
    margin_confidence = p1 - p2
    return margin_confidence

# ==================== Unsupervised Loss Functions ====================
def margin_loss(logits):
    """
    Loss function to maximize margin confidence.
    Minimizing this loss maximizes the margin.
    """
    probabilities = F.softmax(logits, dim=1)
    sorted_probs, _ = torch.topk(probabilities, 2, dim=1)
    p1 = sorted_probs[:, 0]
    p2 = sorted_probs[:, 1]
    margin = p1 - p2
    return -torch.mean(margin)

def ratio_loss(logits):
    """
    Loss function to minimize ratio uncertainty.
    Minimizing this loss minimizes the ratio (p2/p1).
    """
    probabilities = F.softmax(logits, dim=1)
    sorted_probs, _ = torch.topk(probabilities, 2, dim=1)
    p1 = sorted_probs[:, 0]
    p2 = sorted_probs[:, 1]
    ratio = torch.where(p1 > 0, p2 / p1, torch.zeros_like(p1))
    return torch.mean(ratio)

#basic setup!
def load_model():
    """Load the pretrained CIFAR-100 ResNet56 model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet56', pretrained=True).to(device)
    model.eval()
    return model, device

def reverse_fgsm_attack_unsupervised(model, x, loss_fn, alpha=0.005):
    
    #reverse fgsm attack to increase accuracy 
    x_adv = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    outputs = model(x_adv)
    
    #had to include an option to provide the loss_fn since each metric requires different ones 
    uncertainty_loss = loss_fn(outputs)
    
    uncertainty_loss.backward()
    grad_sign = x_adv.grad.data.sign()
    
    with torch.no_grad():
        #decrease the uncertainty loss 
        x_adv = x_adv - alpha * grad_sign
        x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()

#checks how confident or uncertain the model is depending on the metric 
def evaluate_model_confidence(model, dataloader, device, confidence_type='ratio', 
                              ratio_threshold=0.2, margin_threshold=0.35, apply_attack=False):
    model.eval()
    correct = 0
    total = 0
    
    low_confidence_correct_before = 0
    low_confidence_incorrect_before = 0
    low_confidence_correct_after = 0
    low_confidence_incorrect_after = 0
    
    # testing out both the uncertainty metrics and the loss functions pertain to both metrics 
    if confidence_type == 'ratio':
        attack_loss_fn = ratio_loss
    else:
        attack_loss_fn = margin_loss
    
    for data, targets in tqdm(dataloader, desc=f"Evaluating ({'attack' if apply_attack else 'no attack'})", unit="batch"):
        data, targets = data.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(data)
            initial_preds = outputs.argmax(dim=1)
            
            if confidence_type == 'ratio':
                uncertainty_scores = calculate_ratio_uncertainty(outputs)
                low_confidence_mask = uncertainty_scores > ratio_threshold
            else:  # margin ratio calculation is margin confidence not uncertainty so thats' why it's above 
                confidence_scores = calculate_margin_confidence(outputs)
                low_confidence_mask = confidence_scores < margin_threshold
        
        correct_mask = initial_preds == targets
        low_confidence_correct_before += (low_confidence_mask & correct_mask).sum().item()
        low_confidence_incorrect_before += (low_confidence_mask & ~correct_mask).sum().item()
        
        if apply_attack:
            modified_data = data.clone()
            low_confidence_indices = torch.where(low_confidence_mask)[0]
            
            for idx in low_confidence_indices:
                sample_data = data[idx:idx+1].clone()
                
                current_sample = sample_data
                for _ in range(10):
                    with torch.no_grad():
                        current_outputs = model(current_sample)
                        if confidence_type == 'ratio':
                            current_uncertainty = calculate_ratio_uncertainty(current_outputs)
                            threshold = ratio_threshold
                            if current_uncertainty.item() <= threshold:
                                break
                        else: #if the uncertainty is still too high 
                            current_confidence = calculate_margin_confidence(current_outputs)
                            threshold = margin_threshold
                            if current_confidence.item() >= threshold:
                                break
                    
                    # has to change this since the loss function can't use ground truth labels 
                    current_sample = reverse_fgsm_attack_unsupervised(model, current_sample, attack_loss_fn)
                
                modified_data[idx] = current_sample.squeeze(0)
            
            with torch.no_grad():
                final_outputs = model(modified_data)
                final_preds = final_outputs.argmax(dim=1)
                
                if confidence_type == 'ratio':
                    final_uncertainty_scores = calculate_ratio_uncertainty(final_outputs)
                    final_low_confidence_mask = final_uncertainty_scores > ratio_threshold
                else:
                    final_confidence_scores = calculate_margin_confidence(final_outputs)
                    final_low_confidence_mask = final_confidence_scores < margin_threshold
                
                final_correct_mask = final_preds == targets
                low_confidence_correct_after += (final_low_confidence_mask & final_correct_mask).sum().item()
                low_confidence_incorrect_after += (final_low_confidence_mask & ~final_correct_mask).sum().item()
                
                correct += (final_preds == targets).sum().item()
        else:
            correct += (initial_preds == targets).sum().item()
            low_confidence_correct_after += (low_confidence_mask & correct_mask).sum().item()
            low_confidence_incorrect_after += (low_confidence_mask & ~correct_mask).sum().item()
        
        total += targets.size(0)
    
    accuracy = 100. * correct / total
    return (accuracy, low_confidence_correct_before, low_confidence_incorrect_before,
            low_confidence_correct_after, low_confidence_incorrect_after)

# ======================== Main Execution Logic ========================
def run_experiment(confidence_type, threshold):
    """Run experiment for a specific confidence type"""
    print(f"Loading CIFAR-100 ResNet56 model for {confidence_type} confidence...")
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
    print(f"EVALUATING ORIGINAL MODEL - {confidence_type.upper()} CONFIDENCE")
    print("="*60)
    (original_acc, orig_low_conf_correct, orig_low_conf_incorrect, _, _) = \
        evaluate_model_confidence(model, test_loader, device, confidence_type=confidence_type,
                                  ratio_threshold=threshold if confidence_type == 'ratio' else 0,
                                  margin_threshold=threshold if confidence_type == 'margin' else 0,
                                  apply_attack=False)
    
    print(f"\nOriginal accuracy: {original_acc:.2f}%")
    print(f"Original low {confidence_type} confidence correct predictions: {orig_low_conf_correct}")
    print(f"Original low {confidence_type} confidence incorrect predictions: {orig_low_conf_incorrect}")
    
    print(f"\n" + "="*60)
    print(f"EVALUATING WITH REVERSE FGSM ATTACK - {confidence_type.upper()} CONFIDENCE")
    print("="*60)
    (attack_acc, _, _, final_low_conf_correct, final_low_conf_incorrect) = \
        evaluate_model_confidence(model, test_loader, device, confidence_type=confidence_type,
                                  ratio_threshold=threshold if confidence_type == 'ratio' else 0,
                                  margin_threshold=threshold if confidence_type == 'margin' else 0,
                                  apply_attack=True)
    
    print(f"\nAccuracy after reverse FGSM: {attack_acc:.2f}%")
    print(f"Final low {confidence_type} confidence correct predictions: {final_low_conf_correct}")
    print(f"Final low {confidence_type} confidence incorrect predictions: {final_low_conf_incorrect}")
    
    print(f"\n" + "="*60)
    print(f"SUMMARY - {confidence_type.upper()} CONFIDENCE")
    print("="*60)
    print(f"Accuracy change: {original_acc:.2f}% -> {attack_acc:.2f}% ({attack_acc - original_acc:+.2f}%)")

def main():
    """Run experiments for both ratio and margin confidence"""
    print("="*80)
    print("CIFAR-100 CONFIDENCE-BASED UNCERTAINTY ATTACK EXPERIMENTS (UNSUPERVISED)")
    print("="*80)
    
    print("\n" + "#"*80)
    print("EXPERIMENT 1: RATIO UNCERTAINTY (Threshold = 0.2)")
    print("#"*80)
    run_experiment('ratio', 0.2)
    
    print("\n" + "#"*80)
    print("EXPERIMENT 2: MARGIN CONFIDENCE (Threshold = 0.35)")
    print("#"*80)
    run_experiment('margin', 0.35)

if __name__ == "__main__":
    main()

