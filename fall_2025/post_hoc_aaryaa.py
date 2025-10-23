"""
Complete Self-Contained Implementation of TTA-Boosted Post-hoc Calibration
Paper: Test Time Augmentation Meets Post-hoc Calibration
Authors: Hekler et al., AAAI 2023

This script includes all calibration methods and CIFAR-100 evaluation in one file.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from tqdm import tqdm


# ============================================================================
# PART 1: TEST TIME AUGMENTATION
# ============================================================================

class TestTimeAugmentation:
    """
    Test Time Augmentation as described in the paper.
    Fixed parameters matching the experimental setup:
    - Random horizontal flip
    - Rotation (±10°)
    - Zoom (1.0-1.1)
    - Brightness change (±0.1)
    - Symmetric warp (±0.2)
    """
    def __init__(self):
        # Define the augmentation pipeline exactly as specified
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=None,
                scale=(1.0, 1.1),  # Zoom 1.0-1.1
                shear=None
            ),
            transforms.ColorJitter(brightness=0.1),  # Brightness ±0.1
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5)  # Symmetric warp ±0.2
        ])
    
    def __call__(self, image):
        """Apply random augmentation to image"""
        return self.transforms(image)


# ============================================================================
# PART 2: POST-HOC CALIBRATION METHODS
# ============================================================================

class PostHocCalibrator:
    """Base class for post-hoc calibration methods"""
    def __init__(self):
        self.fitted = False
    
    def fit(self, logits, labels):
        """Fit calibrator on validation set"""
        raise NotImplementedError
    
    def calibrate(self, logits):
        """Apply calibration to logits"""
        raise NotImplementedError


class TemperatureScaling(PostHocCalibrator):
    """
    Temperature Scaling (TS) as described in Guo et al. 2017
    Single parameter T to rescale logits
    """
    def __init__(self):
        super().__init__()
        self.temperature = 1.0
    
    def fit(self, logits, labels):
        """
        Fit temperature parameter on validation set
        logits: (N, K) array of logits
        labels: (N,) array of true labels
        """
        logits = torch.FloatTensor(logits)
        labels = torch.LongTensor(labels)
        
        # Define negative log likelihood loss
        def nll_loss(T):
            T = torch.FloatTensor([T])
            scaled_logits = logits / T
            loss = nn.CrossEntropyLoss()(scaled_logits, labels)
            return loss.item()
        
        # Optimize temperature
        result = minimize(nll_loss, x0=1.0, method='Nelder-Mead', 
                         options={'maxiter': 1000})
        self.temperature = result.x[0]
        self.fitted = True
    
    def calibrate(self, logits):
        """Apply temperature scaling to logits"""
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before use")
        scaled_logits = logits / self.temperature
        probs = torch.softmax(torch.FloatTensor(scaled_logits), dim=-1)
        return probs.numpy()


class EnsembleTemperatureScaling(PostHocCalibrator):
    """
    Ensemble Temperature Scaling (ETS) as described in Zhang et al. 2020
    Weighted ensemble of 3 fixed temperatures
    """
    def __init__(self):
        super().__init__()
        self.fixed_temps = [1.0, 1.5, 2.5]  # Standard fixed temperatures
        self.weights = None
    
    def fit(self, logits, labels):
        """
        Fit ensemble weights on validation set
        logits: (N, K) array of logits
        labels: (N,) array of true labels
        """
        logits = torch.FloatTensor(logits)
        labels = torch.LongTensor(labels)
        
        # Compute probabilities for each temperature
        temp_probs = []
        for T in self.fixed_temps:
            scaled_logits = logits / T
            probs = torch.softmax(scaled_logits, dim=-1)
            temp_probs.append(probs)
        
        # Optimize weights
        def nll_loss(weights):
            w = torch.softmax(torch.FloatTensor(weights), dim=0)
            ensemble_probs = sum(w[i] * temp_probs[i] for i in range(len(self.fixed_temps)))
            loss = nn.CrossEntropyLoss()(torch.log(ensemble_probs + 1e-12), labels)
            return loss.item()
        
        result = minimize(nll_loss, x0=np.ones(len(self.fixed_temps)), 
                         method='Nelder-Mead', options={'maxiter': 1000})
        self.weights = torch.softmax(torch.FloatTensor(result.x), dim=0).numpy()
        self.fitted = True
    
    def calibrate(self, logits):
        """Apply ensemble temperature scaling to logits"""
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        logits = torch.FloatTensor(logits)
        ensemble_probs = 0
        for i, T in enumerate(self.fixed_temps):
            scaled_logits = logits / T
            probs = torch.softmax(scaled_logits, dim=-1)
            ensemble_probs += self.weights[i] * probs
        
        return ensemble_probs.numpy()


class IsotonicRegressionCalibration(PostHocCalibrator):
    """
    Isotonic Regression (IR) as described in Zadrozny & Elkan 2002
    """
    def __init__(self):
        super().__init__()
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, logits, labels):
        """
        Fit isotonic regression on validation set
        logits: (N, K) array of logits
        labels: (N,) array of true labels
        """
        # Get predicted probabilities
        probs = torch.softmax(torch.FloatTensor(logits), dim=-1).numpy()
        
        # Get confidence scores (max probability)
        confidences = np.max(probs, axis=1)
        
        # Get predicted labels
        predictions = np.argmax(probs, axis=1)
        
        # Binary accuracy (1 if correct, 0 if incorrect)
        accuracies = (predictions == labels).astype(float)
        
        # Fit isotonic regression
        self.calibrator.fit(confidences, accuracies)
        self.fitted = True
    
    def calibrate(self, logits):
        """Apply isotonic regression to logits"""
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        # Get predicted probabilities
        probs = torch.softmax(torch.FloatTensor(logits), dim=-1).numpy()
        
        # Get max probabilities
        max_probs = np.max(probs, axis=1, keepdims=True)
        
        # Apply calibration
        calibrated_max_probs = self.calibrator.predict(max_probs.flatten()).reshape(-1, 1)
        
        # Scale all probabilities proportionally
        calibrated_probs = probs * (calibrated_max_probs / (max_probs + 1e-12))
        
        # Renormalize
        calibrated_probs = calibrated_probs / (np.sum(calibrated_probs, axis=1, keepdims=True) + 1e-12)
        
        return calibrated_probs


class AccuracyPreservingIsotonicRegression(PostHocCalibrator):
    """
    Accuracy-preserving Isotonic Regression (IRM) as described in Zhang et al. 2020
    """
    def __init__(self):
        super().__init__()
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.original_predictions = None
    
    def fit(self, logits, labels):
        """
        Fit accuracy-preserving isotonic regression on validation set
        logits: (N, K) array of logits
        labels: (N,) array of true labels
        """
        # Get predicted probabilities
        probs = torch.softmax(torch.FloatTensor(logits), dim=-1).numpy()
        
        # Store original predictions
        self.original_predictions = np.argmax(probs, axis=1)
        
        # Get confidence scores (max probability)
        confidences = np.max(probs, axis=1)
        
        # Binary accuracy (1 if correct, 0 if incorrect)
        accuracies = (self.original_predictions == labels).astype(float)
        
        # Fit isotonic regression
        self.calibrator.fit(confidences, accuracies)
        self.fitted = True
    
    def calibrate(self, logits):
        """Apply accuracy-preserving isotonic regression to logits"""
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        # Get predicted probabilities
        probs = torch.softmax(torch.FloatTensor(logits), dim=-1).numpy()
        
        # Get max probabilities
        max_probs = np.max(probs, axis=1, keepdims=True)
        
        # Apply calibration
        calibrated_max_probs = self.calibrator.predict(max_probs.flatten()).reshape(-1, 1)
        
        # Scale probabilities
        calibrated_probs = probs * (calibrated_max_probs / (max_probs + 1e-12))
        
        # Renormalize
        calibrated_probs = calibrated_probs / (np.sum(calibrated_probs, axis=1, keepdims=True) + 1e-12)
        
        return calibrated_probs


# ============================================================================
# PART 3: TTA-BOOSTED CALIBRATION (Algorithm 1)
# ============================================================================

class TTABoostedCalibration:
    """
    Algorithm 1: TTA-based extension of common post-hoc recalibration algorithms
    Exactly as described in the paper
    """
    def __init__(self, model, calibrator, T=4):
        """
        Args:
            model: trained neural network f(·)
            calibrator: post-hoc calibrator c(·) tuned on validation set
            T: number of augmentations (default 4 as per paper)
        """
        self.model = model
        self.calibrator = calibrator
        self.T = T
        self.tta = TestTimeAugmentation()
        
    def predict(self, x):
        """
        Apply TTA-boosted calibration to a single test image
        
        Args:
            x: test image (tensor)
            
        Returns:
            p: calibrated uncertainty prediction
        """
        # Line 1: Initialize iteration counter
        t = 0
        
        # Line 2: Initialize logits list
        L = []
        
        # Line 3: Calculate logits of original image
        with torch.no_grad():
            logits = self.model(x.unsqueeze(0))  # Add batch dimension
            
        # Line 4: Add to list
        L.append(logits)
        
        # Line 5-10: TTA loop
        while t < self.T:
            # Line 6: Apply TTA to test image
            x_aug = self.tta(x)
            
            # Line 7: Calculate logits of augmentation
            with torch.no_grad():
                logits = self.model(x_aug.unsqueeze(0))
            
            # Line 8: Add to list
            L.append(logits)
            
            # Line 9: Increment counter
            t = t + 1
        # Line 10: end while
        
        # Line 11: Initialize mean logits
        l_hat = 0
        
        # Line 12-14: Calculate mean logits
        for logits in L:
            # Line 13: Accumulate
            l_hat = l_hat + logits
        # Line 14: end for
        
        # Line 15: Average
        l_hat = l_hat / (self.T + 1)
        
        # Line 16: Calculate calibrated uncertainty score
        p = self.calibrator.calibrate(l_hat.cpu().numpy())
        
        return p


# ============================================================================
# PART 4: EVALUATION METRICS
# ============================================================================

def compute_ece(confidences, predictions, labels, n_bins=20):
    """
    Compute Expected Calibration Error (ECE) as defined in equation (2)
    
    Args:
        confidences: predicted confidence scores
        predictions: predicted labels
        labels: true labels
        n_bins: number of bins (S=20 as per paper)
    
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Calculate accuracy in bin
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            # Calculate average confidence in bin
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            # Add weighted absolute difference
            ece += prop_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)
    
    return ece


def compute_brier_score(probs, labels, num_classes):
    """
    Compute Brier Score
    
    Args:
        probs: predicted probabilities (N, K)
        labels: true labels (N,)
        num_classes: number of classes K
    
    Returns:
        Brier score
    """
    # One-hot encode labels
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    
    # Compute mean squared error
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    
    return brier


def compute_nll(probs, labels):
    """
    Compute Negative Log Likelihood (NLL)
    
    Args:
        probs: predicted probabilities (N, K)
        labels: true labels (N,)
    
    Returns:
        NLL value
    """
    # Get probabilities of true class
    true_class_probs = probs[np.arange(len(labels)), labels]
    
    # Compute negative log likelihood
    nll = -np.mean(np.log(true_class_probs + 1e-12))
    
    return nll


# ============================================================================
# PART 5: CIFAR-100 EXPERIMENT
# ============================================================================

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================================================
    # 1. DATA PREPARATION
    # ========================================================================
    print("\n" + "="*60)
    print("Step 1: Preparing CIFAR-100 Dataset")
    print("="*60)

    # Define transforms
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Load CIFAR-100
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, 
                                              download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, 
                                             download=True, transform=transform_test)

    # Create train/val split (20% validation as per paper)
    train_size = len(trainset)
    val_size = int(0.2 * train_size)
    train_size = train_size - val_size

    indices = list(range(len(trainset)))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(trainset, train_indices)
    val_subset = Subset(trainset, val_indices)

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(testset)}")

    # Create data loaders
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # ========================================================================
    # 2. MODEL LOADING
    # ========================================================================
    print("\n" + "="*60)
    print("Step 2: Loading Pre-trained Model")
    print("="*60)

    try:
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar100_resnet56', 
                               pretrained=True).to(device)
        print("Loaded CIFAR-100 ResNet-56 model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', 
                               pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 100)
        model = model.to(device)
        print("Using CIFAR-10 model adapted for CIFAR-100")

    model.eval()

    # ========================================================================
    # 3. EXTRACT LOGITS FOR VALIDATION SET
    # ========================================================================
    print("\n" + "="*60)
    print("Step 3: Extracting Logits from Validation Set")
    print("="*60)

    def extract_logits_and_labels(model, data_loader, device):
        all_logits = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Extracting logits"):
                images = images.to(device)
                logits = model(images)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_logits = np.vstack(all_logits)
        all_labels = np.concatenate(all_labels)
        
        return all_logits, all_labels

    val_logits, val_labels = extract_logits_and_labels(model, val_loader, device)
    print(f"Validation logits shape: {val_logits.shape}")

    # ========================================================================
    # 4. FIT POST-HOC CALIBRATORS
    # ========================================================================
    print("\n" + "="*60)
    print("Step 4: Fitting Post-hoc Calibrators")
    print("="*60)

    calibrators = {
        'TS': TemperatureScaling(),
        'ETS': EnsembleTemperatureScaling(),
        'IR': IsotonicRegressionCalibration(),
        'IRM': AccuracyPreservingIsotonicRegression()
    }

    for name, calibrator in calibrators.items():
        print(f"Fitting {name}...")
        calibrator.fit(val_logits, val_labels)

    # ========================================================================
    # 5. EVALUATE ON TEST SET
    # ========================================================================
    print("\n" + "="*60)
    print("Step 5: Evaluating on Test Set")
    print("="*60)

    def evaluate_calibration(model, calibrator, data_loader, device, use_tta=False, T=4):
        all_probs = []
        all_labels = []
        
        model.eval()
        
        if use_tta:
            tta_calibrator = TTABoostedCalibration(model, calibrator, T=T)
            
            for images, labels in tqdm(data_loader, desc="TTA-boosted prediction"):
                batch_probs = []
                for img in images:
                    img = img.to(device)
                    probs = tta_calibrator.predict(img)
                    batch_probs.append(probs[0])
                
                all_probs.append(np.array(batch_probs))
                all_labels.append(labels.numpy())
        else:
            with torch.no_grad():
                for images, labels in tqdm(data_loader, desc="Standard prediction"):
                    images = images.to(device)
                    logits = model(images).cpu().numpy()
                    probs = calibrator.calibrate(logits)
                    all_probs.append(probs)
                    all_labels.append(labels.numpy())
        
        all_probs = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)
        
        predictions = np.argmax(all_probs, axis=1)
        confidences = np.max(all_probs, axis=1)
        
        accuracy = np.mean(predictions == all_labels)
        ece = compute_ece(confidences, predictions, all_labels, n_bins=20)
        brier = compute_brier_score(all_probs, all_labels, num_classes=100)
        nll = compute_nll(all_probs, all_labels)
        
        return {
            'accuracy': accuracy,
            'ece': ece,
            'brier': brier,
            'nll': nll
        }

    results = {}
    for name, calibrator in calibrators.items():
        print(f"\n{name}:")
        print("  Standard calibration...")
        results[f"{name}_standard"] = evaluate_calibration(
            model, calibrator, test_loader, device, use_tta=False
        )
        
        print("  TTA-boosted calibration (T=4)...")
        results[f"{name}_tta"] = evaluate_calibration(
            model, calibrator, test_loader, device, use_tta=True, T=4
        )

    # ========================================================================
    # 6. DISPLAY RESULTS
    # ========================================================================
    print("\n" + "="*60)
    print("RESULTS - CIFAR-100 Test Set Performance")
    print("="*60)
    print("\nComparing Standard vs TTA-Boosted Calibration\n")

    print(f"{'Method':<12} {'Acc (Base)':<12} {'Acc (TTA)':<12} {'ECE (Base)':<12} {'ECE (TTA)':<12} "
          f"{'BS (Base)':<12} {'BS (TTA)':<12} {'NLL (Base)':<12} {'NLL (TTA)':<12}")
    print("-" * 120)

    for base_name in ['TS', 'ETS', 'IR', 'IRM']:
        standard = results[f"{base_name}_standard"]
        tta = results[f"{base_name}_tta"]
        
        print(f"{base_name:<12} "
              f"{standard['accuracy']*100:>10.1f}%  "
              f"{tta['accuracy']*100:>10.1f}%  "
              f"{standard['ece']*100:>10.1f}%  "
              f"{tta['ece']*100:>10.1f}%  "
              f"{standard['brier']*100:>10.1f}  "
              f"{tta['brier']*100:>10.1f}  "
              f"{standard['nll']:>10.2f}  "
              f"{tta['nll']:>10.2f}")

    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)


if __name__ == "__main__":
    main()
