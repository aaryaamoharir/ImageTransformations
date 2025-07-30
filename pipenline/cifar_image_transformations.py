import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import random
import copy

# normalization contants for CIFAR-10 dataset
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def denormalize_tensor(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def normalize_tensor(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor - mean) / std

def tensor_to_pil(tensor):
    denorm = denormalize_tensor(tensor)
    denorm = torch.clamp(denorm, 0, 1)
    return T.ToPILImage()(denorm)

def pil_to_tensor(pil_image):
    tensor = T.ToTensor()(pil_image)
    return normalize_tensor(tensor)

# Transformation functions
class TransformationPool:
    
    def gaussian_noise(image, severity=None):
        if severity is None:
            severity = random.choice([1, 2, 3, 4, 5])
        img_array = np.array(image).astype(np.float32)
        noise_levels = [0.08, 0.12, 0.18, 0.26, 0.38]
        noise_std = noise_levels[severity - 1]
        noise = np.random.normal(0, noise_std * 255, img_array.shape)
        corrupted = img_array + noise
        corrupted = np.clip(corrupted, 0, 255).astype(np.uint8)
        return Image.fromarray(corrupted)
    
    def impulse_noise(image, severity=None):
        if severity is None:
            severity = random.choice([1, 2, 3, 4, 5])
        img_array = np.array(image).astype(np.float32)
        noise_levels = [0.03, 0.06, 0.09, 0.17, 0.27]
        noise_prob = noise_levels[severity - 1]
        mask = np.random.random(img_array.shape[:2])
        img_array[mask < noise_prob / 2] = 0
        img_array[mask > 1 - noise_prob / 2] = 255
        return Image.fromarray(img_array.astype(np.uint8))
    
    def shot_noise(image, severity=None):
        if severity is None:
            severity = random.choice([1, 2, 3, 4, 5])
        img_array = np.array(image).astype(np.float32)
        noise_levels = [60, 25, 12, 5, 3]
        lambda_val = noise_levels[severity - 1]
        scaled = img_array / 255.0 * lambda_val
        noisy = np.random.poisson(scaled) / lambda_val * 255.0
        corrupted = np.clip(noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(corrupted)
    
    def defocus_blur(image, severity=None):
        if severity is None:
            severity = random.choice([1, 2, 3, 4, 5])
        blur_levels = [3, 4, 6, 8, 10]
        radius = blur_levels[severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    #transformations to increase accuracy 

    def enhance_contrast(image, factor=None):
        if factor is None:
            factor = random.uniform(0.5, 2.0)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    

    def enhance_brightness(image, factor=None):
        if factor is None:
            factor = random.uniform(0.5, 2.0)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    

    def enhance_sharpness(image, factor=None):
        if factor is None:
            factor = random.uniform(0.5, 3.0)
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    

    def enhance_color(image, factor=None):
        if factor is None:
            factor = random.uniform(0.5, 2.0)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    

    def motion_blur(image, size=None):
        if size is None:
            size = random.choice([5, 7, 9, 11])
        # Create motion blur kernel
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        
        img_array = np.array(image)
        blurred = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(blurred)
    

    def histogram_equalization(image):

        img_array = np.array(image)
        # Convert to YUV color space and equalize Y channel
        yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(equalized)

#apply individual transformations and corruptions 
class Individual:
    def __init__(self, transformations=None):
        self.transformations = transformations if transformations else []
        self.fitness = 0.0
        self.uncertainty_score = float('inf')
        self.transformed_image = None
        self.prediction = None
        self.confidence = 0.0
    
    def apply_transformations(self, original_image):
        pil_image = tensor_to_pil(original_image)
        
        for transform_name, transform_params in self.transformations:
            transform_func = getattr(TransformationPool, transform_name)
            if transform_params:
                pil_image = transform_func(pil_image, **transform_params)
            else:
                pil_image = transform_func(pil_image)
        
        self.transformed_image = pil_to_tensor(pil_image)
        return self.transformed_image
    
    def mutate(self, mutation_rate=0.3):
        if random.random() < mutation_rate:
            # Add 1-2 new transformations
            num_new = random.choice([1, 2])
            available_transforms = [
                'enhance_contrast', 'enhance_brightness', 'enhance_sharpness',
                'enhance_color', 'motion_blur'
            ]
            
            for _ in range(num_new):
                new_transform = random.choice(available_transforms)
                self.transformations.append((new_transform, {}))
    
    def crossover(self, other):
        # Take random subset from each parent
        parent1_subset = random.sample(self.transformations, 
                                     len(self.transformations) // 2) if self.transformations else []
        parent2_subset = random.sample(other.transformations,
                                     len(other.transformations) // 2) if other.transformations else []
        
        offspring_transforms = parent1_subset + parent2_subset
        return Individual(offspring_transforms)

def calculate_uncertainty_score(model, image_tensor, label, criterion, device):
    image_batch = image_tensor.unsqueeze(0).to(device)
    label_batch = label.unsqueeze(0).to(device) if label.dim() == 0 else label.to(device)
    
    with torch.no_grad():
        outputs = model(image_batch)
        probabilities = torch.softmax(outputs, dim=-1)
        _, prediction = torch.max(outputs, 1)
        
        # msp < 0.8 
        msp = torch.max(probabilities).item()
        
        # confidence 
        pred_confidence = probabilities[0, prediction].item()
        
        is_uncertain = msp < 0.8
        uncertainty_score = 1.0 if is_uncertain else 0.0
        
        return {
            'uncertainty_score': uncertainty_score,
            'prediction': prediction.item(),
            'confidence': pred_confidence,
            'msp': msp,
            'is_uncertain': is_uncertain
        }

def generate_initial_population(population_size=20):
    population = []
    
    # base corruptions to start with
    base_corruptions = [
        ('gaussian_noise', {'severity': 3}),
        ('impulse_noise', {'severity': 3}),
        ('shot_noise', {'severity': 3}),
        ('defocus_blur', {'severity': 3})
    ]
    
    for _ in range(population_size):
        transformations = base_corruptions.copy()
        
        # Add 0-3 random enhancement transformations
        enhancement_transforms = [
            'enhance_contrast', 'enhance_brightness', 'enhance_sharpness',
            'enhance_color', 'motion_blur'
        ]
        
        num_enhancements = random.randint(0, 3)
        for _ in range(num_enhancements):
            transform_name = random.choice(enhancement_transforms)
            transformations.append((transform_name, {}))
        
        population.append(Individual(transformations))
    
    return population

def evolutionary_search(model, original_image, label, criterion, device, 
                       generations=5, population_size=20, top_k=5):
    
    
    population = generate_initial_population(population_size)
    
    best_individuals_history = []
    
    for generation in range(generations):
        
        for individual in population:
            # Apply transformations
            individual.apply_transformations(original_image)
            results = calculate_uncertainty_score(
                model, individual.transformed_image, label, criterion, device
            )
            
            individual.uncertainty_score = results['uncertainty_score']
            individual.prediction = results['prediction']
            individual.confidence = results['confidence']
            
            # online it seems like fitness is a good way to see how uncertain the model is
            individual.fitness = 1.0 / (1.0 + individual.uncertainty_score)
        
        # sort by fitness (higher is better)
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # select top k candidates
        top_candidates = population[:top_k]
        best_individuals_history.append(copy.deepcopy(top_candidates[0]))
        
        
        if generation < generations - 1: 
            # create new population through crossover and mutation
            new_population = []
            
            # keep top candidates 
            new_population.extend(copy.deepcopy(top_candidates))
            
            # generate ofspring through crossover
            while len(new_population) < population_size * 0.7:
                parent1 = random.choice(top_candidates)
                parent2 = random.choice(top_candidates)
                offspring = parent1.crossover(parent2)
                new_population.append(offspring)
            
            # generate new random individuals
            while len(new_population) < population_size:
                new_individual = random.choice(generate_initial_population(1))
                new_population.append(new_individual)
            
            # apply mutations
            for individual in new_population[top_k:]:  # Don't mutate elites
                individual.mutate()
            
            population = new_population
    
    return population[0], best_individuals_history

def evaluate_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    return accuracy, precision, recall, f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # load CIFAR-10 test set (use smaller subset for demonstration)
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # test with 100 
    subset_size = 100  # Process first 100 images
    subset_indices = list(range(subset_size))
    subset = torch.utils.data.Subset(testset, subset_indices)
    testloader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=2)
    
    print("Loading pre-trained CIFAR-10 ResNet56 model...")
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Storage for results
    original_labels = []
    base_corrupted_predictions = []
    evolved_predictions = []
    
    print(f"\nProcessing {subset_size} images with evolutionary enhancement...")
    
    for i, (images, labels) in tqdm(enumerate(testloader), total=len(testloader), desc="Evolutionary Search"):
        image = images.squeeze(0)  # Remove batch dimension
        label = labels.squeeze(0)   # Remove batch dimension
        
        original_labels.append(label.item())
        
        # Apply base corruptions for comparison
        base_individual = Individual([
            ('gaussian_noise', {'severity': 3}),
            ('impulse_noise', {'severity': 3}),
            ('shot_noise', {'severity': 3}),
            ('defocus_blur', {'severity': 3})
        ])
        base_individual.apply_transformations(image)
        
        with torch.no_grad():
            base_output = model(base_individual.transformed_image.unsqueeze(0).to(device))
            _, base_pred = torch.max(base_output, 1)
            base_corrupted_predictions.append(base_pred.item())
        
        # Run evolutionary search
        best_individual, history = evolutionary_search(
            model, image, label, criterion, device,
            generations=3, population_size=15, top_k=3
        )
        
        evolved_predictions.append(best_individual.prediction)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{subset_size} images...")
    
    # Base corrupted results
    base_accuracy, base_precision, base_recall, base_f1 = evaluate_metrics(
        original_labels, base_corrupted_predictions
    )
    
    print(f"\nBASE CORRUPTED IMAGES (4 corruptions, severity 3):")
    print(f"  → Accuracy : {base_accuracy:.4f}")
    print(f"  → Precision: {base_precision:.4f}")
    print(f"  → Recall  : {base_recall:.4f}")
    print(f"  → F1-Score : {base_f1:.4f}")
    
    # Evolved results
    evolved_accuracy, evolved_precision, evolved_recall, evolved_f1 = evaluate_metrics(
        original_labels, evolved_predictions
    )
    
    print(f"\n EVOLUTIONARILY ENHANCED IMAGES:")
    print(f"  → Accuracy : {evolved_accuracy:.4f}")
    print(f"  → Precision: {evolved_precision:.4f}")
    print(f"  → Recall  : {evolved_recall:.4f}")
    print(f"  → F1-Score : {evolved_f1:.4f}")
    


if __name__ == "__main__":
    main()

