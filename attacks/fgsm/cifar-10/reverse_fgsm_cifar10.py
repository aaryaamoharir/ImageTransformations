import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow_datasets as tfds
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

def preprocess_tf_to_torch(image, label):
    # image: tf.uint8 Tensor [32,32,3]
    #img = image.numpy()
    img = T.ToTensor()(image)               # scales to [0,1]
    img = T.Normalize((0.4914, 0.4822, 0.4465),
                      (0.2023, 0.1994, 0.2010))(img)
    return img, int(label)

#load the dataset 
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)
#load the pretrained model 
print("Loading pretrained CIFAR10 ResNet56 model...")
model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet56',
        pretrained=True
    ).eval().to(device)
print("Model loaded successfully.")


criterion = nn.CrossEntropyLoss()

def evaluate_all_metrics(model, dataloader):
    model.eval() 
    all_labels = []
    all_predictions = []
    with torch.no_grad(): 
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) #for top-1 accuracy get the label with the highest value 
            
            all_labels.extend(labels.cpu().numpy()) #store true labels 
            all_predictions.extend(predicted.cpu().numpy()) #store predicted labels
    
    #calculate all the metrics 
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

min_vals = torch.tensor([(0 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)
max_vals = torch.tensor([(1 - m) / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)], device=device).view(1, 3, 1, 1)

def reverse_fgsm_attack(image, epsilon, data_grad, min_pixel_val, max_pixel_val):
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.max(perturbed_image, min_pixel_val)
    perturbed_image = torch.min(perturbed_image, max_pixel_val)

    return perturbed_image

# evaluate accuracy with reverse fgsm attack 
perturbed_correct = 0
perturbed_total = 0
epsilon = 0.03

print(f"\nApplying Reverse FGSM Attack with epsilon={epsilon} and evaluating accuracy...")
model.eval() # Ensure model is in evaluation mode
all_labels_perturbed = []
all_predictions_perturbed = []

for i, data in enumerate(testloader):
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    images.requires_grad = True
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data

    # apply the reverse fgsm attack  to get perturbed images
    perturbed_data = reverse_fgsm_attack(images, epsilon, data_grad, min_vals, max_vals)

    with torch.no_grad():
        output_perturbed = model(perturbed_data)
        _, final_pred = torch.max(output_perturbed.data, 1)

    all_labels_perturbed.extend(labels.cpu().numpy())
    all_predictions_perturbed.extend(final_pred.cpu().numpy())

#calcualte all the metrics and print them out 
perturbed_accuracy = accuracy_score(all_labels_perturbed, all_predictions_perturbed)
perturbed_precision = precision_score(all_labels_perturbed, all_predictions_perturbed, average='macro', zero_division=0)
perturbed_recall = recall_score(all_labels_perturbed, all_predictions_perturbed, average='macro', zero_division=0)
perturbed_f1 = f1_score(all_labels_perturbed, all_predictions_perturbed, average='macro', zero_division=0)
print(f"\nMetrics after Reverse FGSM Attack (epsilon={epsilon}):")
print(f"  Accuracy:  {perturbed_accuracy:.4f}")
print(f"  Precision: {perturbed_precision:.4f}")
print(f"  Recall:    {perturbed_recall:.4f}")
print(f"  F1 Score:  {perturbed_f1:.4f}")
