import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load and download CIFAR-10 data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

num_ensemble_members = 5 # Number of models in the ensemble
ensemble_models = []
num_epochs_per_member = 10 # Epochs to train each member

print(f"\nStarting training for {num_ensemble_members} ensemble members...")
for i in range(num_ensemble_members):
    print(f"\nTraining Ensemble Member {i+1}/{num_ensemble_members}")
    
    # Create a new model instance for each member
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs_per_member):
        running_loss = 0.0
        for batch_idx, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'  Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')
    
    # Store the state dictionary of the trained model
    ensemble_models.append(model.state_dict())
    print(f"  Finished training member {i+1}.")

print("Finished training all ensemble members.")

all_uncertainties = []
all_correct = []

print("\nQuantifying uncertainty on the test set using the Deep Ensemble...")
# Set all ensemble models to evaluation mode
ensemble_instances = []
for state_dict in ensemble_models:
    model_instance = SimpleCNN().to(device)
    model_instance.load_state_dict(state_dict)
    model_instance.eval() # Important: Set to eval mode
    ensemble_instances.append(model_instance)

with torch.no_grad(): # Disable gradient calculations for inference
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        # For each image in the current batch
        for img_idx in range(images.size(0)):
            img = images[img_idx].unsqueeze(0) # Add batch dimension for single image
            true_label = labels[img_idx].item()

            individual_member_probs = []
            for member_model in ensemble_instances:
                output = member_model(img)
                probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
                individual_member_probs.append(probs)

            individual_member_probs = np.array(individual_member_probs) # Shape: (num_members, num_classes)

            # Calculate the mean probability distribution across ensemble members
            mean_predictive_probs = np.mean(individual_member_probs, axis=0)

            # Calculate the predictive variance across ensemble members for each class
            # We'll use the variance of the predicted probabilities for the predicted class
            # as our uncertainty measure.

            # Find the predicted class based on the mean probabilities
            predicted_class_idx = np.argmax(mean_predictive_probs)

            # The uncertainty is the variance of the probabilities for the predicted class
            # across the ensemble members.
            uncertainty_value = np.var(individual_member_probs[:, predicted_class_idx])

            is_correct = (predicted_class_idx == true_label)

            all_uncertainties.append(uncertainty_value)
            all_correct.append(is_correct)

all_uncertainties = np.array(all_uncertainties)
all_correct = np.array(all_correct)

# Create bins for the uncertainty values
num_bins = 20
# Define bins based on the range of calculated uncertainties
uncertainty_bins = np.linspace(all_uncertainties.min(), all_uncertainties.max(), num_bins + 1)
bin_indices = np.digitize(all_uncertainties, uncertainty_bins)

correct_counts = np.zeros(num_bins)
incorrect_counts = np.zeros(num_bins)

# Populate the bins with counts of correct and incorrect predictions
for i in range(1, num_bins + 1):
    bin_mask = (bin_indices == i)
    correct_counts[i-1] = np.sum(all_correct[bin_mask])
    incorrect_counts[i-1] = np.sum(~all_correct[bin_mask])

# Plot the bar graph
fig, ax = plt.subplots(figsize=(12, 6))
# Calculate bin centers for plotting
bin_centers = (uncertainty_bins[:-1] + uncertainty_bins[1:]) / 2

width = (uncertainty_bins[1] - uncertainty_bins[0]) * 0.4 # Adjust bar width for side-by-side display

ax.bar(bin_centers - width/2, correct_counts, width, label='Correct Images', color='skyblue')
ax.bar(bin_centers + width/2, incorrect_counts, width, label='Incorrect Images', color='salmon')

ax.set_xlabel('Uncertainty (Predictive Variance)', fontsize=12)
ax.set_ylabel('Number of Images', fontsize=12)
ax.set_title('Correct vs. Incorrect Predictions per Uncertainty Bin (Deep Ensemble)', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig("ensemble")
