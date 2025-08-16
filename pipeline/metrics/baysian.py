import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.distributions import Normal, Categorical
from torch.distributions.kl import kl_divergence

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

# Helper function for reparameterization trick
def reparameterize(mu, rho):
    sigma = torch.log(1 + torch.exp(rho))
    eps = torch.randn_like(sigma)
    return mu + sigma * eps

# Bayesian Linear Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        self.prior_sigma = prior_sigma
        self.prior_weight_dist = Normal(0, self.prior_sigma)
        self.prior_bias_dist = Normal(0, self.prior_sigma)

        self.kl_loss = 0

    def forward(self, input):
        weight = reparameterize(self.weight_mu, self.weight_rho)
        bias = reparameterize(self.bias_mu, self.bias_rho)

        posterior_weight_dist = Normal(self.weight_mu, torch.log(1 + torch.exp(self.weight_rho)))
        posterior_bias_dist = Normal(self.bias_mu, torch.log(1 + torch.exp(self.bias_rho)))

        kl_weight = kl_divergence(posterior_weight_dist, self.prior_weight_dist).sum()
        kl_bias = kl_divergence(posterior_bias_dist, self.prior_bias_dist).sum()
        self.kl_loss = kl_weight + kl_bias

        return nn.functional.linear(input, weight, bias)

# Bayesian Convolutional Layer
class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, prior_sigma=1.0):
        super(BayesianConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).uniform_(-5, -4))

        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(-5, -4))

        self.prior_sigma = prior_sigma
        self.prior_weight_dist = Normal(0, self.prior_sigma)
        self.prior_bias_dist = Normal(0, self.prior_sigma)

        self.kl_loss = 0

    def forward(self, input):
        weight = reparameterize(self.weight_mu, self.weight_rho)
        bias = reparameterize(self.bias_mu, self.bias_rho)

        posterior_weight_dist = Normal(self.weight_mu, torch.log(1 + torch.exp(self.weight_rho)))
        posterior_bias_dist = Normal(self.bias_mu, torch.log(1 + torch.exp(self.bias_rho)))

        kl_weight = kl_divergence(posterior_weight_dist, self.prior_weight_dist).sum()
        kl_bias = kl_divergence(posterior_bias_dist, self.prior_bias_dist).sum()
        self.kl_loss = kl_weight + kl_bias

        return nn.functional.conv2d(input, weight, bias, self.stride, self.padding)

class BayesianCNN(nn.Module):
    def __init__(self, prior_sigma=1.0):
        super(BayesianCNN, self).__init__()
        self.conv1 = BayesianConv2d(3, 64, kernel_size=5, padding=2, prior_sigma=prior_sigma)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = BayesianConv2d(64, 128, kernel_size=5, padding=2, prior_sigma=prior_sigma)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = BayesianLinear(128 * 8 * 8, 256, prior_sigma=prior_sigma)
        self.fc2 = BayesianLinear(256, 128, prior_sigma=prior_sigma)
        self.fc3 = BayesianLinear(128, 10, prior_sigma=prior_sigma)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_kl_loss(self):
        kl_sum = 0
        for m in self.modules():
            if isinstance(m, (BayesianLinear, BayesianConv2d)):
                kl_sum += m.kl_loss
        return kl_sum

model = BayesianCNN(prior_sigma=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

nll_criterion = nn.CrossEntropyLoss(reduction='sum')

num_epochs = 15
kl_weight = 1.0 / len(trainset)

print("\nStarting BNN Training with Variational Inference...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        data_likelihood_loss = nll_criterion(outputs, labels)
        kl_divergence_loss = model.get_kl_loss() * kl_weight

        loss = data_likelihood_loss + kl_divergence_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')

print("Finished BNN Training.")
def shannon_entropy(probs):
    probs = np.clip(probs, a_min=1e-12, a_max=1.0)
    return -np.sum(probs * np.log2(probs), axis=1)

num_forward_passes = 50
all_uncertainties = []
all_correct = []

print("\nQuantifying uncertainty on the test set...")
model.eval()

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        for img_idx in range(images.size(0)):
            img = images[img_idx].unsqueeze(0)
            true_label = labels[img_idx].item()

            individual_predictions = []
            for _ in range(num_forward_passes):
                output = model(img)
                probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
                individual_predictions.append(probs)

            individual_predictions = np.array(individual_predictions)
            mean_predictive_probs = np.mean(individual_predictions, axis=0)
            uncertainty_value = shannon_entropy(mean_predictive_probs.reshape(1, -1))[0]
            predicted_class = np.argmax(mean_predictive_probs)
            is_correct = (predicted_class == true_label)

            all_uncertainties.append(uncertainty_value)
            all_correct.append(is_correct)

all_uncertainties = np.array(all_uncertainties)
all_correct = np.array(all_correct)

# --- ADDED THRESHOLD METRIC ---
threshold = 0.5  # Set your desired threshold here

high_uncertainty_indices = np.where(all_uncertainties > threshold)
correct_high_uncertainty = np.sum(all_correct[high_uncertainty_indices])
incorrect_high_uncertainty = np.sum(~all_correct[high_uncertainty_indices])

print(f"\nAnalysis for uncertainty > {threshold}:")
print(f"Correctly classified images with high uncertainty: {correct_high_uncertainty}")
print(f"Incorrectly classified images with high uncertainty: {incorrect_high_uncertainty}")
print(f"Total images with high uncertainty: {correct_high_uncertainty + incorrect_high_uncertainty}")
# --- END ADDED THRESHOLD METRIC ---

num_bins = 20
uncertainty_bins = np.linspace(all_uncertainties.min(), all_uncertainties.max(), num_bins + 1)
bin_indices = np.digitize(all_uncertainties, uncertainty_bins)

correct_counts = np.zeros(num_bins)
incorrect_counts = np.zeros(num_bins)

for i in range(1, num_bins + 1):
    bin_mask = (bin_indices == i)
    correct_counts[i-1] = np.sum(all_correct[bin_mask])
    incorrect_counts[i-1] = np.sum(~all_correct[bin_mask])

fig, ax = plt.subplots(figsize=(12, 6))
bin_centers = (uncertainty_bins[:-1] + uncertainty_bins[1:]) / 2

width = (uncertainty_bins[1] - uncertainty_bins[0]) * 0.4

ax.bar(bin_centers - width/2, correct_counts, width, label='Correct Images', color='skyblue')
ax.bar(bin_centers + width/2, incorrect_counts, width, label='Incorrect Images', color='salmon')

ax.set_xlabel('Uncertainty (Shannon Entropy)', fontsize=12)
ax.set_ylabel('Number of Images', fontsize=12)
ax.set_title('Correct vs. Incorrect Predictions per Uncertainty Bin (Bayesian CNN with VI)', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig("bayians")
