# Import core PyTorch libraries
import torch
import torch.nn as nn  # Neural network modules (e.g., layers, loss functions)
import torch.optim as optim  # Optimizers like SGD
# Import torchvision utilities for datasets and transformations
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# For timing the training duration
import time



# Define a Convolutional Neural Network (LeNet-style)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer: input=1 channel (grayscale), output=32 channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        # Max pooling layer with 2x2 window (downsampling)
        self.pool = nn.MaxPool2d(2)
        # Second convolutional layer: input=32, output=64 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # Fully connected layer (from flattened conv features to 128 units)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Output layer: maps 128 features to 10 output classes (digits 0–9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply conv1 -> ReLU -> maxpool
        x = self.pool(torch.relu(self.conv1(x)))
        # Apply conv2 -> ReLU -> maxpool
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        # Apply first fully connected layer with ReLU
        x = torch.relu(self.fc1(x))
        # Output layer (no softmax needed — handled by CrossEntropyLoss)
        return self.fc2(x)

# Setup

# Set computation device (CPU only in this case)
device = torch.device('cpu')
print(f"Using device: {device}")


# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()  # Convert images to PyTorch tensors (normalizes pixel values to [0,1])
])

# Download and load training data
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Wrap datasets in DataLoaders for batching and shuffling
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000)


# Initialize model, loss function, and optimizer
model = CNN().to(device)  # Move model to the selected device
criterion = nn.CrossEntropyLoss()  # CrossEntropy combines softmax + negative log likelihood
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer


# Training loop
start_time = time.time()  # Track total training time

# Loop through multiple epochs
for epoch in range(5):
    model.train()  # Set model to training mode
    running_loss = 0.0  # Track loss for the epoch

    # Loop through each batch of training data
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # Forward pass -> loss -> backward pass -> update weights
        optimizer.zero_grad()  # Clear previous gradients
        output = model(data)  # Compute model predictions
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        running_loss += loss.item()  # Accumulate loss


    # Evaluate the model on test data
    model.eval()  # Set model to evaluation mode
    correct = 0  # Counter for correct predictions

    with torch.no_grad():  # No need to compute gradients for inference
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).argmax(dim=1)  # Get index of max log-probability (predicted class)
            correct += (output == target).sum().item()

    # Compute and display accuracy for this epoch
    accuracy = correct / len(test_set)
    print(f"Epoch {epoch + 1}: Loss = {running_loss:.2f}, Test Accuracy = {accuracy:.4f}")

# Total training time
total_time = time.time() - start_time
print(f"\nTotal training time: {total_time:.2f} seconds")

# Final evaluation on test set
model.eval()
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data).argmax(dim=1)
        correct += (output == target).sum().item()

# Print final test accuracy
print(f"Final Test Accuracy: {correct / len(test_set):.4f}")
