# Import system and timing utilities
import os
import time

# Import core PyTorch functionality
import torch
import torch.nn as nn
import torch.optim as optim

# Import PyTorch's distributed training tools
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import torchvision utilities for data loading and transformation
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

# Define a basic Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer: 1 input channel, 32 output channels, 5x5 filter with padding
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)  # Max pooling to reduce spatial dimensions
        # Second convolutional layer: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # Fully connected layer: input features are flattened from conv output
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Output layer: 10 classes for MNIST digits
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply first conv + ReLU + pooling
        x = self.pool(torch.relu(self.conv1(x)))
        # Apply second conv + ReLU + pooling
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten the output for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        # First dense layer + ReLU
        x = torch.relu(self.fc1(x))
        # Final output (logits for 10 classes)
        return self.fc2(x)

# Main training function
def main():
    # Retrieve the rank and world size from environment variables
    # Rank = unique ID of this process (0, 1, ...)
    # World size = total number of processes/nodes
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Initialize the default process group for communication (Gloo = CPU backend)
    dist.init_process_group(backend="gloo")
    # Set device to CPU (you’re not using GPU)
    device = torch.device("cpu")
    # Set random seed for reproducibility
    torch.manual_seed(0)
    # Prepare dataset and define data transformations
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    # Use a DistributedSampler to ensure each process gets a different data shard
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)

    # Test loader doesn't need sharding (will be used only by rank 0)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # Create model and wrap it with DDP for synchronized updates
    model = CNN().to(device)
    model = DDP(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Start timer
    start_time = time.time()

    # Train for 5 epochs
    for epoch in range(5):
        model.train()
        train_sampler.set_epoch(epoch)  # Shuffle data differently at each epoch
        running_loss = 0.0

        # Iterate through local shard of training data
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()           # Reset gradients
            output = model(data)            # Forward pass
            loss = criterion(output, target)  # Compute loss
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights
            running_loss += loss.item()     # Accumulate loss

        # Print epoch loss only from the main process (rank 0)
        if rank == 0:
            print(f"Epoch {epoch + 1} complete. Loss: {running_loss:.2f}")

    # Evaluate only from rank 0 to avoid duplicate output
    if rank == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).argmax(dim=1)  # Get predicted class
                correct += (output == target).sum().item()  # Count correct predictions

        # Calculate and print final accuracy and training time
        accuracy = correct / len(test_dataset)
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print(f"Total training time: {time.time() - start_time:.2f} seconds")

    # Shut down the process group after training is done
    dist.destroy_process_group()

# Entry point for the script
if __name__ == "__main__":
    main()
