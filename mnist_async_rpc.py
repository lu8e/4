# Import required modules
import os
import time
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define a simple CNN model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Two convolutional layers followed by ReLU and MaxPooling
        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5), nn.ReLU(), nn.MaxPool2d(2)
        )
        # Two fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(320, 50), nn.ReLU(), nn.Linear(50, 10)
        )

    def forward(self, x):
        # Forward pass through conv and fc layers
        x = self.conv(x)
        x = x.view(-1, 320)  # Flatten before FC layer
        return self.fc(x)


# Remote function to perform a training step

def train_step(remote_model_rref, data, target):
    # Pull the model from the remote reference (RRef)
    model = remote_model_rref.to_here().get_model()
    model.train()

    # Set up optimizer and compute forward pass
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)

    # Backpropagation and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Return scalar loss for logging
    return loss.item()


# Parameter Server class that holds the model

class ParameterServer:
    def __init__(self):
        # Initialize model and store it on CPU
        self.model = Net().to("cpu")

    def get_model(self):
        # Allows workers to retrieve the model
        return self.model


# Worker function: fetches model and trains
def run_worker(ps_rref, rank):
    # Download MNIST and create data loader
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    start = time.time()

    # Train for 3 epochs
    for epoch in range(3):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Send each batch to the parameter server to train remotely
            data, target = data.to("cpu"), target.to("cpu")
            loss = rpc.rpc_sync("ps", train_step, args=(ps_rref, data, target))
            epoch_loss += loss

        print(f"[Worker {rank}] Epoch {epoch + 1} Loss: {epoch_loss:.2f}")

    end = time.time()
    print(f"[Worker {rank}] Total training time: {end - start:.2f} seconds")

    # Evaluate final model on test set (optional)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    model = ps_rref.to_here().get_model()
    model.eval()
    correct = 0

    # Run evaluation without tracking gradients
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

    accuracy = correct / len(test_dataset)
    print(f"[Worker {rank}] Final Test Accuracy: {accuracy:.4f}")


# Entry point: starts as either PS or Worker

if __name__ == "__main__":
    import sys
    role = sys.argv[1]        # Role: "ps" or "worker"
    rank = int(sys.argv[2])   # Rank: unique ID per process
    world_size = int(sys.argv[3])  # Total number of processes
    master_ip = sys.argv[4]   # IP of the master node

    # Set up RPC communication settings
    os.environ["MASTER_ADDR"] = master_ip
    os.environ["MASTER_PORT"] = "29500"

    # Launch Parameter Server
    if role == "ps":
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        ps = ParameterServer()
        print("[PS] Running parameter server...")
        rpc.shutdown()

    # Launch Worker
    else:
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
        ps_rref = rpc.remote("ps", ParameterServer)
        run_worker(ps_rref, rank)
        rpc.shutdown()
