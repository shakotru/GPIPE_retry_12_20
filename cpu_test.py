import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchgpipe import GPipe

def main():
    # Simulate 4 GPUs on CPU
    devices = ["cpu", "cpu", "cpu", "cpu"]

    # Define the model (7 layers)
    layers = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    # Partition layers into 4 stages
    partitions = [2, 2, 2, 1]  # roughly equal layers per stage
    model = GPipe(layers, balance=partitions, devices=devices, chunks=4)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Dummy input and target
    x = torch.randn(16, 32, device="cpu")  # batch size 16
    y = torch.randint(0, 10, (16,), device="cpu")

    t0 = time.time()

    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    t1 = time.time()

    print("==== CPU 4-Stage GPipe Smoke Test ====")
    print(f"Loss: {loss.item():.4f}")
    print(f"Step time: {t1 - t0:.4f} s")

if __name__ == "__main__":
    main()
