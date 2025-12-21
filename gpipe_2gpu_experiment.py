import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchgpipe import GPipe

def main():
    devices = [0, 1]
    torch.cuda.set_device(devices[0])

    # Model split into two stages
    layers = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 10),
    )

    # 2 stages
    partitions = [2, 3]  # number of layers per stage
    model = GPipe(layers, balance=partitions, devices=devices, chunks=8)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(256, 1024, device=f"cuda:{devices[0]}")
    y = torch.randint(0, 10, (256,), device=f"cuda:{devices[0]}")

    torch.cuda.synchronize()
    t0 = time.time()

    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()

    print("==== 2 GPU GPipe ====")
    print(f"Loss: {loss.item():.4f}")
    print(f"Step time: {t1 - t0:.4f} s")

if __name__ == "__main__":
    main()
