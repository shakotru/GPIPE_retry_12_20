import torch
import torch.nn as nn
import torch.optim as optim
import time

def main():
    device = "cuda:0"
    torch.cuda.set_device(0)

    # Model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 10),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Dummy data
    x = torch.randn(256, 1024, device=device)
    y = torch.randint(0, 10, (256,), device=device)

    torch.cuda.synchronize()
    t0 = time.time()

    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()

    print("==== 1 GPU Baseline ====")
    print(f"Loss: {loss.item():.4f}")
    print(f"Step time: {t1 - t0:.4f} s")

if __name__ == "__main__":
    main()
