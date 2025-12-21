import torch
import torch.nn as nn
import torch.optim as optim
import time

DEPTHS = [8, 16, 32]
HIDDENS = [128, 256]
BATCH_SIZES = [8, 32]

INPUT_DIM = 128
NUM_CLASSES = 10
LR = 1e-3


def build_model(depth, hidden):
    layers = []
    dim = INPUT_DIM
    for _ in range(depth):
        layers.append(nn.Linear(dim, hidden))
        layers.append(nn.ReLU())
        dim = hidden
    layers.append(nn.Linear(dim, NUM_CLASSES))
    return nn.Sequential(*layers)


def run_experiment(depth, hidden, batch_size):
    device = "cuda:0"
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    model = build_model(depth, hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, INPUT_DIM, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)

    # Warmup
    optimizer.zero_grad()
    loss_fn(model(x), y).backward()
    optimizer.step()

    torch.cuda.synchronize()
    t0 = time.time()

    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()

    step_time = t1 - t0
    throughput = batch_size / step_time
    mem = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"DEPTH={depth}, HIDDEN={hidden}, BS={batch_size} | "
        f"Loss={loss.item():.4f}, "
        f"Time={step_time:.4f}s, "
        f"Throughput={throughput:.1f}/s, "
        f"Mem={mem:.2f}GB"
    )


if __name__ == "__main__":
    print("==== 1 GPU BASELINE ====")
    for d in DEPTHS:
        for h in HIDDENS:
            for b in BATCH_SIZES:
                run_experiment(d, h, b)
