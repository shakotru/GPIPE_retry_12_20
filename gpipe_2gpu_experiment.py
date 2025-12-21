import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchgpipe import GPipe

DEPTHS = [8, 16, 32]
HIDDENS = [128, 256]
BATCH_SIZES = [8, 32]
CHUNKS = [1, 2, 4]

INPUT_DIM = 128
NUM_CLASSES = 10
LR = 1e-3


def build_layers(depth, hidden):
    layers = []
    dim = INPUT_DIM
    for _ in range(depth):
        layers.append(nn.Linear(dim, hidden))
        layers.append(nn.ReLU())
        dim = hidden
    layers.append(nn.Linear(dim, NUM_CLASSES))
    return nn.Sequential(*layers)


def run_experiment(depth, hidden, batch_size, chunks):
    devices = [0, 1]
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    layers = build_layers(depth, hidden)

    # split layers evenly across 2 stages
    num_layers = len(layers)
    balance = [num_layers // 2, num_layers - num_layers // 2]

    model = GPipe(
        layers,
        balance=balance,
        devices=devices,
        chunks=chunks,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, INPUT_DIM, device="cuda:0")
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device="cuda:0")

    # Warmup
    optimizer.zero_grad()
    loss_fn(model(x), y.to(model(x).device)).backward()
    optimizer.step()

    torch.cuda.synchronize()
    t0 = time.time()

    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y.to(out.device))
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()

    step_time = t1 - t0
    throughput = batch_size / step_time
    mem = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"DEPTH={depth}, HIDDEN={hidden}, BS={batch_size}, CHUNKS={chunks} | "
        f"Loss={loss.item():.4f}, "
        f"Time={step_time:.4f}s, "
        f"Throughput={throughput:.1f}/s, "
        f"Mem={mem:.2f}GB"
    )


if __name__ == "__main__":
    print("==== 2 GPU GPIPE ====")
    for d in DEPTHS:
        for h in HIDDENS:
            for b in BATCH_SIZES:
                for c in CHUNKS:
                    run_experiment(d, h, b, c)
