import torch
import torch.nn as nn
import time
import csv
from datetime import datetime
import itertools
import os

# -----------------------------
# Configurable experiment grid
# -----------------------------
DEPTHS = [8, 16, 32, 64, 128]
HIDDEN_DIMS = [128, 256, 512, 1024]
BATCH_SIZES = [8, 32, 64, 128]
CHUNKS = [1, 2, 4, 8, 16]   # kept for consistency (no real effect on 1 GPU)
INPUT_DIM = 1024
NUM_CLASSES = 10
STEPS = 5  # warm runs already amortized

device = torch.device("cuda:0")

# -----------------------------
# Model definition
# -----------------------------
class MLP(nn.Module):
    def __init__(self, depth, hidden_dim):
        super().__init__()
        layers = []
        layers.append(nn.Linear(INPUT_DIM, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# CSV setup
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_name = f"gpipe_1gpu_results_{timestamp}.csv"

with open(csv_name, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "gpu_count",
        "depth",
        "hidden_dim",
        "batch_size",
        "chunks",
        "INPUT_DIM",
        "NUM_CLASSES",
        "forward_time_s",
        "backward_time_s",
        "total_step_time_s",
        "throughput_samples_per_s",
        "max_memory_gb",
        "loss"
    ])

# -----------------------------
# Main experiment loop
# -----------------------------
loss_fn = nn.CrossEntropyLoss()

for depth, hidden_dim, batch_size, chunks in itertools.product(
    DEPTHS, HIDDEN_DIMS, BATCH_SIZES, CHUNKS
):
    model = MLP(depth, hidden_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(batch_size, INPUT_DIM, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # -----------------------------
    # Timed iteration
    # -----------------------------
    optimizer.zero_grad()

    # Forward (WITH autograd)
    torch.cuda.synchronize()
    t0 = time.time()
    out = model(x)
    torch.cuda.synchronize()
    t1 = time.time()
    forward_time = t1 - t0

    # Backward
    torch.cuda.synchronize()
    t2 = time.time()
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t3 = time.time()
    backward_time = t3 - t2

    total_time = t3 - t0
    throughput = batch_size / total_time
    max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # -----------------------------
    # Write CSV row
    # -----------------------------
    with open(csv_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            1,
            depth,
            hidden_dim,
            batch_size,
            chunks,
            INPUT_DIM,
            NUM_CLASSES,
            forward_time,
            backward_time,
            total_time,
            throughput,
            max_mem,
            loss.item()
        ])

    print(
        f"[1 GPU] depth={depth}, hidden={hidden_dim}, "
        f"batch={batch_size}, chunks={chunks} | "
        f"step={total_time:.4f}s, throughput={throughput:.1f}"
    )

print(f"\n✅ Results saved to: {csv_name}")
