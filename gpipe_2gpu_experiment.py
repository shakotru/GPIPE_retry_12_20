import torch
import torch.nn as nn
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
import time, csv
from datetime import datetime
import itertools

DEPTHS = [8, 16, 32, 64, 128]
HIDDEN_DIMS = [128, 256, 512, 1024]
BATCH_SIZES = [8, 32, 64, 128]
CHUNKS = [1, 2, 4, 8, 16]
INPUT_DIM = 1024
NUM_CLASSES = 10
devices = [torch.device(f"cuda:{i}") for i in range(2)]

class MLP(nn.Module):
    def __init__(self, depth, hidden_dim):
        super().__init__()
        layers = [nn.Linear(INPUT_DIM, hidden_dim), nn.ReLU()]
        for _ in range(depth-2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_name = f"gpipe_2gpu_results_{timestamp}.csv"

with open(csv_name,"w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "gpu_count","depth","hidden_dim","batch_size","chunks", "INPUT_DIM", "NUM_CLASSES",
        "forward_time_s","backward_time_s","total_step_time_s",
        "throughput_samples_per_s","max_memory_gb","loss"
    ])

loss_fn = nn.CrossEntropyLoss()

for depth, hidden_dim, batch_size, chunks in itertools.product(DEPTHS,HIDDEN_DIMS,BATCH_SIZES,CHUNKS):
    model = MLP(depth, hidden_dim)
    sample = torch.randn(1, INPUT_DIM, device=devices[0])
    balance = balance_by_time(len(devices), model.net, sample)
    gpipe_model = GPipe(model.net, balance=balance, devices=devices, chunks=chunks)

    optimizer = torch.optim.SGD(gpipe_model.parameters(), lr=0.01)

    x = torch.randn(batch_size, INPUT_DIM, device=devices[0])
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=devices[0])

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        out = gpipe_model(x)
        y_tmp = y.to(out.device)
        loss = loss_fn(out, y_tmp)
        loss.backward()
        optimizer.step()

    for d in devices:
        torch.cuda.reset_peak_memory_stats(d)

    # Forward timing
    for d in devices: torch.cuda.synchronize(d)
    t0 = time.time()
    out = gpipe_model(x)
    for d in devices: torch.cuda.synchronize(d)
    t1 = time.time()
    forward_time = t1 - t0

    # Move target to last stage device
    y_device = y.to(out.device)

    # Backward timing
    optimizer.zero_grad()
    for d in devices: torch.cuda.synchronize(d)
    t2 = time.time()
    loss = loss_fn(out, y_device)
    loss.backward()
    optimizer.step()
    for d in devices: torch.cuda.synchronize(d)
    t3 = time.time()
    backward_time = t3 - t2

    total_time = t3 - t0
    throughput = batch_size / total_time
    max_mem = max(torch.cuda.max_memory_allocated(d) for d in devices)/(1024**3)

    with open(csv_name,"a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow([2, depth, hidden_dim, batch_size, chunks, INPUT_DIM, NUM_CLASSES, forward_time, backward_time, total_time, throughput, max_mem, loss.item()])

    print(f"[2 GPU] depth={depth}, hidden={hidden_dim}, batch={batch_size}, chunks={chunks} | step={total_time:.4f}s, throughput={throughput:.1f}")

print(f"\n✅ Results saved to: {csv_name}")
