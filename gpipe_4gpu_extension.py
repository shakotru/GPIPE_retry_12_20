import torch
import torch.nn as nn
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
import time, csv
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
DEPTH = 225
HIDDEN_DIM = 4096
BATCH_SIZE = 512
INPUT_DIM = 1024
NUM_CLASSES = 10
DEVICES = [torch.device(f"cuda:{i}") for i in range(4)]

NUM_STEPS = 12
WARMUP_STEPS = 3

INITIAL_CHUNKS = 2
MIN_CHUNKS = 1
MAX_CHUNKS = 16
ADAPT_EVERY = 2
IMPROVE_THRESH = 0.95
REGRESS_THRESH = 1.05

# ---------------------------
# Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self, depth, hidden_dim):
        super().__init__()
        layers = [nn.Linear(INPUT_DIM, hidden_dim), nn.ReLU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(DEPTH, HIDDEN_DIM)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(BATCH_SIZE, INPUT_DIM, device=DEVICES[0])
y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICES[0])

# ---------------------------
# Pipeline balance
# ---------------------------
sample = torch.randn(1, INPUT_DIM, device=DEVICES[0])
balance = balance_by_time(len(DEVICES), model.net, sample)

# ---------------------------
# Initialize GPipe
# ---------------------------
chunks = INITIAL_CHUNKS
gpipe_model = GPipe(model.net, balance=balance, devices=DEVICES, chunks=chunks)
optimizer = torch.optim.SGD(gpipe_model.parameters(), lr=0.01)

# ---------------------------
# Warmup
# ---------------------------
print("\n[Warmup]")
for _ in range(WARMUP_STEPS):
    optimizer.zero_grad()
    out = gpipe_model(x)
    loss = loss_fn(out, y.to(out.device))
    loss.backward()
    optimizer.step()
for d in DEVICES: torch.cuda.synchronize(d)

# ---------------------------
# Adaptive run
# ---------------------------
prev_step_time = None
step_times = []

print("\n[Adaptive Run]")
for step in range(NUM_STEPS):
    optimizer.zero_grad()
    for d in DEVICES: torch.cuda.synchronize(d)
    t0 = time.time()

    out = gpipe_model(x)
    loss = loss_fn(out, y.to(out.device))
    loss.backward()
    optimizer.step()

    for d in DEVICES: torch.cuda.synchronize(d)
    t1 = time.time()

    step_time = t1 - t0
    step_times.append(step_time)
    throughput = BATCH_SIZE / step_time

    print(f"[Step {step}] chunks={chunks} | step_time={step_time:.4f}s | throughput={throughput:.1f} samples/s")

    # ---------------------------
    # Adaptive controller
    # ---------------------------
    if step % ADAPT_EVERY == 0 and prev_step_time is not None:
        ratio = step_time / prev_step_time
        if ratio < IMPROVE_THRESH:
            new_chunks = min(chunks * 2, MAX_CHUNKS)
            decision = "increase"
        elif ratio > REGRESS_THRESH:
            new_chunks = max(chunks // 2, MIN_CHUNKS)
            decision = "decrease"
        else:
            new_chunks = chunks
            decision = "keep"

        print(f"  [Controller] prev={prev_step_time:.4f}s current={step_time:.4f}s ratio={ratio:.3f} → {decision}")

        if new_chunks != chunks:
            print(f"  [Controller] Rebuilding GPipe: {chunks} → {new_chunks}")
            chunks = new_chunks
            gpipe_model = GPipe(model.net, balance=balance, devices=DEVICES, chunks=chunks)
            optimizer = torch.optim.SGD(gpipe_model.parameters(), lr=0.01)

    prev_step_time = step_time

print("\n[Summary] Step times:", ["{:.3f}".format(t) for t in step_times])
