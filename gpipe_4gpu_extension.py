import torch
import torch.nn as nn
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
import time, csv
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
DEPTH = 128
HIDDEN_DIM = 4096
BATCH_SIZE = 512
INPUT_DIM = 1024
NUM_CLASSES = 10
DEVICES = [torch.device(f"cuda:{i}") for i in range(4)]

NUM_STEPS = 12
WARMUP_STEPS = 3

# ---------------------------
# Adaptive controller params
# ---------------------------
num_stages = len(DEVICES)
MIN_CHUNKS = 1
MAX_CHUNKS = 4 * num_stages
#INITIAL_CHUNKS = MAX_CHUNKS
INITIAL_CHUNKS = 1
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
# CSV for logging
# ---------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_name = f"adaptive_gpipe_4gpu_{timestamp}.csv"
with open(csv_name,"w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "step","depth","hidden_dim","batch_size","chunks","step_time_s","throughput_samples_per_s","loss"
    ])

# ---------------------------
# Adaptive run with probing
# ---------------------------
prev_step_time = None
step_times = []

print("\n[Adaptive Run with Probing]")
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

    # Print step info including model params
    print(f"[Step {step}] depth={DEPTH} hidden={HIDDEN_DIM} batch={BATCH_SIZE} chunks={chunks} "
          f"| step_time={step_time:.4f}s | throughput={throughput:.1f}")

    # ---------------------------
    # Probing mechanism
    # ---------------------------
    if step % ADAPT_EVERY == 0 and step > 0:
        for test_chunks in [chunks//2, chunks*2]:
            if MIN_CHUNKS <= test_chunks <= MAX_CHUNKS:
                temp_model = GPipe(model.net, balance=balance, devices=DEVICES, chunks=test_chunks)
                temp_optimizer = torch.optim.SGD(temp_model.parameters(), lr=0.01)
                temp_optimizer.zero_grad()
                for d in DEVICES: torch.cuda.synchronize(d)
                probe_start = time.time()
                out_probe = temp_model(x)
                loss_probe = loss_fn(out_probe, y.to(out_probe.device))
                loss_probe.backward()
                temp_optimizer.step()
                for d in DEVICES: torch.cuda.synchronize(d)
                probe_time = time.time() - probe_start
                ratio = probe_time / step_time
                print(f"  [Probe] Testing chunks={test_chunks} | probe_time={probe_time:.4f}s | ratio={ratio:.3f}")

                if ratio < 1.0:
                    print(f"  [Controller] Switching chunks {chunks} → {test_chunks}")
                    chunks = test_chunks
                    gpipe_model = GPipe(model.net, balance=balance, devices=DEVICES, chunks=chunks)
                    optimizer = torch.optim.SGD(gpipe_model.parameters(), lr=0.01)
                    break

    prev_step_time = step_time

    # ---------------------------
    # Log to CSV
    # ---------------------------
    with open(csv_name,"a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow([step, DEPTH, HIDDEN_DIM, BATCH_SIZE, chunks, step_time, throughput, loss.item()])

print("\n[Summary] Step times:", ["{:.3f}".format(t) for t in step_times])
print(f"\n✅ CSV log saved to: {csv_name}")
