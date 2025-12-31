import torch
import torch.nn as nn
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
import time, csv
from datetime import datetime
import itertools

# !!!!!Model setup params!!!!!
DEPTHS = [225] # list of model depths to test
HIDDEN_DIMS = [4096] # size of the model's hidden layers
BATCH_SIZES = [512] # batch sizes to test
CHUNKS = [4] # number of microbatches
INPUT_DIM = 1024  # input feature size
NUM_CLASSES = 10  # number of output classes
devices = [torch.device(f"cuda:{i}") for i in range(4)] # list of CUDA devices

# -----------------------------
# model definition
# -----------------------------
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

# -----------------------------------------
# CSV setup for outputting test results
#------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_name = f"gpipe_4gpu_results_{timestamp}.csv"

with open(csv_name,"w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "gpu_count","depth","hidden_dim","batch_size","chunks", "INPUT_DIM", "NUM_CLASSES",
        "forward_time_s","backward_time_s","total_step_time_s",
        "throughput_samples_per_s","max_memory_gb","loss"
    ])

# -----------------------------
# main experiment loop
# -----------------------------
loss_fn = nn.CrossEntropyLoss()

for depth, hidden_dim, batch_size, chunks in itertools.product(DEPTHS,HIDDEN_DIMS,BATCH_SIZES,CHUNKS):
    model = MLP(depth, hidden_dim) # create the model
    sample = torch.randn(1, INPUT_DIM, device=devices[0]) # sample input for balancing
    balance = balance_by_time(len(devices), model.net, sample) # balance layers across devices
    gpipe_model = GPipe(model.net, balance=balance, devices=devices, chunks=chunks) # wrap model with GPipe

    optimizer = torch.optim.SGD(gpipe_model.parameters(), lr=0.01) # optimizer setup

    # generate random input data
    x = torch.randn(batch_size, INPUT_DIM, device=devices[0])
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=devices[0])

    # warmup runs (not timed)
    for _ in range(3):
        optimizer.zero_grad()
        out = gpipe_model(x)
        y_tmp = y.to(out.device)
        loss = loss_fn(out, y_tmp)
        loss.backward()
        optimizer.step()

    for d in devices:
        torch.cuda.reset_peak_memory_stats(d) # reset memory stats

    # forward timing
    for d in devices: torch.cuda.synchronize(d) # synchronize devices
    t0 = time.time()
    out = gpipe_model(x) # forward pass
    for d in devices: torch.cuda.synchronize(d) # synchronize devices
    t1 = time.time()
    forward_time = t1 - t0

    y_device = y.to(out.device) # move target to last stage device

    # backward timing
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
        writer.writerow([4, depth, hidden_dim, batch_size, chunks, INPUT_DIM, NUM_CLASSES, forward_time, backward_time, total_time, throughput, max_mem, loss.item()])

    print(f"[4 GPU] depth={depth}, hidden={hidden_dim}, batch={batch_size}, chunks={chunks} | step={total_time:.4f}s, throughput={throughput:.1f}")

print(f"\n✅ Results saved to: {csv_name}")
