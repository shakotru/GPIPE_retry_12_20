import torch
import torch.nn as nn
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
import time, csv
from datetime import datetime

# ---------------------------
# config
# ---------------------------
DEPTH = 128 # number of layers in the model
HIDDEN_DIM = 4096 # size of the model's hidden layers
BATCH_SIZE = 512 # batch size for training
INPUT_DIM = 1024 # input feature size
NUM_CLASSES = 10 # number of output classes
DEVICES = [torch.device(f"cuda:{i}") for i in range(2)] # list of CUDA devices

NUM_STEPS = 12 # total number of training steps
WARMUP_STEPS = 3 # number of warmup steps (not timed)

# adaptive chunk settings
num_stages = len(DEVICES)
MIN_CHUNKS = 1 # minimum number of chunks
MAX_CHUNKS = 4 * num_stages # maximum number of chunks
INITIAL_CHUNKS = 1 # initial number of chunks

ADAPT_EVERY = 2 # adapt chunks every N steps
IMPROVE_THRESH = 0.95 # when to detect significant improvement in step time
REGRESS_THRESH = 1.05 # when to detect significant worsening of step time

# ---------------------------
# model
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

model = MLP(DEPTH, HIDDEN_DIM) # create the model
loss_fn = nn.CrossEntropyLoss() # loss function

x = torch.randn(BATCH_SIZE, INPUT_DIM, device=DEVICES[0]) # random input data
y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICES[0]) # random target data

# ---------------------------
# automatic balance for GPipe
# ---------------------------
sample = torch.randn(1, INPUT_DIM, device=DEVICES[0]) # sample input for balancing
balance = balance_by_time(len(DEVICES), model.net, sample) # balance layers across devices

# ---------------------------
# initialize GPipe
# ---------------------------
chunks = INITIAL_CHUNKS # set initial chunks
gpipe_model = GPipe(model.net, balance=balance, devices=DEVICES, chunks=chunks) # wrap model with GPipe
optimizer = torch.optim.SGD(gpipe_model.parameters(), lr=0.01) # optimizer setup

# ---------------------------
# CSV setup
# ---------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_name = f"gpipe_2gpu_adaptive_{timestamp}.csv"
with open(csv_name,"w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "step","depth","hidden_dim","batch_size","chunks",
        "step_time_s","throughput_samples_per_s","loss"
    ])

# ---------------------------
# warmup
# ---------------------------
print("\n[Warmup]")
for _ in range(WARMUP_STEPS):
    optimizer.zero_grad()
    out = gpipe_model(x) # forward pass
    loss = loss_fn(out, y.to(out.device)) # compute loss
    loss.backward() # backward pass
    optimizer.step() # update parameters
for d in DEVICES: torch.cuda.synchronize(d) # synchronize devices

# ---------------------------
# adaptive run
# ---------------------------
prev_step_time = None # previous step time for comparison
step_times = [] # list to store step times

print("\n[Adaptive Run with Probing]")
for step in range(NUM_STEPS):
    optimizer.zero_grad()
    for d in DEVICES: torch.cuda.synchronize(d) # synchronize devices
    t0 = time.time()

    out = gpipe_model(x) # forward pass
    loss = loss_fn(out, y.to(out.device)) # compute loss
    loss.backward() # backward pass
    optimizer.step() # update parameters

    for d in DEVICES: torch.cuda.synchronize(d) # synchronize devices
    t1 = time.time()

    step_time = t1 - t0 # calculate step time
    step_times.append(step_time)
    throughput = BATCH_SIZE / step_time # calculate throughput

    print(f"[Step {step}] depth={DEPTH} hidden={HIDDEN_DIM} batch={BATCH_SIZE} chunks={chunks} | step_time={step_time:.4f}s | throughput={throughput:.1f}")
    
    # Write CSV
    with open(csv_name,"a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow([step, DEPTH, HIDDEN_DIM, BATCH_SIZE, chunks, step_time, throughput, loss.item()])

    # ---------------------------
    # Adaptive controller + probing
    # ---------------------------
    if step % ADAPT_EVERY == 0 and prev_step_time is not None:
        ratio = step_time / prev_step_time # calculate step time ratio

        # Decide new chunks based on step time ratio
        if ratio < IMPROVE_THRESH:
            new_chunks = min(chunks * 2, MAX_CHUNKS) # increase chunks
            decision = "increase"
        elif ratio > REGRESS_THRESH:
            new_chunks = max(chunks // 2, MIN_CHUNKS) # decrease chunks, use floor division
            decision = "decrease"
        else:
            new_chunks = chunks # keep chunks
            decision = "keep"

        print(f"  [Controller] prev={prev_step_time:.4f}s current={step_time:.4f}s ratio={ratio:.3f} → {decision}")

        # Probing smaller/larger chunk values
        probe_candidates = [max(chunks//2, MIN_CHUNKS), min(chunks*2, MAX_CHUNKS)]
        for probe in probe_candidates:
            if probe != chunks:
                gpipe_probe = GPipe(model.net, balance=balance, devices=DEVICES, chunks=probe) # create probe model
                optimizer_probe = torch.optim.SGD(gpipe_probe.parameters(), lr=0.01) # probe optimizer
                optimizer_probe.zero_grad()
                for d in DEVICES: torch.cuda.synchronize(d) # synchronize devices
                t_probe0 = time.time()
                out_probe = gpipe_probe(x) # forward pass
                loss_probe = loss_fn(out_probe, y.to(out_probe.device)) # compute loss
                loss_probe.backward() # backward pass
                optimizer_probe.step() # update parameters
                for d in DEVICES: torch.cuda.synchronize(d) # synchronize devices
                t_probe1 = time.time()
                probe_time = t_probe1 - t_probe0 # calculate probe time
                ratio_probe = probe_time / step_time # calculate probe ratio
                print(f"  [Probe] Testing chunks={probe} | probe_time={probe_time:.4f}s | ratio={ratio_probe:.3f}")
                if probe_time < step_time:
                    new_chunks = probe # update chunks
                    print(f"  [Controller] Switching chunks {chunks} → {new_chunks}")

        # Rebuild GPipe if chunk number changed
        if new_chunks != chunks:
            chunks = new_chunks # update chunks
            gpipe_model = GPipe(model.net, balance=balance, devices=DEVICES, chunks=chunks) # rebuild GPipe
            optimizer = torch.optim.SGD(gpipe_model.parameters(), lr=0.01) # rebuild optimizer

    prev_step_time = step_time # update previous step time

print("\n[Summary] Step times:", ["{:.3f}".format(t) for t in step_times])
print(f"\n✅ CSV saved to: {csv_name}")
