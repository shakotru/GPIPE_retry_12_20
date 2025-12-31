import torch
import torch.nn as nn
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
import time, csv
from datetime import datetime

# ---------------------------
# config
# ---------------------------
DEPTH = 128 #number of layers in the model
HIDDEN_DIM = 4096 #size of the model's hidden layers
BATCH_SIZE = 512 #batch size for training
INPUT_DIM = 1024 #input feature size
NUM_CLASSES = 10 #number of output classes
DEVICES = [torch.device(f"cuda:{i}") for i in range(4)] #list of CUDA devices

NUM_STEPS = 12 #total number of training steps
WARMUP_STEPS = 3 #number of warmup steps (not timed)

# ---------------------------
# adaptive controller params
# ---------------------------
num_stages = len(DEVICES)
MIN_CHUNKS = 1 #minimum number of chunks
MAX_CHUNKS = 4 * num_stages # maximum number of chunks
#INITIAL_CHUNKS = MAX_CHUNKS 
INITIAL_CHUNKS = 1 #initial number of chunks
ADAPT_EVERY = 2 #adapt chunks every N steps
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

model = MLP(DEPTH, HIDDEN_DIM) #create the model
loss_fn = nn.CrossEntropyLoss() #loss function
x = torch.randn(BATCH_SIZE, INPUT_DIM, device=DEVICES[0]) # random input data
y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICES[0]) # random target data

# ---------------------------
# pipeline balance
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
# warmup
# ---------------------------
print("\n[Warmup]")
for _ in range(WARMUP_STEPS):
    optimizer.zero_grad()
    out = gpipe_model(x) # forward pass
    loss = loss_fn(out, y.to(out.device)) # compute loss
    loss.backward() # backward pass
    optimizer.step() # update parameters
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
# adaptive run with probing
# ---------------------------
prev_step_time = None # previous step time for comparison
step_times = [] # list to store step times

print("\n[Adaptive Run with Probing]")
for step in range(NUM_STEPS):
    optimizer.zero_grad()
    for d in DEVICES: torch.cuda.synchronize(d) #synchronize devices
    t0 = time.time()

    out = gpipe_model(x) # forward pass
    loss = loss_fn(out, y.to(out.device)) # compute loss
    loss.backward() # backward pass
    optimizer.step() # update parameters

    for d in DEVICES: torch.cuda.synchronize(d) #synchronize devices
    t1 = time.time()

    step_time = t1 - t0 # calculate step time
    step_times.append(step_time)
    throughput = BATCH_SIZE / step_time # calculate throughput

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
                for d in DEVICES: torch.cuda.synchronize(d) #synchronize devices
                probe_start = time.time()
                out_probe = temp_model(x)
                loss_probe = loss_fn(out_probe, y.to(out_probe.device))
                loss_probe.backward()
                temp_optimizer.step()
                for d in DEVICES: torch.cuda.synchronize(d) #synchronize devices
                probe_time = time.time() - probe_start
                ratio = probe_time / step_time # ratio of probe time to current step time
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
