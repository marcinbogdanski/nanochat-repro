"""Part 1 of two-part script to demonstrate gpu-side profiling with Nsight Systems.

Part 1 records a multi-rank trace with events, NVTX ranges, and NCCL activity in single file: `example_nsight.nsys-rep`

Run both parts:
uv run dev/example_nsight.sh
"""
print("--------------------------------------------------------------------------------")
print("                   Part 1: Profiling with Nsight Systems")
print("--------------------------------------------------------------------------------")
import os
import torch
import torch.nn as nn
import torch.distributed as dist

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 8192, bias=False)
        self.linear2 = nn.Linear(8192, 4096, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

# Distributed Init
rank = int(os.environ['LOCAL_RANK'])
device = torch.device("cuda", rank)
torch.cuda.set_device(device)
dist.init_process_group(backend='nccl')

# Reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Model and Optimizer
model = SmallModel().to(device)
compiled = torch.compile(model)
with torch.no_grad():
    for p in model.parameters():
        dist.broadcast(p, src=0)
model.train()
lr = 0.01
capture_step = 4

def record_event(name):
    event = torch.cuda.Event(enable_timing=True)
    # NVTX records range on CPU side, then in Nsight one needs to correlate CPU-side API call with GPU-side events
    with torch.cuda.nvtx.range(f"custom_event rank={rank} {name}"):
        event.record()

x = torch.randn(1024, 4096, device=device)
for step in range(5):

    # Profiler: start
    torch.cuda.synchronize()
    if step == capture_step:
        torch.cuda.profiler.start()

    # Forward
    record_event("forward.begin")
    with torch.cuda.nvtx.range("forward"):
        out = compiled(x)
        loss = out.square().mean()
    record_event("forward.end")

    # Backward
    record_event("backward.begin")
    with torch.cuda.nvtx.range("backward"):
        loss.backward()
    record_event("backward.end")

    # Comms
    record_event("comms.begin")
    with torch.cuda.nvtx.range("comms"):
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    record_event("comms.end")

    # Optimizer Step
    record_event("optimizer.begin")
    with torch.cuda.nvtx.range("optimizer_step"):
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
        model.zero_grad()
    record_event("optimizer.end")

    # Profiler: stop
    torch.cuda.synchronize()
    if step == capture_step:
        torch.cuda.profiler.stop()

    # Print
    if rank == 0:
        print(f"Iteration {step}, Loss: {loss.item()}")

dist.destroy_process_group()
