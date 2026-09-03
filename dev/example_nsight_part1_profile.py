"""Part 1 of two-part script to demonstrate gpu-side profiling with Nsight Systems.

Part 1 records a multi-rank trace with events, NVTX ranges, and NCCL activity in single file: `example_nsight.nsys-rep`

Run both parts:
uv run dev/example_nsight.sh
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist



"""
============================================== CUSTOM OP SECTION =======================================================
Custom operations for marking boundaries in compiled regions.

I want to be able to see something like this in the profiler:
RANK 0: [      forward     ][     backward     ][      comms      ][      optimizer     ]
        [ layer1 ][ layer2 ][ layer2 ][ layer1 ]
RANK 1: [      forward     ][     backward     ][      comms      ][      optimizer     ]
        [ layer1 ][ layer2 ][ layer2 ][ layer1 ]

Later, I will implement DDP-like backward overlap, and I want to visually see it in the profiler.

Why this way:
- torch.profiler and record_function() - automatic GPU projections are not reliable in compiled regions
- nsight nvtx.range only - only marks GPU sections launched from current thread, since backward has own thread it doesn't work for us
- nsight events alone - event.record() is a side effect, so it won't work reliably inside compiled regions
- nsight event in opaque custom op (both fwd/bwd) - should work reliably and not get "moved" by the compiler
  + clone() is not free: it introduces a device copy and additional graph ordering dependencies; I recommend testing impact

How it works:
- the functions below define custom operations for both forward and backward passes
- the ops are opaque to the compiler, so events will stay "glued" to the .clone() call

First layer caveat:
Tracking of the backward pass depends on 'x' tensor requiring gradient, otherwise the clone_boundary_backward() will not be called.
In our example this is _not_ true at the start of the first layer, it is true starting from the 1-to-2 layer transition and onwards.
Because of that we manually insert events for the 'layer1_forward.begin' and 'layer1_backward.end'.
Furthermore, because record_event() is not reliable when used inside compiled regions (unless in custom op),
we need to call these additional record_event() from the outside of the compiled regions (before forward() and after backward() ).
"""

def record_event(name):
    """Create a wrapped CUDA event, allowing us to tie a "name" to the event later.

    CPU:    [    NVTX range with name "foo"   ]
    CPU:          [ event id=1234 enqueued ]
                            +----------- same event ID ---------v
    GPU:                                                [ event id=1234 reached ]

    In post-processing we can correlate the NVTX name "foo" with the CPU-side api call for event id=1234.
    Then we can use event id to find the corresponding GPU-side timestamp.
    
    Using this function directly in compiled regions is not reliable.
    """
    event = torch.cuda.Event(enable_timing=True)
    with torch.cuda.nvtx.range(f"custom_event rank={rank} {name}"):
        event.record()

@torch.library.custom_op("example_nsight::clone_boundary", mutates_args=())
def clone_boundary(x: torch.Tensor, left: str | None, right: str | None) -> torch.Tensor:
    """Custom forward pass, opaque to torch.compile. The clone() call ensures proper graph placement and ordering."""
    if left is not None:
        record_event(f"{left}_forward.end")
    out = x.clone()
    if right is not None:
        record_event(f"{right}_forward.begin")
    return out

@torch.library.custom_op("example_nsight::clone_boundary_backward", mutates_args=())
def clone_boundary_backward(x: torch.Tensor, left: str | None, right: str | None) -> torch.Tensor:
    """Custom backward pass, same concept as the forward op, but reversed."""
    if right is not None:
        record_event(f"{right}_backward.end")
    out = x.clone()
    if left is not None:
        record_event(f"{left}_backward.begin")
    return out

@clone_boundary.register_fake
def _(x, left, right):
    """Fake implementation needed during graph capture during compilation."""
    return torch.empty_like(x)

@clone_boundary_backward.register_fake
def _(x, left, right):
    """Fake implementation needed during graph capture during compilation."""
    return torch.empty_like(x)

def clone_boundary_setup_context(ctx, inputs, output):
    """Setup context for clone_boundary custom op."""
    ctx.left = inputs[1]
    ctx.right = inputs[2]

def clone_boundary_autograd(ctx, grad):
    """This is called during the backward pass, call custom op to ensure proper placement."""
    return clone_boundary_backward(grad, ctx.left, ctx.right), None, None  # grad for x, None for left, None for right

clone_boundary.register_autograd(clone_boundary_autograd, setup_context=clone_boundary_setup_context)  # Actually register
"""
============================================== CUSTOM OP SECTION =======================================================
"""


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 8192, bias=False)
        self.linear2 = nn.Linear(8192, 4096, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = clone_boundary(x, left="layer1", right="layer2")  # mark end of layer1 and start of layer2
        x = self.linear2(x)
        x = clone_boundary(x, left="layer2", right=None)      # mark end of layer2
        return x

# Distributed Init
rank = int(os.environ['LOCAL_RANK'])
device = torch.device("cuda", rank)
torch.cuda.set_device(device)
dist.init_process_group(backend='nccl')

if rank == 0:
    print("--------------------------------------------------------------------------------")
    print("                   Part 1: Profiling with Nsight Systems")
    print("--------------------------------------------------------------------------------")

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

x = torch.randn(1024, 4096, device=device)
for step in range(5):

    # Profiler: start
    torch.cuda.synchronize()
    if step == capture_step:
        torch.cuda.profiler.start()

    # Forward
    record_event("forward.begin")
    record_event("layer1_forward.begin")         # mark start of layer1
    with torch.cuda.nvtx.range("forward"):
        out = compiled(x)
        loss = out.square().mean()
    record_event("forward.end")

    # Backward
    record_event("backward.begin")
    with torch.cuda.nvtx.range("backward"):
        loss.backward()
    record_event("layer1_backward.end")          # we reached the end of layer1 in backward
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
