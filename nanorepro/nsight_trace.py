"""
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
import os
import torch

_TRACE_ENABLED = os.environ.get("NANOREPRO_TRACE", "").lower() in {"1", "true", "yes", "on"}

def is_trace_enabled():
    return _TRACE_ENABLED

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
    if not _TRACE_ENABLED:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    event = torch.cuda.Event(enable_timing=True)
    with torch.cuda.nvtx.range(f"custom_event rank={rank} {name}"):
        event.record()

def clone_boundary(x, left, right):
    """Wrapper for the custom clone_boundary op that checks if tracing is enabled."""
    if not _TRACE_ENABLED:
        return x
    return _clone_boundary(x, left, right)


@torch.library.custom_op("nanorepro::clone_boundary", mutates_args=())
def _clone_boundary(x: torch.Tensor, left: str | None, right: str | None) -> torch.Tensor:
    """Custom forward pass, opaque to torch.compile. The clone() call ensures proper graph placement and ordering."""
    if left is not None:
        record_event(f"{left}_forward.end")
    out = x.clone()
    if right is not None:
        record_event(f"{right}_forward.begin")
    return out

@torch.library.custom_op("nanorepro::clone_boundary_backward", mutates_args=())
def _clone_boundary_backward(x: torch.Tensor, left: str | None, right: str | None) -> torch.Tensor:
    """Custom backward pass, same concept as the forward op, but reversed."""
    if right is not None:
        record_event(f"{right}_backward.end")
    out = x.clone()
    if left is not None:
        record_event(f"{left}_backward.begin")
    return out

@_clone_boundary.register_fake
def _(x, left, right):
    """Fake implementation needed during graph capture during compilation."""
    return torch.empty_like(x)

@_clone_boundary_backward.register_fake
def _(x, left, right):
    """Fake implementation needed during graph capture during compilation."""
    return torch.empty_like(x)

def _clone_boundary_setup_context(ctx, inputs, output):
    """Setup context for clone_boundary custom op."""
    ctx.left = inputs[1]
    ctx.right = inputs[2]

def _clone_boundary_autograd(ctx, grad):
    """This is called during the backward pass, call custom op to ensure proper placement."""
    return _clone_boundary_backward(grad, ctx.left, ctx.right), None, None  # grad for x, None for left, None for right

_clone_boundary.register_autograd(_clone_boundary_autograd, setup_context=_clone_boundary_setup_context)  # Actually register
