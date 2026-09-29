# Inference performance: a reading guide

The web server has two independent, opt-in decode flags:

| Option | What changes |
| --- | --- |
| `--cuda-graphs` | Record a fixed-shape, one-token forward pass and replay its CUDA work. |
| `--compile-decode` | Compile the model's existing transformer/output regions for one-token decoding. |
| `--compute-dtype=bf16` (default) | Preconvert linear weights to BF16 once, retaining BF16 computation and the selected attention backend. |
| `--compute-dtype=fp16` | Prepare inference weights with squared-ReLU scaling, then use FP16 linear weights and computation. |

Defaults remain eager decoding with BF16 and FlashAttention. Prefill, sampling,
stop decisions, and Python tool execution remain eager with either decode flag.
There are no training CUDA Graph changes.

The server and benchmark automatically preconvert BF16 linear weights after
loading. Previously these were stored in FP32 and cast during every forward
pass. Preconversion avoids repeated casts and reduces weight-memory traffic;
it uses the same rounded weights for BF16 multiplication. No MLP scaling is
needed for BF16. Other parameters retain their existing dtypes.

## Commands

For p15's older GPU, use SDPA and the prepared FP16 path:

```bash
uv run python -m scripts.chat_web --run=d24 --no-fa --compute-dtype=fp16 \
    --compile-decode --cuda-graphs --host=0.0.0.0
```

For one GPU on x399, preserve native BF16 computation and FlashAttention:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.chat_web --run=d24 \
    --compile-decode --cuda-graphs --host=0.0.0.0
```

Remove either decode flag to try it independently. The first eligible request
warms the decoder and, when requested, compiles/captures it. This can take tens
of seconds or longer. A larger context bucket needs another setup. Startup's
"Model loaded" message does not mean decode compilation has finished.

## Read the implementation in steps

The history deliberately separates the prerequisites from the optimizations:

1. **Independent cache storage and stable smear state.** `KVCache` holds one K/V
   allocation per layer. Compiling mutations of slices of one large allocation
   can introduce expensive copies of that shared storage. The previous embedding
   used by smear gets its own buffer, updated in place during decode.
2. **Fixed-shape decode operations.** RoPE reads the position on the GPU.
   The SDPA path updates the cache at that position and masks unused slots and
   tokens outside the attention window. Shapes do not grow with each token.
   The ordinary eager path still trims the cache to the current length.
3. **CUDA Graph runner.** `nanorepro/decode.py` owns the static input, cache,
   smear state, and output buffers. It warms up, captures one forward pass, and
   replays it after copying the next token into the input buffer. Fresh requests
   reset the buffers and copy in an eager prefill. Retained output logits are
   cloned because graph replay overwrites the output storage.
4. **FlashAttention compiler wrapper.** The custom operation declares K/V cache
   mutations and supplies a fake output for compiler shape tracing. It calls
   the same FlashAttention kernel; its purpose is to make that boundary usable
   by `torch.compile`.
5. **Compiled decode.** `compile_layer_regions(..., for_decode=True)` uses the
   model's existing region helpers, also used by `scripts/chat_sft.py`. Decode
   regions require `fullgraph=True`, so graph breaks fail visibly. Compiler-owned
   CUDA Graphs are disabled; `--cuda-graphs` independently controls manual replay.
6. **FP16 preparation.** The separate helper in `nanorepro/inference.py` rescales
   the dense MLP and converts linear weights before compilation/capture.

Follow-up commits tighten compilation, check training backward, repair an
existing streaming test's missing seed, add this guide and the benchmark, and
automatically preconvert BF16 linear weights for inference.
To inspect each delta:

```bash
git log --reverse --oneline a59670a..inference-decode-optimizations
git show <commit>
```

## Why FP16 needs preparation

The dense MLP computes `W_down * relu(W_up * x)^2`. An activation of 271
overflows when squared in FP16, even though the final result can be small.
The preparation divides `W_up` by 16 and multiplies `W_down` by 256:

```text
(256 * W_down) * relu((W_up / 16) * x)^2
    = W_down * relu(W_up * x)^2
```

This identity holds in real arithmetic for these bias-free linear layers.
Floating-point rounding still changes, and the fixed scale is not a guarantee
against overflow for every checkpoint or input. It avoids additional per-token
scaling operations. Other model parameters retain their existing dtypes.

Preparation is idempotent and inference-only. The web server and benchmark
automatically apply it for BF16 and FP16. Direct Python callers can call
`prepare_inference(model)` on an eval model loaded with their chosen
`compute_dtype`, before compilation; FP32 is a no-op. The existing
`prepare_fp16_inference(model)` helper also remains available. Prepared weights are frozen,
and `model.train()` asks you to reload the checkpoint. Checkpoint files are not
changed. Do not save these transformed weights as an ordinary training checkpoint.

## Lifetime and limits

An Engine retains one decoder, growing its capacity in powers of two from 128
up to 4096 tokens, capped by the model's position limit. Capacity is selected
from prompt length plus the requested output budget, not the eventual stop
position. A later shorter request reuses the larger allocation. For SDPA,
attending across unused masked slots costs work, so capacity matters.

The optimized path supports dense eval models on CUDA, one sample at a time,
with metrics disabled. Batches, CPU, MoE, metrics-enabled models, and requests
above the 4096-token optimization limit use eager decoding. Model position
limits still apply. Compilation/capture errors propagate rather than silently
disabling an explicitly requested optimization.

A generator owns the reusable decoder until it finishes or is closed. Close
abandoned `generate_stream()` generators; the web server already does this for
disconnects. Overlapping generators on the same Engine use eager decoding for
the overlapping request. These buffers are tied to the model's weights/device;
create a new Engine after changing the model.

## Reproduce the four experiments

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m dev.benchmark_decode --run=d24
CUDA_VISIBLE_DEVICES=0 uv run python -m dev.benchmark_decode --run=d24 \
    --compute-dtype=fp16 --no-fa
```

Defaults are a 1024-token prompt, 128 decode steps after the first token, and
five measured requests. Each JSON result reports median decode tokens/s,
median time to first token, the individual rates, and setup time. GPU work is
synchronized at timing boundaries. Setup includes a whole warmup request.
The combined mode reuses compilation from the compile-only mode; its setup
time is **not** a cold compile measurement. Compiler disk caches also affect
setup time between runs.

The benchmark also feeds up to 32 identical reference decode tokens through
each optimized path and reports logit differences and top-token agreement.
This avoids confusing sampling divergence with numerical error. Prepared BF16
and prepared FP16 each have their own eager reference; this is not a quality
evaluation comparing FP16 with BF16. Attention kernel choices and compilation
can change rounding, so bitwise equality is not promised.

Small-model tests need no checkpoint:

```bash
uv run python -m unittest dev.tests.test_inference
```

They cover cache/window correctness, graph reuse and growth, cancellation,
fallback, tool tokens, retained logits, FlashAttention mutations, compiled
decode, FP16 overflow prevention, and an ordinary training forward/backward.
GPU-specific tests skip on CPU; FlashAttention tests require Ampere or newer.
