# Assorted Development Notes

## 2026.06.19 - FA3 community kernel

Training default d12 model, with BPB eval every 250 steps, causes bumps in `plot17_value_embed.weight_update_ratio` plots.

- 8xH100 with `varunneal/flash-attention-3` - clean, no bumps
- 2x3090 with `kernels-community/flash-attn3` - yes, bumps exits
- 2x3090 with SDPA - clean, no bumps

Two runs with same params on same 2x3090 system produce bumps/no-bumps depending if FA3 is enabled.

Me, Claude and Codex inspected the code, and found not other paths through with eval could affect subsequent training code.

Since dissabling FA3 or changing kernel removes the issue, I am inclined to tenatively put it as issue in `kernels-community/flash-attn3`

Would be cool to investigate further at some point.

## 2026.06.18 - Post-Block Residual RMS magnitude

The magnitude of Post-Block Residual RMS differs significantly between:

- `scaling4_1e18_d12`: 2 ranks, device_batch_size=16, grad_accum=8, residual max at step 100: ~49
- `scaling3_1e18_d12`: 8 ranks, device_batch_size=32, grad_accum=1, fp8=auto/H100, residual max at step 100: ~101

Why? Codex/Claude did code review and batching/grad-accum seems ok, they say it's because weights diverge between runs.

I'm not satisfied with that explanation. Since BPB eval during runs overlap, i'm leaving it for now.

## 2026.04.02 - MoE

MoE path is not optimized - inspect the compiler graph and benchmark.

## 2026.03.28 - FP8

On some runs `--fp8` causes loss to be null. This matches Nanochat behavior, but would be investigated.
