# Assorted Development Notes

## 2026.07.25 - Bug: seed/param init divergent across ranks

In this repo parameter initialization across ranks was dependent on identical seeding. The problem was that `manual_seed(42)` calls were behind `--deterministic` argument only:

```python
if args.deterministic:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
```

This means that in normal runs before `cd191a7` (including Scaling Laws runs), parameters were initialized with different RNG across the ranks. This had following effects:

- params with `is_small=False` (`wte`, `lm_head`, `value_embeds` and all Muon params) - were initialized differently on each rank, and only synced at the end of the first optimizer pass
- small params (`resid_lambdas`, `x0_lambdas`, `smear_lambda`, `backout_lambda`) - are initialized to constant values and not affected
- single parameter `smear_gate.weight` - was initialized differently across ranks and never synced

The bug has few potentially detrimental effects:

**Runs were unseeded** while intention was to use same seed. Our Scaling Laws experiments repeated each config n=1 times. As each config has different model shape, and resulting matrix shapes affect random init even on same seed. This means each of our Scaling Laws runs would have had effectively differently randomly initialized model, even on same seed. In context of Scaling Laws I'm going to hand wave this.

**1st step cross-rank divergence and sync** - on first step params across ranks were different, so grads had higher variance. In theory this could introduce "shock" to the network on 1st step, which may or may not have had an effect. Given amount of noise present in normal training and initial warmup LR at 1/40 I don't see this 1st-step-only issue having a significant effect.

**Through-whole-run divergent `smear_gate`** - this param had different init across ranks and was never synced. Notably each rank shared same optimizer update for `smear_gate` across ranks through whole training. So divergence was limited to initialization `uniform_(self.smear_gate.weight, 0.0, 0.02)` and stayed constant during training (WD=0), but was never removed through the entire run. The impact was damped by a) small initialization, and b) multiplication by `smear_lambda` which is learned and is initialized to zero. Still, this was present to the very end, through final annealing. This is potentially a problem.

To isolate effect of `smear_gate` divergence, I did few runs at d12, ~110M scaling params, 2e17 FLOPs, 502 steps, batch 2^19, local 4x3090. Two sets of runs were performed, 9 independent runs each:

- set `A` - post fix, same seed across ranks, all params including `smear_gate` identical init across ranks. The result variance is only due to numerical non-determinism of the run
- set `B` - post-fix, same seed across ranks, all params excluding `smear_gate` identical init, `smear_gate` initialized independently on each rank with `uniform_(self.smear_gate.weight, 0.0, 0.02)`. The result variance is effect of `smear_gate` divergence and run numerical non-determinism

Result:

| Set             | n | mean BPB |  std     |
|-----------------|---|----------|----------|
| A (fixed)       | 9 |  0.97845 |  0.00020 |
| B (divergent)   | 9 |  0.97842 |  0.00026 |
| A-B             | - |  0.00002 |        - |

The mean difference is on the order of 10x smaller than either A or B spread. It seems the `smear_gate` has no measurable effect in this case.

## 2026.07.11 - Muon static buffers

Implemented static param/grad buffers for Muon parameters. This avoids creating temporary buffers in DistMuon.step() and saves a copy. Parameter .data/.grad fields become pointers to static buffers. Reduce-scatter and all-gather operate on the buffers directly. This introduces coupling between model and optimizer.

Tests on 4x3090, depth=20:

| Configuration        | Micro-batch | Grad accum | Peak memory | Tok/s  | Speedup |
|----------------------|------------:|-----------:|------------:|-------:|--------:|
| Before               |           6 |         21 |  21.922 GiB | 64,587 |       — |
| Static, same config  |           6 |         21 |  19.725 GiB | 64,633 |  +0.07% |
| Static, larger batch |           7 |         18 |  21.676 GiB | 64,825 |  +0.37% |

It ain't much, but it's honest work.

## 2026.06.19 - FA3 community kernel

Training default d12 model, with BPB eval every 250 steps, causes bumps in `plot17_value_embed.weight_update_ratio` plots.

- 8xH100 with `varunneal/flash-attention-3` - clean, no bumps
- 2x3090 with `kernels-community/flash-attn3` - yes, bumps exits
- 2x3090 with SDPA - clean, no bumps

Two runs with same params on same 2x3090 system produce bumps/no-bumps depending if FA3 is enabled.

Me, Claude and Codex inspected the code, and found not other paths through with eval could affect subsequent training code.

Since disabling FA3 or changing kernel removes the issue, I am inclined to tentatively put it as issue in `kernels-community/flash-attn3`

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
