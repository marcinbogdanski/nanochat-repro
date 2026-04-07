import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels progress bars
import json
import time
import math
import torch
import pickle
import argparse
import datasets
datasets.disable_progress_bars()
import wandb
import torch.nn.functional as F
from mynanochat.gpt import GPTConfig, GPTModel
from mynanochat.dataloader import DataLoader
from mynanochat.core_eval import evaluate_core_metric
from mynanochat.loss_eval import evaluate_bpb
from mynanochat.generate import generate_test_samples
from mynanochat.checkpoint import save_checkpoint, load_checkpoint
from mynanochat.fp8 import LinearFP8
from mynanochat.common import ddp_init, wandb_init, FileLogger

def main():

    parser = argparse.ArgumentParser(description="Train a GPT model with Muon optimizer.")
    # Logging
    parser.add_argument('--run', type=str, default=None, help='WandB run name (optional).')
    # FP8 training
    parser.add_argument('--compute-dtype', type=str, default='bf16', help="Data type for computation, supported: 'bf16', 'fp32').")
    parser.add_argument('--no-fa3', action='store_true', help="Disable Flash Attention 3, for reproducibility.")
    parser.add_argument('--fp8', action='store_true', help="Enable FP8 training, eval stays in compute dtype.")
    # Model architecture
    parser.add_argument('--depth', type=int, default=20, help='Number of transformer layers.')
    parser.add_argument('--aspect-ratio', type=int, default=64, help='Total embedding dimension will be depth * aspect_ratio.')
    parser.add_argument('--head-dim', type=int, default=128, help='Head dimension for multi-head attention. Total embedding dimension must be divisible by this.')
    parser.add_argument('--max-seq-len', type=int, default=2048, help='Context length (block size).')
    parser.add_argument('--window-pattern', type=str, default="SSSL", help='Sliding window patter: L=full, S=half context')
    parser.add_argument('--moe', action='store_true', help='Use Mixture of Experts (MoE) layers instead of dense MLPs.')
    # Training horizon
    parser.add_argument('--dataset', type=str, default='climbmix', choices=['fineweb', 'climbmix'], help='Training dataset to use.')
    parser.add_argument('--num-iterations', type=int, default=-1, help='Maximum number of training steps. Set to -1 to calculate from params.')
    parser.add_argument('--target-param-data-ratio', type=float, default=12, help='Calc num-iterations to maintain optimal data:param ratio (Chinchilla etc.). Measured empirically in Nanochat.')
    # Optimization
    parser.add_argument('--device-batch-size', type=int, default=32, help='Micro batch size per device.')
    parser.add_argument('--total-batch-size', type=int, default=-1, help='Total batch size across all devices. (default: -1, auto-calculate)')
    parser.add_argument('--embedding-lr', type=float, default=0.3, help='Base learning rate for embedding parameters.')
    parser.add_argument('--unembedding-lr', type=float, default=0.008, help='Base learning rate for unembedding parameters.')
    parser.add_argument('--weight-decay', type=float, default=0.28, help='Weight decay for Muon optimizer.')
    parser.add_argument('--matrix-lr', type=float, default=0.02, help='Base learning rate for matrix parameters.')
    parser.add_argument('--scalar-lr', type=float, default=0.5, help='Learning rate for scalars: resid_lambas, x0_lambdas.')
    parser.add_argument('--warmup-steps', type=int, default=40, help='Number of steps for LR warmup')
    parser.add_argument('--warmdown-ratio', type=float, default=0.65, help='Ratio of iterations for LR warmdown')
    parser.add_argument('--final-lr-frac', type=float, default=0.05, help='Final LR fraction of initial LR')
    parser.add_argument('--resume', action='store_true', help='Whether to resume from the latest checkpoint if available.')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic settings for reproducibility.')
    # Evaluations
    parser.add_argument('--eval-every', type=int, default=250, help='Evaluate every N steps.')
    parser.add_argument('--eval-tokens', type=int, default=40*524288, help='Number of tokens to use for evaluation.')
    parser.add_argument('--core-metric-every', type=int, default=2000, help='Evaluate core metric every N steps.')
    parser.add_argument('--core-metric-max-per-task', type=int, default=500, help='Number of examples for core metric evaluation.')
    parser.add_argument('--sample-every', type=int, default=1000, help='Generate samples every N steps.')
    parser.add_argument('--save-every', type=int, default=-1, help='Save model every N steps.')
    parser.add_argument('--log-every', type=int, default=1, help='Log training metrics every N steps.')
    parser.add_argument('--log-metrics', action='store_true', help='Collect and log detailed tensor metrics. Slows down training.')
    parser.add_argument('--log-wandb-every', type=int, default=10, help='Log selected training metrics to WandB every N steps.')

    args = parser.parse_args()
    user_config = vars(args).copy()
   
    # Compute setup and helpers
    device, ddp_master, ddp_world_size = ddp_init()
    print0 = print if os.environ.get("RANK", "0") == "0" else lambda *args, **kwargs: None
    synchronize = lambda: torch.cuda.synchronize() if device.startswith("cuda") else None
    compute_dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[args.compute_dtype]
    wandb_logger = wandb_init(args.run, user_config, ddp_master)
    run_path = os.path.join(os.path.dirname(__file__), "../runs/default")
    file_logger = FileLogger(run_path, user_config)  # dummy on non-master processes

    # Warnings
    warnings = []
    if args.no_fa3:
        warnings.append("FA3 disabled, which may reduce training speed and cause non-determinism. Only set this flag if you need reproducibility or are debugging.")
    if args.fp8 and args.compute_dtype == 'fp32':
        warnings.append("Using FP8 training with FP32 compute. This is a valid configuration but may lead to worse performance compared to using BF16 or FP16 compute.")
    if args.log_metrics:
        warnings.append("Detailed tensor metrics logging is enabled, which will slow down training. Only enable this if you need to debug or analyze training dynamics.")
    if args.deterministic:
        warnings.append("Deterministic mode enabled. This will disable certain optimizations and may reduce training speed. Only enable this if you need reproducibility.")
    if args.log_metrics or args.deterministic:
        warnings.append("Torch compile is disabled due to log_metrics or deterministic flags, which may reduce training speed. Only disable compilation if you need reproducibility or detailed metrics.")
    if warnings:
        print0("!" * 120)
        print0("\n".join(warnings))
        print0("!" * 120)

    # Tokenizer
    tok_base_path = os.path.expanduser("~/.cache/nanochat/tokenizer")
    tokenizer_path = os.path.join(tok_base_path, "tokenizer.pkl")
    tokenizer = pickle.load(open(tokenizer_path, "rb"))
    token_bytes_path = os.path.join(tok_base_path, "token_bytes.pt")
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
   
    # Reproducibility
    if args.deterministic:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
    
    # Precision
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")  # uses tf32 instead of fp32 for matmuls

    # Determinism
    # Also need to disable torch.compile for reproducibility
    if args.deterministic:
        assert args.no_fa3, "FA3 can't reliably be set to deterministic mode due to bug in upstream implementation"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    # Mode Setup
    def create_model_meta(depth):
        # Hyperparameters
        vocab_size = tokenizer.n_vocab
        base_dim = depth * args.aspect_ratio
        model_dim = ((base_dim + args.head_dim-1) // args.head_dim) * args.head_dim  # nudge up towards closest multiple of head_dim
        num_heads = model_dim // args.head_dim
        # Model
        model_config = GPTConfig(
            block_size=args.max_seq_len,
            vocab_size=vocab_size,
            n_layer=depth,
            n_head=num_heads,
            n_embd=model_dim,
            window_pattern=args.window_pattern,
            moe_enable=args.moe,
            moe_n_experts=8,
            moe_top_k=2,
        )
        with torch.device('meta'):
            model_meta = GPTModel(
                model_config,
                compute_dtype=compute_dtype,
                enable_fa3=not args.no_fa3,
                fp8_training=args.fp8,
                enable_metrics=args.log_metrics,
            )
        return model_meta
    model = create_model_meta(args.depth)
    model.to_empty(device=device)
    model.init_weights()

    # FP8 Print
    num_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    num_eligible = sum([m.is_fp8_legal() for m in model.modules() if isinstance(m, LinearFP8)])
    print0(f"  Eligible for FP8: {num_eligible}/{num_linear} linear layers")

    orig_model = model
    if not args.deterministic and not args.log_metrics:
        model = torch.compile(model)

    # (1) Scaling laws / transfer recipe
    # - target_param_data_ratio: at fixed FLOPs, sweep model size vs training horizon,
    #   find the compute-optimal tokens/param ratio
    # - choose d12 as the main reference tuning point
    # - at/around d12, sweep batch size
    # - at/around d12, sweep learning-rate-related hyperparameters
    # - sweep weight decay across several depths, fit a transfer rule
    # - then use paper-based / empirical scaling rules to map reference hyperparameters
    #   from d12 to the actual target model
    param_counts: dict = model.number_scaling_params()
    scaling_params = param_counts['active_transformer_matrices'] + param_counts['lm_head']
    target_tokens = int(args.target_param_data_ratio * scaling_params)
    print0(f"Scaling info:")
    print0(f"  Scaling params (matrices + lm_head): {scaling_params:,}")
    print0(f"  Target tokens (scaling_params * target_param_data_ratio): {target_tokens:,}")

    model_d12_ref = create_model_meta(depth=12)
    d12_params_dict = model_d12_ref.number_scaling_params()
    ref_d12_scaling_params = d12_params_dict['active_transformer_matrices'] + d12_params_dict['lm_head']
    print0(f"  Reference d12 scaling params (matrices + lm_head): {ref_d12_scaling_params:,}")
    ref_d12_target_tokens_D_REF = args.target_param_data_ratio * ref_d12_scaling_params
    ref_d12_batch_size_B_REF = 2**19    # 2**19=524288, measured empirically in nanochat for d12

    # (2) Batch size calculation
    if args.total_batch_size > 0:
        total_batch_size = args.total_batch_size
        print0(f"Using user-provided total_batch_size={total_batch_size} without scaling.")
    else:
        # Power Lines paper (Bopt=D^0.383), https://arxiv.org/abs/2505.13738
        target_token_ratio = target_tokens / ref_d12_target_tokens_D_REF
        proposed_batch_size = ref_d12_batch_size_B_REF * target_token_ratio**0.383
        total_batch_size = 2 ** round(math.log2(proposed_batch_size))
        print0(f"Calculated total_batch_size={total_batch_size} based on Power Lines scaling with "
               f"target_token_ratio={target_token_ratio:.2f}. Proposed batch size before rounding: {proposed_batch_size:.2f}")

    # (3) Learning rate scaling
    # SGD - linear is standard
    # AdamW - sqrt scaling is standard (lr_scale = batch_ratio ** 0.5)
    # Muon - blindly use AdamW scaling in our case
    batch_ratio = total_batch_size / ref_d12_batch_size_B_REF
    batch_lr_scale = batch_ratio ** 0.5

    # (4) Weight decay scaling
    # T_epoch framework, https://arxiv.org/abs/2405.13698
    scaled_weight_decay = args.weight_decay * math.sqrt(total_batch_size / ref_d12_batch_size_B_REF) * (ref_d12_target_tokens_D_REF / target_tokens)
    print0(f"Scaled weight decay: {args.weight_decay} -> {scaled_weight_decay}")

    # Training Hyperparameters
    micro_batch = args.device_batch_size
    assert total_batch_size % (args.max_seq_len*micro_batch*ddp_world_size) == 0
    grad_accum = total_batch_size // (args.max_seq_len*micro_batch*ddp_world_size)
    print0(f"Training hyperparameters: micro_batch={micro_batch}, total_batch_size={total_batch_size}, grad_accum={grad_accum}")

    # Optimizers
    # [0] AdamW for embeddings and scalars, [1] Muon for large matrix params
    optimizers = model.setup_optimizer(
        embedding_lr=args.embedding_lr * batch_lr_scale,
        matrix_lr=args.matrix_lr * batch_lr_scale,
        unembedding_lr=args.unembedding_lr * batch_lr_scale,
        scalar_lr=args.scalar_lr * batch_lr_scale,
        weight_decay=scaled_weight_decay
    )

    # Calc Max Steps
    if args.num_iterations > 0:
        max_steps = args.num_iterations
        print0(f"Using user-provided num_iterations={max_steps} without scaling.")
    else:
        max_steps = target_tokens // total_batch_size  # floor the division
        print0(f"Calculated max_steps={max_steps} based on scaling_params * target_param_data_ratio / total_batch_size")

    # WD for Optimizers
    def get_wd(step: int):
        # cosine decay to zero over the course of training
        return scaled_weight_decay * 0.5 * (1.0 + math.cos(math.pi * step / max_steps))

    # LR / Muon Scheduler functions
    def get_lr(step: int):
        warmup_steps = args.warmup_steps
        warmdown_steps = round(args.warmdown_ratio * max_steps)
        if step < warmup_steps:
            return (step+1) / warmup_steps
        if step <= max_steps - warmdown_steps:
            return 1.0
        else:
            progress = (max_steps - step) / warmdown_steps
            return (progress * 1.0) + (1.0 - progress) * args.final_lr_frac

    def get_muon_momentum(step: int):
        warmdown_steps = round(args.warmdown_ratio * max_steps)
        warmdown_start = max_steps - warmdown_steps
        if step < 400:
            # linearly increase momentum from 0.85 to 0.97 over first 400 steps
            muon_frac = step / 400
            muon_momentum = (1.0 - muon_frac) * 0.85 + muon_frac * 0.97
            return muon_momentum
        elif step < max_steps - warmdown_steps:
            # keep momentum at 0.97 during main phase of training
            return 0.97
        else:
            # linearly decrease momentum from 0.97 to 0.90 over warmdown
            progress = (step - warmdown_start) / warmdown_steps
            muon_momentum = (1.0 - progress) * 0.97 + progress * 0.90
            return muon_momentum

    # Train Dataloader
    train_loader = DataLoader(
        dataset_or_folderpath=args.dataset,
        split="train",
        batch_size=micro_batch,
        block_size=args.max_seq_len,
        tokenizer=tokenizer,
    )
    print0(f"Train dataloader initialized with dataset {args.dataset} shards {train_loader.first_shard} - {train_loader.last_shard}")

    # Eval Dataloader
    assert args.eval_tokens % (micro_batch * args.max_seq_len * ddp_world_size) == 0
    eval_steps = args.eval_tokens // (micro_batch * args.max_seq_len * ddp_world_size)
    print0(f"Eval BPB every {args.eval_every} steps, eval_steps={eval_steps}")
    eval_loader = DataLoader(
        dataset_or_folderpath=args.dataset,
        split="val",
        batch_size=micro_batch,
        block_size=args.max_seq_len,
        tokenizer=tokenizer,
    )
    print0(f"Eval dataloader initialized with dataset {args.dataset} shards {eval_loader.first_shard} - {eval_loader.last_shard}")
    
    # Checkpoint Resume
    if args.resume:
        print0("Resuming from latest checkpoint...")
        checkpoint_path = os.path.join(os.path.dirname(__file__), "../runs/default")
        loaded_vars = load_checkpoint(checkpoint_path, model, optimizers, train_loader, device)
        step = loaded_vars["step"]
        total_time = loaded_vars["total_time"]
        smooth_dt = loaded_vars["smooth_dt"]
        smooth_tloss = loaded_vars["smooth_tloss"]
        print0(f"Resumed checkpoint from step {step}")
    else:
        step, total_time, smooth_dt, smooth_tloss = 0, 0.0, 0.0, 0.0

    # Training Loop
    start_step = step
    while True:

        # BPB Evaluation
        # Always eval on step 0 to get memory allocation warmup (helps if GPU mem super tight)
        if step == start_step or (args.eval_every > 0 and (step % args.eval_every == 0 or step == max_steps)):
            bpb, total_nats, total_bytes = evaluate_bpb(model, token_bytes, eval_loader, eval_steps, device)
            print0(f"BPB Eval {step} | BPB {bpb:.14f} | nats {total_nats:.1f} | bytes {total_bytes:.1f}")
            wandb_logger.log({'step': step, 'total_training_time': total_time, 'val/bpb': bpb})
            file_logger.log('bpb_eval', {'step': step, 'val/bpb': bpb, 'val/total_nats': total_nats, 'val/total_bytes': total_bytes})

        # Core Metric
        # Use original model because shapes keep changing
        if args.core_metric_every > 0 and step > start_step and (step % args.core_metric_every == 0 or step == max_steps):
            core_metric, core_accuracies, core_eval_time = evaluate_core_metric(orig_model, tokenizer, device, args.core_metric_max_per_task)
            print0(f"CORE {step} | core metric {core_metric:.14f} | dt {core_eval_time:.2f}s")
            wandb_logger.log({'step': step, 'core_metric': core_metric, 'centered_results': core_accuracies})
            file_logger.log('core_metric', {'step': step, 'core_metric': core_metric, 'centered_results': core_accuracies, 'core_eval_time': core_eval_time})

        # Generate
        if ddp_master and args.sample_every > 0 and step > start_step and (step % args.sample_every == 0 or step == max_steps):
            print0("Generating test samples...")
            generated_samples = generate_test_samples(orig_model, tokenizer, device)
            print0("\n".join(generated_samples))
            file_logger.log('generate', {'step': step, 'generated_samples': generated_samples})

        # Save Model
        if args.save_every > 0 and step > start_step and (step % args.save_every == 0 or step == max_steps):
            print0("Saving model...")
            path = os.path.join(os.path.dirname(__file__), "../runs/default")
            loop_vars = {'step': step, 'total_time': total_time, 'smooth_dt': smooth_dt, 'smooth_tloss': smooth_tloss}
            checkpoint_md5sum = save_checkpoint(path, model, optimizers, train_loader, loop_vars, user_config)
            print0(f"Saved model_{step:06d}.pt with MD5 sum: {checkpoint_md5sum}")
            file_logger.log('save_model', {'step': step, 'checkpoint_md5sum': checkpoint_md5sum})

        # Exit Condition
        if step == max_steps:
            break

        # Training
        model.train()
        synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = time.time()
        loss_accum = 0.0
        for opt in optimizers:
            opt.zero_grad()
        for _ in range(grad_accum):
            x, y = train_loader.get_batch_bos()
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
            loss = loss / grad_accum
            loss_accum += loss.detach()
            loss.backward()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)

        # LR Scheduler
        lrm = get_lr(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = group['initial_lr'] * lrm
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_wd(step)
        for group in optimizers[1].param_groups:  # [0] is AdamW, [1] is Muon
            group['momentum'] = muon_momentum
            group['weight_decay'] = muon_weight_decay

        # Update MoE balancing
        model.update_moe_balancing()
        model.zero_moe_counters()

        # Optimizer Step
        for opt in optimizers:
            opt.step()

        # Sync & Time
        synchronize()
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        dt = (time.time() - ts)
        smooth_dt = 0.9 * smooth_dt + 0.1 * dt
        debiased_smooth_dt = smooth_dt / (1 - 0.9**(step+1))
        total_time += dt

        # Logs
        tps = int(total_batch_size / dt)
        pct = step / max_steps * 100
        smooth_tloss = 0.9 * smooth_tloss + (1 - 0.9) * train_loss.item()
        debiased_smooth_tloss = smooth_tloss / (1 - 0.9**(step+1))
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
        remaining_steps = max_steps - step
        eta_seconds = debiased_smooth_dt * remaining_steps
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        print0(f"Step {step}/{max_steps} ({pct:.2f}%) | "
                f"loss {debiased_smooth_tloss:.16f} {loss_accum.item():.4f} | "
                f"lrm {lrm:.3f} | dt {dt*1e3:.2f}ms {debiased_smooth_dt*1e3:.2f}ms | tps {tps:,} | "
                f"mem {max_mem:.3f} GB | shard {train_loader.shard_idx} | "
                f"time {total_time_str} | eta {eta_str}")
        if step % args.log_wandb_every == 0:
            wandb_logger.log({
                'step': step,
                'total_training_time': total_time,
                'train/loss': debiased_smooth_tloss,
                'train/lrm': lrm,
                'train/dt': dt,
                'train/tok_per_sec': tps,
            })
        if step % args.log_every == 0:
            gpt_metrics_dict = orig_model.collect_metrics()  # requires grads to still be attached
            log_dict = {
                'step': step,
                'train/train_loss': train_loss.item(),
                'train/smooth_train_loss': smooth_tloss,
                'train/debiased_smooth_tloss': debiased_smooth_tloss,
                'train/lrm': lrm,
                'train/muon_momentum': muon_momentum,
                'train/muon_weight_decay': muon_weight_decay,
                'other/dt': dt,
                'other/tps': tps,
                'other/max_mem': max_mem,
                'other/shard_idx': train_loader.shard_idx,
                'other/total_time': total_time,
            }
            log_dict.update(gpt_metrics_dict)
            file_logger.log('train', log_dict)
            orig_model.clear_metrics()  # avoid footguns
        
        # Advance Step
        step += 1

    wandb_logger.finish()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
