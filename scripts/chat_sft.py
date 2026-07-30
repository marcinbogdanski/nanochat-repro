import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for older PyTorch
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels progress bars
import json
import time
import pickle
import argparse
import torch
from nanorepro.gpt import GPTConfig, GPTModel
from nanorepro.loss_eval import evaluate_bpb
from nanorepro.checkpoint import get_latest_checkpoint_step, save_checkpoint
from nanorepro.common import get_base_path, ddp_init, wandb_init, FileLogger
from nanorepro.dataloader import DataLoaderSFT
from nanorepro.fp8 import LinearFP8
from nanorepro.tasks import TaskMixture, TaskSmolTalk, TaskMMLU, TaskGSM8K
BASE_DIR = get_base_path()

class DummyOptimizer:
    def __init__(self):
        pass
    def state_dict(self):
        return {}

def main():

    parser = argparse.ArgumentParser(description="Train a GPT model with Muon optimizer.")
    # Logging
    parser.add_argument('--run', type=str, default=None, help='WandB run name (optional).')
    # FP8 training
    parser.add_argument('--compute-dtype', type=str, default='bf16', help="Data type for computation, supported: 'bf16', 'fp32').")
    parser.add_argument('--no-fa3', action='store_true', help="Disable Flash Attention 3, for reproducibility.")
    parser.add_argument('--fp8', type=str, default='auto', choices=['auto', 'true', 'false'], help="Enable FP8 training, eval is always in compute dtype.")
    # Model Architecture
    # inherited from pretrain checkpoint
    # Training Horizon
    parser.add_argument('--num-iterations', type=int, default=-1, help='Maximum number of training steps. Set to -1 to calculate from params.')
    # Optimization
    parser.add_argument('--device-batch-size', type=int, default=None, help='Micro batch size per device. (default: None, load from checkpoint)')
    parser.add_argument('--total-batch-size', type=int, default=None, help='Total batch size across all devices. (default: None, load from checkpoint)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic settings for reproducibility.')
    # Evaluations
    parser.add_argument('--eval-every', type=int, default=250, help='Evaluate every N steps (-1 to disable, apart from 0 and last step).')
    parser.add_argument('--eval-tokens', type=int, default=40*524288, help='Number of tokens to use for evaluation. (default: 80*524288)')
    parser.add_argument('--log-metrics', action='store_true', help='Collect and log detailed tensor metrics. Slows down training.')
    # Data mixture
    parser.add_argument("--mmlu-epochs", type=int, default=3, help="Num MMLU epochs to use (multiple choice questions, default=3)")
    parser.add_argument("--gsm8k-epochs", type=int, default=4, help="Number of GSM8K epochs to use (math and tool use, default=4)")

    args = parser.parse_args()
    user_config = vars(args).copy()

    # Compute setup and helpers
    device, ddp_master, ddp_world_size = ddp_init()
    enable_fp8 = (args.fp8 == "true" or (args.fp8 == "auto" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9))
    print0 = print if os.environ.get("RANK", "0") == "0" else lambda *args, **kwargs: None
    synchronize = lambda: torch.cuda.synchronize() if device.startswith("cuda") else None
    compute_dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[args.compute_dtype]
    wandb_logger = wandb_init(args.run, user_config, ddp_master)
    run_path = os.path.join(BASE_DIR, "runs_sft", args.run if args.run is not None else "default")
    file_logger = FileLogger(run_path)  # dummy on non-master processes
    file_logger.log('user_config', step=None, data=user_config, override=True)  # override=True to initialize empty on all ranks

    # Warnings
    warnings = []
    if args.no_fa3:
        warnings.append("FA3 disabled, which may reduce training speed. Only set this flag if you need reproducibility.")
    if not enable_fp8:
        warnings.append("FP8 training disabled, which may reduce training speed. To enable, set --fp8=true or --fp8=auto on supported hardware.")
    if enable_fp8 and args.compute_dtype == 'fp32':
        warnings.append("Using FP8 training with FP32 compute. This is a valid but may lead to worse performance.")
    if args.log_metrics:
        warnings.append("Detailed tensor metrics logging is enabled, which may slow down training.")
    if args.deterministic:
        warnings.append("Deterministic mode enabled. This will disable torch.compile and some optimizations and *will* reduce training speed.")
    if warnings:
        print0("!" * 120)
        print0("\n".join(warnings))
        print0("!" * 120)

    # Tokenizer
    tok_base_path = os.path.join(BASE_DIR, "tokenizer")
    tokenizer_path = os.path.join(tok_base_path, "tokenizer.pkl")
    tokenizer = pickle.load(open(tokenizer_path, "rb"))
    token_bytes_path = os.path.join(tok_base_path, "token_bytes.pt")
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    print0("Vocabulary size:", tokenizer.n_vocab)

    # Reproducibility
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

    # Model Setup
    run_name = args.run if args.run is not None else "default"
    checkpoints_path = os.path.join(BASE_DIR, "runs", run_name)
    latest_checkpoint_step = get_latest_checkpoint_step(checkpoints_path)
    latest_meta_path = os.path.join(checkpoints_path, f"meta_{latest_checkpoint_step:06d}.json")  # last saved file
    with open(latest_meta_path, "r") as f:
        pretrain_metadata = json.load(f)
    model_config = GPTConfig(**pretrain_metadata["model_config"])
    with torch.device("meta"):
        model = GPTModel(
            model_config,
            compute_dtype=compute_dtype,
            enable_fa3=not args.no_fa3,
            fp8_training=enable_fp8,
            enable_metrics=args.log_metrics,
        )
    model.to_empty(device=device)
    model.init_weights()  # RoPE buffers, rest of weights will be loaded from checkpoint
    print0("Model configuration:")
    for k, v in model.config.to_dict().items():
        print0(f"  {k:>16}: {v}")
    file_logger.log('model_config', step=None, data=model.config.to_dict())

    # Load Model State
    model_path = os.path.join(checkpoints_path, f"model_{latest_checkpoint_step:06d}.pt")
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)

    # Sync across ranks - technically not needed since we seed identically
    if torch.distributed.is_initialized():
        with torch.no_grad():
            for p in model.parameters():
                torch.distributed.broadcast(p, src=0)
            for b in model.buffers():
                torch.distributed.broadcast(b, src=0)

    # FP8 Print
    num_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    num_eligible = sum([m.is_fp8_legal() for m in model.modules() if isinstance(m, LinearFP8)])
    print0(f"Layers eligible for FP8: {num_eligible} / {num_linear}")

    # Compile
    orig_model = model
    if not args.deterministic:
        model = torch.compile(model, dynamic=False)

    # Training Hyperparameters
    pretrain_user_cfg = pretrain_metadata["user_config"]
    pretrain_hyperparm_cfg = pretrain_metadata["training_hyperparameters"]
    total_batch_size = args.total_batch_size if args.total_batch_size is not None else pretrain_hyperparm_cfg['total_batch_size']
    micro_batch = args.device_batch_size if args.device_batch_size is not None else pretrain_user_cfg['device_batch_size']
    max_seq_len = pretrain_user_cfg['max_seq_len']
    assert total_batch_size % (max_seq_len*micro_batch*ddp_world_size) == 0
    grad_accum = total_batch_size // (max_seq_len*micro_batch*ddp_world_size)
    print0(f"Training hyperparameters: micro_batch={micro_batch}, total_batch_size={total_batch_size}, grad_accum={grad_accum}")

    # Optimizers
    optimizers = [DummyOptimizer(), DummyOptimizer()]

    # Steps Related
    flops_per_token = model.estimate_flops_per_token()
    flops_per_iter = flops_per_token * total_batch_size
    print0(f"Estimated FLOPs per token: {flops_per_token:e}")

    # Log calculated hyperparameters
    training_hyperparameters = {
        'total_batch_size': total_batch_size,
        'micro_batch': micro_batch,
        'grad_accum': grad_accum,
        'flops_per_token': flops_per_token,
        'flops_per_iter': flops_per_iter,
    }
    file_logger.log('training_hyperparameters', step=None, data=training_hyperparameters)

    # Train Dataloader
    tasks_train = TaskMixture([
        TaskSmolTalk(split="train"),                                                          # 460K tasks
        *[TaskMMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],  # 100K tasks per epoch
        *[TaskGSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],         #   8K tasks per epoch
    ])
    train_loader = DataLoaderSFT(
        tasks=tasks_train,
        batch_size=micro_batch,
        block_size=max_seq_len,
        tokenizer=tokenizer,
        device=device,
        num_iterations=args.num_iterations,
    )

    # Eval Dataloader
    assert args.eval_tokens % (micro_batch * max_seq_len * ddp_world_size) == 0
    eval_steps = args.eval_tokens // (micro_batch * max_seq_len * ddp_world_size)
    tasks_eval = TaskMixture([
        TaskSmolTalk(split="test"),                        # 24K tasks
        TaskMMLU(subset="all", split="test", stop=5200),   #  5.2K tasks - match training ratio before repetition (whole test set is 14K)
        TaskGSM8K(subset="main", split="test", stop=420),  #  0.42K tasks (whole test set is 1.32K)
    ])
    eval_loader = DataLoaderSFT(
        tasks=tasks_eval,
        batch_size=micro_batch,
        block_size=max_seq_len,
        tokenizer=tokenizer,
        device=device,
        num_iterations=-1,  # run through the whole eval set
    )

    # Training Loop
    step, total_time, smooth_tloss = 0, 0.0, 0.0
    x, y = train_loader.get_batch_bos()
    while True:
        total_flops = step * total_batch_size * flops_per_token

        # Fetch last_step flag and sync across ranks
        last_step = train_loader.last_step
        if torch.distributed.is_initialized():
            last_step_t = torch.tensor(last_step, dtype=torch.int32, device=device)
            torch.distributed.all_reduce(last_step_t, op=torch.distributed.ReduceOp.MAX)
            last_step = bool(last_step_t.item())

        # BPB Evaluation
        # Always eval on step 0 to get a initial baseline
        if args.eval_every > 0 and (step % args.eval_every == 0 or last_step):
            time_start = time.time()
            bpb, total_nats, total_bytes = evaluate_bpb(model, token_bytes, eval_loader, eval_steps, device)
            time_end = time.time()
            print0(f"BPB evaluation took {time_end - time_start:.2f} seconds.")
            print0(f"BPB Eval {step} | BPB {bpb:.14f} | nats {total_nats:.1f} | bytes {total_bytes}")
            wandb_logger.log({'step': step, 'total_training_flops': total_flops, 'total_training_time': total_time, 'val/bpb': bpb})
            bpb_eval_data = {'val/bpb': bpb, 'val/total_nats': total_nats, 'val/total_bytes': total_bytes}
            file_logger.log0('bpb_eval', step, data=bpb_eval_data)

        # Save Model
        if last_step:
            print0("Saving model...")
            loop_vars = {'step': step, 'total_time': total_time, 'smooth_tloss': smooth_tloss}
            checkpoint_md5sum = save_checkpoint(run_path, orig_model, optimizers, train_loader, loop_vars, user_config, training_hyperparameters)
            print0(f"Saved model_{step:06d}.pt with MD5 sum: {checkpoint_md5sum}")
            file_logger.log('save_model', step, {'checkpoint_md5sum': checkpoint_md5sum})

        # Exit Condition
        if last_step:
            print0("Training completed.")
            break

        # Training
        # dummy for now
        print("=" * 50)
        print(f"Step {step}:")
        print(f"Inputs shape: {x.shape}, Targets shape: {y.shape}")
        print(f"Inputs:\n{x}")
        print(f"Targets:\n{y}")
        x, y = train_loader.get_batch_bos()  # Fetch next batch

        # Advance Step
        step += 1

if __name__ == "__main__":
    main()
