import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for older PyTorch
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels progress bars
import pickle
import argparse
import torch
from nanorepro.common import get_base_path, ddp_init, wandb_init, FileLogger
from nanorepro.dataloader import DataLoaderSFT
from nanorepro.tasks import TaskMixture, TaskSmolTalk, TaskMMLU, TaskGSM8K
BASE_DIR = get_base_path()

def main():

    parser = argparse.ArgumentParser(description="Train a GPT model with Muon optimizer.")
    # Logging
    parser.add_argument('--run', type=str, default=None, help='WandB run name (optional).')
    # FP8 training
    parser.add_argument('--compute-dtype', type=str, default='bf16', help="Data type for computation, supported: 'bf16', 'fp32').")
    parser.add_argument('--no-fa3', action='store_true', help="Disable Flash Attention 3, for reproducibility.")
    parser.add_argument('--fp8', type=str, default='auto', choices=['auto', 'true', 'false'], help="Enable FP8 training, eval is always in compute dtype.")
    # Model Architecture
    parser.add_argument('--max-seq-len', type=int, default=2048, help='Context length (block size).')

    # Training Horizon
    parser.add_argument('--num-iterations', type=int, default=-1, help='Maximum number of training steps. Set to -1 to calculate from params.')
    # Optimization
    parser.add_argument('--device-batch-size', type=int, default=32, help='Micro batch size per device.')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic settings for reproducibility.')
    # Evaluations
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
    # TODO
    # ...

    # Train Dataloader
    tasks_train = TaskMixture([
        TaskSmolTalk(split="train"),
        *[TaskMMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
        *[TaskGSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
    ])
    train_loader = DataLoaderSFT(
        tasks=tasks_train,
        batch_size=args.device_batch_size,
        block_size=args.max_seq_len,
        tokenizer=tokenizer,
        device=device,
        num_iterations=args.num_iterations,
    )

    # Eval Dataloader
    # TODO

    # Training Loop
    step = 0
    x, y = train_loader.get_batch_bos()
    while True:
        done = train_loader.last_step
        # TODO: sync

        if done:
            print0("Training completed.")
            break

        # Training step
        print("=" * 50)
        print(f"Step {step}:")
        print(f"Inputs shape: {x.shape}, Targets shape: {y.shape}")
        print(f"Inputs:\n{x}")
        print(f"Targets:\n{y}")
        x, y = train_loader.get_batch_bos()  # Fetch next batch
        
        break

if __name__ == "__main__":
    main()
