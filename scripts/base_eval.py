"""
Evaluate a model after training.

Run like:
uv run python -m scripts.base_eval --eval-tokens=524288 --run=scaling3_6e18_d16
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=4 -m scripts.base_eval -- --eval-tokens=524288 --run=scaling3_6e18_d16
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for older PyTorch
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels progress bars
import torch
import pickle
import argparse

from nanorepro.dataloader import DataLoader
from nanorepro.loss_eval import evaluate_bpb
from nanorepro.checkpoint import load_model
from nanorepro.common import get_base_path, ddp_init
from nanorepro.core_eval import evaluate_core_metric
BASE_DIR = get_base_path()


def main():

    parser = argparse.ArgumentParser(description="Eval a base model.")
    # Logging
    parser.add_argument('--run', type=str, default=None, help='Run to load')
    # FP8 training
    parser.add_argument('--compute-dtype', type=str, default='bf16', help="Data type for computation, supported: 'bf16', 'fp32').")
    parser.add_argument('--no-fa3', action='store_true', help="Disable Flash Attention 3, for reproducibility.")
    # Training Horizon
    parser.add_argument('--dataset', type=str, default=None, choices=['fineweb', 'climbmix'], help='Training dataset to use (default: use one indicated in checkpoint)')
    # Optimization
    parser.add_argument('--device-batch-size', type=int, default=None, help='Micro batch size per device. (default: None, load from checkpoint)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic settings for reproducibility.')
    # Evaluations
    parser.add_argument('--eval-tokens', type=int, default=80*524288, help='Number of tokens to use for evaluation. (default: 80*524288)')
    parser.add_argument('--core-metric-max-per-task', type=int, default=500, help='Number of examples for core metric evaluation (-1 to use all).')
    args = parser.parse_args()

    # Compute setup and helpers
    device, ddp_master, ddp_world_size = ddp_init()
    print0 = print if os.environ.get("RANK", "0") == "0" else lambda *args, **kwargs: None
    compute_dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[args.compute_dtype]

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
    model, pretrain_metadata = load_model(
        checkpoints_path=checkpoints_path,
        compute_dtype=compute_dtype,
        enable_fa3=not args.no_fa3,
        fp8_training=False,    # doesn't matter
        backward_overlap=False, # useful only in training
        enable_metrics=False,
        device=device,
        step=None)                  # latest checkpoint
    print0("Model configuration:")
    for k, v in model.config.to_dict().items():
        print0(f"  {k:>16}: {v}")

    # Compile
    orig_model = model
    if not args.deterministic:
        model = torch.compile(model, dynamic=False)

    # Hyperparameter Transfer and Calculation
    step = pretrain_metadata["step"]
    pretrain_user_cfg = pretrain_metadata["user_config"]
    max_seq_len = pretrain_user_cfg['max_seq_len']
    micro_batch = args.device_batch_size if args.device_batch_size is not None else pretrain_user_cfg['device_batch_size']
    dataset = args.dataset if args.dataset is not None else pretrain_user_cfg['dataset']

    # Eval Dataloader
    assert args.eval_tokens % (micro_batch * max_seq_len * ddp_world_size) == 0
    eval_steps = args.eval_tokens // (micro_batch * max_seq_len * ddp_world_size)
    eval_loader = DataLoader(
        dataset_or_folderpath=dataset,
        split="val",
        batch_size=micro_batch,
        block_size=max_seq_len,
        tokenizer=tokenizer,
        device=device,
    )
    print0(f"Eval dataloader initialized with dataset {dataset} shards {eval_loader.first_shard} - {eval_loader.last_shard}")
    

    # BPB Evaluation
    bpb, total_nats, total_bytes = evaluate_bpb(model, token_bytes, eval_loader, eval_steps, device)
    print0(f"BPB Eval {step} | BPB {bpb:.14f} | nats {total_nats:.1f} | bytes {total_bytes}")
    
    # CORE Evaluation
    core_metric, core_results_list, core_eval_time = evaluate_core_metric(orig_model, tokenizer, device, args.core_metric_max_per_task)
    print0(f"CORE {step} | core metric {core_metric:.14f} | dt {core_eval_time:.2f}s")
    for res in core_results_list:
        print0(res)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()


