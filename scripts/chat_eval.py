"""
Evaluate a SFT tuned model.

Run like:
uv run python -m scripts.chat_eval --data-mixture=ext --eval-tokens=524288 --run=scaling3_6e18_d16-counting
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- --data-mixture=ext --eval-tokens=524288 --run=scaling3_6e18_d16-counting
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for older PyTorch
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels progress bars
import torch
import pickle
import argparse

from nanorepro.dataloader import DataLoaderSFT
from nanorepro.tasks import TaskMixture, TaskSmolTalk
from nanorepro.tasks import TaskArc, TaskMMLU  # categorical
from nanorepro.tasks import TaskSpellingBee, TaskGSM8K, TaskHumanEval  # generative
from nanorepro.loss_eval import evaluate_bpb
from nanorepro.checkpoint import load_model
from nanorepro.common import get_base_path, ddp_init
from nanorepro.chatcore_eval import evaluate_chatcore_metric
BASE_DIR = get_base_path()


def main():
    parser = argparse.ArgumentParser(description="Eval a SFT model.")
    # Logging
    parser.add_argument('--run', type=str, default=None, help='Run to load')
    # FP8 training
    parser.add_argument('--compute-dtype', type=str, default='bf16', help="Data type for computation, supported: 'bf16', 'fp32').")
    parser.add_argument('--no-fa', action='store_true', help="Disable Flash Attention, for reproducibility.")
    # Optimization
    parser.add_argument('--device-batch-size', type=int, default=None, help='Micro batch size per device. (default: None, load from checkpoint)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic settings for reproducibility.')
    # Evaluations
    parser.add_argument('--eval-tokens', type=int, default=40*524288, help='Number of tokens to use for evaluation. (default: 40*524288)')
    parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="Number of examples for ChatCORE categorical tasks (MMLU, ARC, -1 use all)")
    parser.add_argument("--chatcore-max-sample", type=int, default=24, help="Number of examples for ChatCORE generative tasks (GSM8K, HumanEval, -1 use all)")
    # Data Mixture
    parser.add_argument("--data-mixture", type=str, default=None, choices=["core", "ext"], help="'core' is SmolTalk + MMLU + GSM8K, 'ext' adds identity conversations and spelling tasks.")
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
        assert args.no_fa, "FA3 can't reliably be set to deterministic mode due to bug in upstream implementation, FA2 not tested"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    # Model Setup
    run_name = args.run if args.run is not None else "default"
    checkpoints_path = os.path.join(BASE_DIR, "runs_sft", run_name)
    model, pretrain_metadata = load_model(
        checkpoints_path=checkpoints_path,
        compute_dtype=compute_dtype,
        enable_fa=not args.no_fa,
        fp8_training=False,    # doesn't matter
        enable_metrics=False,
        device=device,
        step=None)                  # latest checkpoint
    print0("Model configuration:")
    for k, v in model.config.to_dict().items():
        print0(f"  {k:>16}: {v}")

    # Compile
    if not args.deterministic:
        # This by itself does not switch on compiled path yet, just makes it available.
        # To use pass use_compiled_if_available=True to forward()
        # For eval only, splits are not useful; -1 compiles all layers together
        model.compile_layer_regions(layers_per_region=-1)

    # Hyperparameter Transfer and Calculation
    step = pretrain_metadata["step"]
    pretrain_user_cfg = pretrain_metadata["user_config"]
    pretrain_hyperparam_cfg = pretrain_metadata["training_hyperparameters"]
    max_seq_len = pretrain_hyperparam_cfg['max_seq_len']
    micro_batch = args.device_batch_size if args.device_batch_size is not None else pretrain_hyperparam_cfg['device_batch_size']
    data_mixture = args.data_mixture if args.data_mixture is not None else pretrain_user_cfg['data_mixture']

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
    )

    # BPB Evaluation
    bpb, total_nats, total_bytes = evaluate_bpb(model, token_bytes, eval_loader, eval_steps, device)
    print0(f"BPB Eval {step} | BPB {bpb:.14f} | nats {total_nats:.1f} | bytes {total_bytes}")

    # ChatCORE Evaluation
    if data_mixture == "core":
        tasks_dict = {
            "arc_easy": TaskArc("ARC-Easy", "test"),
            "arc_challenge": TaskArc("ARC-Challenge", "test"),
            "mmlu": TaskMMLU("all", "test"),
            "gsm8k": TaskGSM8K("main", "test"),
            "human_eval": TaskHumanEval("test"),
        }
    elif data_mixture == "ext":
        tasks_dict = {
            "arc_easy": TaskArc("ARC-Easy", "test"),
            "arc_challenge": TaskArc("ARC-Challenge", "test"),
            "mmlu": TaskMMLU("all", "test"),
            "gsm8k": TaskGSM8K("main", "test"),
            "human_eval": TaskHumanEval("test"),
            "spelling_bee": TaskSpellingBee("test", stop=256),
        }
    else:
        raise ValueError(f"Unknown data mixture: {data_mixture}")
    chatcore_metric, chatcore_cat, chatcore_gen, chatcore_results_list, chatcore_total_time = evaluate_chatcore_metric(
        tasks_dict=tasks_dict,
        model=model,
        tokenizer=tokenizer,
        micro_batch=micro_batch,
        max_prompt_len=max_seq_len,
        num_samples=1,
        temperature=0.0,
        top_k=50,
        max_new_tokens=512,
        max_problems_cat=args.chatcore_max_cat,
        max_problems_gen=args.chatcore_max_sample,
    )
    print0(f"ChatCORE {step} | chatcore metric {chatcore_metric:.14f} | categorical {chatcore_cat} | generative {chatcore_gen} | dt {chatcore_total_time:.2f}s")
    for res in chatcore_results_list:
        print0(res)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()


