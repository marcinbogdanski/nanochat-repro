import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for older PyTorch
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels progress bars
import time
import pickle
import argparse
import torch
from nanorepro.loss_eval import evaluate_bpb
from nanorepro.checkpoint import save_checkpoint, load_model, load_optimizer_state
from nanorepro.common import get_base_path, ddp_init, collect_provenance, wandb_init, download_file_rank0, FileLogger
from nanorepro.dataloader import DataLoaderSFT
from nanorepro.fp8 import LinearFP8
from nanorepro.tasks import TaskMixture, TaskSmolTalk, TaskMMLU, TaskGSM8K
from nanorepro.tasks import TaskSimpleSpelling, TaskSpellingBee, TaskCustomJSON, TaskArc, TaskHumanEval
from nanorepro.chatcore_eval import evaluate_chatcore_metric
from nanorepro.calculator import CalculatorAndCounter
from nanorepro.engine import Engine
BASE_DIR = get_base_path()

@torch.inference_mode()
def generate_test_samples_sft(orig_model, tokenizer):
    prompts = [
        "What is the capital of France?",
        "What is the chemical symbol of gold?",
        "If yesterday was Friday, then what will tomorrow be?",
        "What is the opposite of hot?",
        "What are the planets of the solar system?",
        "What is your favorite color?",
        "If 5*x + 3 = 13, then what is x?",
    ]

    was_training = orig_model.training
    orig_model.eval()
    try:
        bos_token = tokenizer.encode_single_token('<|bos|>')
        user_start_token = tokenizer.encode_single_token('<|user_start|>')
        user_end_token = tokenizer.encode_single_token('<|user_end|>')
        assistant_start_token = tokenizer.encode_single_token('<|assistant_start|>')
        assistant_end_token = tokenizer.encode_single_token('<|assistant_end|>')
        calculator = CalculatorAndCounter(tokenizer)
        engine = Engine(orig_model, stop_tokens=[assistant_end_token, bos_token], tool_handler=calculator)
        results = []
        for prompt in prompts:
            tokens = [bos_token, user_start_token] + tokenizer.encode(prompt) + [user_end_token, assistant_start_token]
            gen_results = engine.generate(
                tokens,
                num_samples=1,
                max_new_tokens=16,
                temperature=0.0,
                top_k=50,
                seed=42,
            )
            for res in gen_results:
                gen_text = tokenizer.decode(res)
                results.append(gen_text)
        return results
    finally:
        orig_model.train(was_training)

def main():

    parser = argparse.ArgumentParser(description="Train a GPT model with Muon optimizer.")
    # Init
    parser.add_argument('--run', type=str, default="default", help="Current run name (default: 'default').")
    parser.add_argument('--wandb', action='store_true', help="Enable logging to Weights & Biases, uses name from --run.")
    parser.add_argument('--load-optimizer', type=int, default=1, help="Whether to load the optimizer states from base checkpoint (default: 1).")
    # FP8 training
    parser.add_argument('--compute-dtype', type=str, default='bf16', help="Data type for computation, supported: 'bf16', 'fp32').")
    parser.add_argument('--no-fa', action='store_true', help="Disable Flash Attention, for reproducibility.")
    parser.add_argument('--fp8', type=str, default='auto', choices=['auto', 'true', 'false'], help="Enable FP8 training, eval is always in compute dtype.")
    # Model Architecture
    # inherited from pretrain checkpoint
    # Training Horizon
    parser.add_argument('--num-iterations', type=int, default=-1, help='Maximum number of training steps. Set to -1 to calculate from params.')
    # Optimization
    parser.add_argument('--device-batch-size', type=int, default=None, help='Micro batch size per device. (default: None, load from checkpoint)')
    parser.add_argument('--total-batch-size', type=int, default=None, help='Total batch size across all devices. (default: None, load from checkpoint)')
    parser.add_argument('--embedding-lr', type=float, default=None, help='Base learning rate for embedding parameters. (default: None, load from checkpoint)')
    parser.add_argument('--unembedding-lr', type=float, default=None, help='Base learning rate for unembedding parameters. (default: None, load from checkpoint)')
    parser.add_argument('--matrix-lr', type=float, default=None, help='Base learning rate for matrix parameters. (default: None, load from checkpoint)')
    parser.add_argument('--init-lr-frac', type=float, default=0.8, help='Initial LR fraction of base LR')
    parser.add_argument('--warmdown-ratio', type=float, default=0.5, help='Ratio of iterations for LR warmdown')
    parser.add_argument('--final-lr-frac', type=float, default=0.0, help='Final LR fraction of initial LR')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic settings for reproducibility.')
    # Evaluations
    parser.add_argument('--eval-every', type=int, default=200, help='Evaluate every N steps (-1 to disable, apart from 0 and last step).')
    parser.add_argument('--eval-tokens', type=int, default=40*524288, help='Number of tokens to use for evaluation. (default: 40*524288)')
    parser.add_argument("--chatcore-every", type=int, default=200, help="Evaluate core metric every N steps (-1 to disable)")
    parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="Number of examples for ChatCORE categorical tasks (MMLU, ARC, -1 use all)")
    parser.add_argument("--chatcore-max-sample", type=int, default=24, help="Number of examples for ChatCORE generative tasks (GSM8K, HumanEval, -1 use all)")
    parser.add_argument('--sample-every', type=int, default=200, help='Generate samples every N steps (-1 to disable).')
    parser.add_argument('--log-every', type=int, default=1, help='Log training metrics every N steps.')
    parser.add_argument('--log-metrics', action='store_true', help='Collect and log detailed tensor metrics. Slows down training.')
    parser.add_argument('--log-wandb-every', type=int, default=10, help='Log selected training metrics to WandB every N steps.')
    # Data mixture
    parser.add_argument("--mmlu-epochs", type=int, default=3, help="Num MMLU epochs to use (multiple choice questions, default=3)")
    parser.add_argument("--gsm8k-epochs", type=int, default=4, help="Number of GSM8K epochs to use (math and tool use, default=4)")
    parser.add_argument("--data-mixture", type=str, default="core", choices=["core", "ext"], help="'core' is SmolTalk + MMLU + GSM8K, 'ext' adds identity conversations and spelling tasks.")
    # Optimizations
    # default: backward overlap disabled, all transformer layers form a single compiled region, all muon params of particular shape form single communication bucket
    # enable --backward-overlap and both layers-per-compiled-region and muon-params-per-bucket will set to sensible defaults (1 and world_size respectively)
    parser.add_argument('--backward-overlap', action='store_true', help='Overlap distributed optimizer comms with backward pass. When enabling, use --layers-per-compiled-region to control compiled regions split.')
    parser.add_argument('--layers-per-compiled-region', type=int, default=None, help='Number of layers per compiled region. Valid values: -1 (all transformer layers) or positive int (default -1 if backward overlap disabled, 1 otherwise)')
    parser.add_argument('--muon-params-per-bucket', type=int, default=None, help='Number of Muon optimizer parameters per communication bucket. Valid values: -1 (one bucket per param shape) or positive value divisible by world_size. (defaults: -1 if backward overlap disabled, world_size otherwise).')
    args = parser.parse_args()

    # Run Guard
    run_path = os.path.join(BASE_DIR, "runs_sft", args.run)
    print0 = print if os.environ.get("RANK", "0") == "0" else lambda *args, **kwargs: None
    if args.run != "default" and os.path.exists(run_path):
        print0(f"Run path '{run_path}' already exists. Exiting")
        exit(1)

    # DDP Init
    device, ddp_master, ddp_world_size = ddp_init()

    # Resolve user config
    if args.layers_per_compiled_region is None:
        args.layers_per_compiled_region = 1 if args.backward_overlap else -1  # 1 is optimal on my 4x3090 d20; if overlap disabled then -1 to fuse all layers
    if args.muon_params_per_bucket is None:
        args.muon_params_per_bucket = ddp_world_size if args.backward_overlap else -1  # world_size if overlap enabled, otherwise group all params per shape
    user_config = vars(args).copy()

    # Compute setup and helpers
    enable_fp8 = (args.fp8 == "true" or (args.fp8 == "auto" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9))
    synchronize = lambda: torch.cuda.synchronize() if device.startswith("cuda") else None
    compute_dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[args.compute_dtype]
    wandb_logger = wandb_init("nanochat-sft", args.run if args.wandb else None, user_config, ddp_master)
    file_logger = FileLogger(run_path)
    file_logger.log('user_config', step=None, data=user_config)
    file_logger.log('provenance', step=None, data=collect_provenance(run_path))

    # Warnings
    warnings = []
    if args.no_fa:
        warnings.append("FA disabled, which may reduce training speed. Only set this flag if you need reproducibility.")
    if not enable_fp8:
        warnings.append("FP8 training disabled, which may reduce training speed. To enable, set --fp8=true or --fp8=auto on supported hardware.")
    if enable_fp8 and args.compute_dtype == 'fp32':
        warnings.append("Using FP8 training with FP32 compute. This is a valid but may lead to worse performance.")
    if args.backward_overlap and ddp_world_size == 1:
        warnings.append("Backward overlap requires multi-GPU run to have an effect.")
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
        assert args.no_fa, "FA3 can't reliably be set to deterministic mode due to bug in upstream implementation, FA2 not tested"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    # Model Setup
    checkpoints_path = os.path.join(BASE_DIR, "runs", args.run)
    model, pretrain_metadata = load_model(
        checkpoints_path=checkpoints_path,
        compute_dtype=compute_dtype,
        enable_fa=not args.no_fa,
        fp8_training=enable_fp8,
        enable_metrics=args.log_metrics,
        device=device,
        step=None)
    print0("Model configuration:")
    for k, v in model.config.to_dict().items():
        print0(f"  {k:>16}: {v}")
    file_logger.log('model_config', step=None, data=model.config.to_dict())

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
    if not args.deterministic:
        # This by itself does not switch on compiled path yet, just makes it available.
        # To use, pass use_compiled_if_available=True to forward()
        model.compile_layer_regions(layers_per_region=args.layers_per_compiled_region)

    # Hyperparameter Transfer and Calculation
    pretrain_user_cfg = pretrain_metadata["user_config"]
    pretrain_hyperparam_cfg = pretrain_metadata["training_hyperparameters"]
    max_seq_len = pretrain_hyperparam_cfg['max_seq_len']
    embedding_lr = args.embedding_lr if args.embedding_lr is not None else pretrain_user_cfg['embedding_lr']
    unembedding_lr = args.unembedding_lr if args.unembedding_lr is not None else pretrain_user_cfg['unembedding_lr']
    matrix_lr = args.matrix_lr if args.matrix_lr is not None else pretrain_user_cfg['matrix_lr']
    # Batch size
    param_counts: dict = model.number_scaling_params()
    
    total_batch_size = args.total_batch_size if args.total_batch_size is not None else pretrain_hyperparam_cfg['total_batch_size']

    # Grad Accumulation
    device_batch_size = args.device_batch_size if args.device_batch_size is not None else pretrain_hyperparam_cfg['device_batch_size']
    assert total_batch_size % (max_seq_len*device_batch_size*ddp_world_size) == 0
    grad_accum = total_batch_size // (max_seq_len*device_batch_size*ddp_world_size)
    print0(f"Training hyperparameters: micro_batch={device_batch_size}, total_batch_size={total_batch_size}, grad_accum={grad_accum}")

    # Optimizers
    adamw_optim, muon_optim, backward_scheduler = model.setup_optimizer(
        embedding_lr=embedding_lr * args.init_lr_frac,
        matrix_lr=matrix_lr * args.init_lr_frac,
        unembedding_lr=unembedding_lr * args.init_lr_frac,
        scalar_lr=0.5 * args.init_lr_frac,
        router_lr=0.005 * args.init_lr_frac,  # not used unless MoE is enabled
        smear_backout_lr=0.2 * args.init_lr_frac,
        weight_decay=0.0,
        backward_overlap=args.backward_overlap,
        muon_params_per_bucket=args.muon_params_per_bucket,
        enable_metrics=args.log_metrics,
    )
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    checkpoint_step = pretrain_metadata["step"]
    if args.load_optimizer == 1:
        print0(f"Loading optimizer states from checkpoint step {checkpoint_step}")
        optim_path = os.path.join(checkpoints_path, f"optim_{checkpoint_step:06d}_rank{rank:d}.pt")
        adamw_sd, muon_sd = load_optimizer_state(optim_path, device, adamw_optim, muon_optim)
        # Load AdamW
        base_lrs = [group['lr'] for group in adamw_optim.param_groups]
        adamw_optim.load_state_dict(adamw_sd)
        for group, lr in zip(adamw_optim.param_groups, base_lrs):
            group['lr'] = lr
            group['initial_lr'] = lr
        # Load Muon
        base_lrs = [group['lr'] for group in muon_optim.param_groups]
        muon_optim.load_state_dict(muon_sd)
        for group, lr in zip(muon_optim.param_groups, base_lrs):
            group['lr'] = lr
            group['initial_lr'] = lr
    else:
        print0("Skipping loading optimizer states as per --load-optimizer flag")

    # Steps Related
    flops_per_token = model.estimate_flops_per_token()
    flops_per_iter = flops_per_token * total_batch_size
    print0(f"Estimated FLOPs per token: {flops_per_token:e}")

    # Log calculated hyperparameters
    training_hyperparameters = {
        'max_seq_len': max_seq_len,
        'device_batch_size': device_batch_size,
        'total_batch_size': total_batch_size,
        'grad_accum': grad_accum,
        'flops_per_token': flops_per_token,
        'flops_per_iter': flops_per_iter,
    }
    file_logger.log('training_hyperparameters', step=None, data=training_hyperparameters)

    # LR Scheduler
    def get_lr(progress: float):
        if progress <= 1.0 - args.warmdown_ratio:
            return 1.0
        else:
            decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio  # 0..1 inside warmdown
            return (1.0 - decay) * 1.0 + decay * args.final_lr_frac

    # Muon Momentum Scheduler
    def get_muon_momentum(step):
        muon_frac = min(step / 300, 1.0)
        muon_momentum = (1.0 - muon_frac) * 0.85 + muon_frac * 0.95
        return muon_momentum

    # Train Dataloader
    if args.data_mixture == "core":
        tasks_train = TaskMixture([
            TaskSmolTalk(split="train"),                                                          # 460K tasks
            *[TaskMMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],  # 100K tasks per epoch
            *[TaskGSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],         #   8K tasks per epoch
        ])
    elif args.data_mixture == "ext":
        # Script to generate identity_conversations.jsonl is in dev/generate_sft_data.py
        # Here for convenience I'm using one from Nanochat
        url = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
        identity_conversations_filepath = os.path.join(BASE_DIR, "train_bundle", "identity_conversations.jsonl")
        download_file_rank0(identity_conversations_filepath, url)
        tasks_train = TaskMixture([
            TaskSmolTalk(split="train"),                                                          # 460K tasks
            TaskCustomJSON(filepath=identity_conversations_filepath),                             #   1K synthetic
            TaskCustomJSON(filepath=identity_conversations_filepath),                             #   1K synthetic
            *[TaskMMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],  # 100K tasks per epoch
            *[TaskGSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],         #   8K tasks per epoch
            TaskSimpleSpelling(split="train", stop=200000),                                       # 200K tasks
            TaskSpellingBee(split="train", stop=80000),                                           #  80K tasks
        ])
    else:
        raise ValueError(f"Unknown training mixture: {args.data_mixture}")
    train_loader = DataLoaderSFT(
        tasks=tasks_train,
        batch_size=device_batch_size,
        block_size=max_seq_len,
        tokenizer=tokenizer,
        device=device,
    )

    # Eval Dataloader
    assert args.eval_tokens % (device_batch_size * max_seq_len * ddp_world_size) == 0
    eval_steps = args.eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
    tasks_eval = TaskMixture([
        TaskSmolTalk(split="test"),                        # 24K tasks
        TaskMMLU(subset="all", split="test", stop=5200),   #  5.2K tasks - match training ratio before repetition (whole test set is 14K)
        TaskGSM8K(subset="main", split="test", stop=420),  #  0.42K tasks (whole test set is 1.32K)
    ])
    eval_loader = DataLoaderSFT(
        tasks=tasks_eval,
        batch_size=device_batch_size,
        block_size=max_seq_len,
        tokenizer=tokenizer,
        device=device,
    )

    # ChatCORE Eval Tasks
    chatcore_tasks = None
    if args.chatcore_every > 0:
        if args.data_mixture == "core":
            chatcore_tasks = {
                "arc_easy": TaskArc("ARC-Easy", "test"),
                "arc_challenge": TaskArc("ARC-Challenge", "test"),
                "mmlu": TaskMMLU("all", "test"),
                "gsm8k": TaskGSM8K("main", "test"),
                "human_eval": TaskHumanEval("test"),
            }
        elif args.data_mixture == "ext":
            chatcore_tasks = {
                "arc_easy": TaskArc("ARC-Easy", "test"),
                "arc_challenge": TaskArc("ARC-Challenge", "test"),
                "mmlu": TaskMMLU("all", "test"),
                "gsm8k": TaskGSM8K("main", "test"),
                "human_eval": TaskHumanEval("test"),
                "spelling_bee": TaskSpellingBee("test", stop=256),
            }

    # Training Loop
    step, total_time, smooth_tloss, bpb, min_val_bpb = 0, 0.0, 0.0, float('inf'), float('inf')
    x, y = train_loader.get_batch_bos()
    # Progress tracking and stop conditions
    trained_consumed = 0  # data items that were actually used for training
    bpb_eval_data, chatcore_metric_data, train_log_dict = None, None, None
    while True:
        # Sync the furthest 'consumed' across all ranks, so trained_progress and LR schedules are consistent across ranks
        max_trained_consumed = trained_consumed
        if torch.distributed.is_initialized():
            max_trained_consumed_t = torch.tensor(max_trained_consumed, dtype=torch.int64, device=device)
            torch.distributed.all_reduce(max_trained_consumed_t, op=torch.distributed.ReduceOp.MAX)
            max_trained_consumed = max_trained_consumed_t.item()

        # Stop Conditions
        last_step = args.num_iterations > 0 and step >= args.num_iterations
        # This caps training at approximately one epoch even if num_iterations asks for more
        if max_trained_consumed >= len(tasks_train):
            last_step = True
        # Progress Tracking
        if args.num_iterations > 0:
            trained_progress = step / args.num_iterations
        else:
            trained_progress = max_trained_consumed / len(tasks_train)
        total_flops = step * total_batch_size * flops_per_token

        # BPB Evaluation
        # Always eval on step 0 to get a initial baseline
        if args.eval_every > 0 and (step % args.eval_every == 0 or last_step):
            time_start = time.time()
            bpb, total_nats, total_bytes = evaluate_bpb(model, token_bytes, eval_loader, eval_steps, device)
            time_end = time.time()
            min_val_bpb = min(min_val_bpb, bpb)
            print0(f"BPB evaluation took {time_end - time_start:.2f} seconds.")
            print0(f"BPB Eval {step} | BPB {bpb:.14f} | nats {total_nats:.1f} | bytes {total_bytes}")
            wandb_logger.log({'step': step, 'total_training_flops': total_flops, 'total_training_time': total_time, 'val/bpb': bpb, 'val/min_val_bpb': min_val_bpb})
            bpb_eval_data = {'val/bpb': bpb, 'val/total_nats': total_nats, 'val/total_bytes': total_bytes, 'val/min_val_bpb': min_val_bpb}
            file_logger.log0('bpb_eval', step, data=bpb_eval_data)

        # ChatCORE Metric
        if args.chatcore_every > 0 and step > 0 and (step % args.chatcore_every == 0 or last_step):
            chatcore_metric, chatcore_cat, chatcore_gen, chatcore_results_list, chatcore_total_time = evaluate_chatcore_metric(
                tasks_dict=chatcore_tasks,
                model=model,
                tokenizer=tokenizer,
                micro_batch=device_batch_size,
                max_prompt_len=max_seq_len,
                num_samples=1,
                temperature=0.0,
                top_k=50,
                max_new_tokens=512,
                max_problems_cat=args.chatcore_max_cat,
                max_problems_gen=args.chatcore_max_sample,
            )
            print0(f"ChatCORE {step} | chatcore metric {chatcore_metric:.14f} | categorical {chatcore_cat} | generative {chatcore_gen} | dt {chatcore_total_time:.2f}s")
            chatcore_accuracies = {result['task_label']: result['centered_accuracy'] for result in chatcore_results_list}
            wandb_logger.log({'step': step, 'total_training_flops': total_flops, 'chatcore_metric': chatcore_metric,
                              'chatcore_cat': chatcore_cat, 'chatcore_gen': chatcore_gen, 'centered_results': chatcore_accuracies})
            chatcore_metric_data = {'chatcore_metric': chatcore_metric, 'chatcore_cat': chatcore_cat, 'chatcore_gen': chatcore_gen,
                                    'centered_results': chatcore_accuracies, 'chatcore_eval_time': chatcore_total_time}
            file_logger.log0('chatcore_metric', step, data=chatcore_metric_data)

        # Generate
        if ddp_master and args.sample_every > 0 and step > 0 and (step % args.sample_every == 0 or last_step):
            print0("Generating test samples...")
            generated_samples = generate_test_samples_sft(model, tokenizer)
            print0("\n".join(generated_samples))
            file_logger.log0('generate', step, {'generated_samples': generated_samples})

        # Save Model
        if last_step:
            print0("Saving model...")
            loop_vars = {'step': step, 'total_time': total_time, 'smooth_tloss': smooth_tloss, 'val_bpb': bpb, 'min_val_bpb': min_val_bpb}
            checkpoint_md5sum, optim_md5sum = save_checkpoint(run_path, model, [adamw_optim, muon_optim], train_loader, loop_vars, user_config, training_hyperparameters)
            file_logger.log('save_model', step, {'checkpoint_md5sum': checkpoint_md5sum, 'optim_md5sum': optim_md5sum})

        # Exit Condition
        if last_step:
            print0("Training completed.")
            break

        # Training
        model.train()
        synchronize()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        ts = time.time()
        loss_accum = 0.0
        for opt in [adamw_optim, muon_optim]:
            opt.zero_grad()
        fwd_metrics = []  # nested list: n_grad_accum, dict(...)
        for ga_idx in range(grad_accum):
            split_compiled_regions = args.backward_overlap and ddp_world_size > 1 and ga_idx == grad_accum - 1
            _, loss, metrics = model(x, y, return_logits=False, use_compiled_if_available=True, split_compiled_regions=split_compiled_regions)
            fwd_metrics.append(metrics)
            rank_tloss = loss.detach()
            loss = loss / grad_accum
            loss_accum += loss.detach()
            if ga_idx == grad_accum - 1:
                backward_scheduler.backward_overlap_begin()  # no-op if backward overlap is disabled
            loss.backward()
            trained_consumed = train_loader.consumed  # cache to reflect training reality
            x, y = train_loader.get_batch_bos()    # fetch the next batch
        backward_scheduler.backward_overlap_end()  # no-op if backward overlap is disabled

        # LR Scheduler
        lrm = get_lr(trained_progress)
        for opt in [adamw_optim, muon_optim]:
            for group in opt.param_groups:
                group['lr'] = group['initial_lr'] * lrm
        muon_momentum = get_muon_momentum(step)
        for group in muon_optim.param_groups:
            group['momentum'] = muon_momentum

        # Update MoE balancing
        model.update_moe_balancing()
        model.zero_moe_counters()

        # Optimizer Step
        for opt in [adamw_optim, muon_optim]:
            opt.step()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)

        # Sync & Time
        synchronize()
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.startswith("cuda") else 0.0
        dt = (time.time() - ts)
        total_time += dt

        # Logs
        tps = int(total_batch_size / dt)
        pct = trained_progress * 100
        smooth_tloss = 0.9 * smooth_tloss + (1 - 0.9) * rank_tloss.item()
        debiased_smooth_tloss = smooth_tloss / (1 - 0.9**(step+1))
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print0(f"Step {step} ({pct:.2f}%) | "
                f"loss {debiased_smooth_tloss:.16f} {loss_accum.item():.4f} | "
                f"lrm {lrm:.3f} | dt {dt*1e3:.2f}ms | tps {tps:,} | "
                f"mem {max_mem:.3f} GB | time {total_time_str}")
        if step % args.log_wandb_every == 0:
            wandb_logger.log({
                'step': step,
                'total_training_flops': total_flops,
                'total_training_time': total_time,
                'train/loss': debiased_smooth_tloss,
                'train/lrm': lrm,
                'train/dt': dt,
                'train/tok_per_sec': tps,
            })
        if step % args.log_every == 0 or last_step:
            train_log_dict = {
                'step': step,
                'train/train_loss': loss_accum.item(),
                'train/rank_tloss': rank_tloss.item(),
                'train/smooth_rank_tloss': smooth_tloss,
                'train/debiased_smooth_rank_tloss': debiased_smooth_tloss,
                'train/lrm': lrm,
                'train/muon_momentum': muon_momentum,
                'train/muon_weight_decay': 0.0,  # compatibility with pre-training schema
                'other/dt': dt,
                'other/tps': tps,
                'other/max_mem': max_mem,
                'other/progress': trained_progress,
                'other/total_flops': total_flops,
                'other/total_time': total_time,
            }
            # Metrics - super ugly
            if args.log_metrics:
                opt_metrics = {**adamw_optim.get_metrics(), **muon_optim.get_metrics()}
                metrics_list = model.collect_metrics(fwd_metrics, opt_metrics)  # requires grads to still be attached
                train_log_dict['metrics'] = metrics_list
            file_logger.log('train', step, train_log_dict)
    
        # Advance Step
        step += 1

    file_logger.log('run_summary', step=None, data={
        'user_config': user_config,
        'model_config': model.config.to_dict(),
        'param_counts': param_counts,
        'training_hyperparameters': training_hyperparameters,
        'final_bpb_eval': bpb_eval_data,
        'final_chatcore_metric': chatcore_metric_data,
        'final_train_log': train_log_dict,
    })

    wandb_logger.finish()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
