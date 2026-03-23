import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels prograss bars
import json
import time
import math
import torch
import pickle
import argparse
import datasets
datasets.disable_progress_bars()
import wandb
from contextlib import nullcontext
import torch.nn.functional as F
from mynanochat.gpt import GPTConfig, GPTModel
from mynanochat.dataloader import DataLoader
from mynanochat.adamw import DistAdamW
from mynanochat.muon import Muon, DistMuon
from mynanochat.core_eval import evaluate_core_metric

class WandBDummy:
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

@torch.inference_mode()
def sample_one_token(logits, temperature=1.0, top_k=None, sample_rng=None):
    assert logits.ndim == 2  # B,C
    assert temperature >= 0.0
    assert top_k is None or 0 < top_k <= logits.size(-1)
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)  # greedy
    if top_k is None:
        probs = F.softmax(logits / temperature, dim=-1)  # B,C
        ix = torch.multinomial(probs, num_samples=1, generator=sample_rng)  # B,1
        return ix
    else:
        topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)  # B,k
        probs = F.softmax(topk_logits / temperature, dim=-1)  # B,k
        ix = torch.multinomial(probs, num_samples=1, generator=sample_rng)  # B,1
        return torch.gather(topk_indices, -1, ix)  # B,1

@torch.inference_mode()
def generate(model, autocast_ctx, idx, max_new_tokens, temperature=0.0, top_k=None, sample_rng=None):
    """Generate max_tokens starting from idx[B,T]"""
    assert isinstance(idx, torch.Tensor)
    assert idx.dtype == torch.long
    assert len(idx.shape) == 2  # B,T
    assert isinstance(max_new_tokens, int)
    
    is_training = model.training
    model.eval()

    block_size = model.config.block_size
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_tail = idx[:, -block_size:]      # B,T  sliding window
            with autocast_ctx:
                logits, _ = model(idx_tail)      # B,T,C <- B,T
            logits = logits[:, -1, :]            # B,C <- B,T,C  discard all but last
            xcol = sample_one_token(logits, temperature=temperature, top_k=top_k, sample_rng=sample_rng)  # B,1
            idx = torch.cat((idx, xcol), dim=1)  # B,T+1  append
    
    model.train(is_training)
    return idx


def main():

    parser = argparse.ArgumentParser(description="Train a GPT model with Muon optimizer.")
    # WandB
    parser.add_argument('--run', type=str, default=None, help='WandB run name (optional).')
    # Model
    parser.add_argument('--depth', type=int, default=20, help='Number of transformer layers.')
    parser.add_argument('--aspect-ratio', type=int, default=64, help='Total embedding dimension will be depth * aspect_ratio.')
    parser.add_argument('--head-dim', type=int, default=128, help='Head dimension for multi-head attention. Total embedding dimension must be divisible by this.')
    parser.add_argument('--max-seq-len', type=int, default=2048, help='Context length (block size).')
    parser.add_argument('--window-pattern', type=str, default="SSSL", help='Sliding window patter: L=full, S=half context')
    # Training horizon
    parser.add_argument('--num-iterations', type=int, default=-1, help='Maximum number of training steps. Set to -1 to calculate from params.')
    parser.add_argument('--target-param-data-ratio', type=float, default=10.5, help='Calc num-iterations to maintain optimal data:param ratio (Chinchilla etc.). Measured empirically in Nanochat.')
    # Optimization
    parser.add_argument('--device-batch-size', type=int, default=32, help='Micro batch size per device.')
    parser.add_argument('--total-batch-size', type=int, default=-1, help='Total batch size across all devices. (default: -1, auto-calculate)')
    parser.add_argument('--embedding-lr', type=float, default=0.3, help='Base learning rate for embedding parameters.')
    parser.add_argument('--unembedding-lr', type=float, default=0.004, help='Base learning rate for unembedding parameters.')
    parser.add_argument('--weight-decay', type=float, default=0.2, help='Weight decay for Muon optimizer.')
    parser.add_argument('--matrix-lr', type=float, default=0.02, help='Base learning rate for matrix parameters.')
    parser.add_argument('--scalar-lr', type=float, default=0.5, help='Learning rate for scalars: resid_lambas, x0_lambdas.')
    parser.add_argument('--adam_beta1', type=float, default=0.8, help='Beta 1 for AdamW optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.95, help='Beta 2 for AdamW optimizer.')
    parser.add_argument('--warmup-ratio', type=float, default=0.0, help='Ratio of iteratioins for LR warmup')
    parser.add_argument('--warmdown-ratio', type=float, default=0.5, help='Ratio of iteratioins for LR warmdown')
    parser.add_argument('--final-lr-frac', type=float, default=0.0, help='Final LR fraction of initial LR')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic settings for reproducibility.')
    # Evaluations
    parser.add_argument('--eval-every', type=int, default=250, help='Evaluate every N steps.')
    parser.add_argument('--eval-tokens', type=int, default=40*524288, help='Number of tokens to use for evaluation.')
    parser.add_argument('--core-metric-every', type=int, default=2000, help='Evaluate core metric every N steps.')
    parser.add_argument('--core-metric-max-per-task', type=int, default=500, help='Number of examples for core metric evaluation.')
    parser.add_argument('--sample-every', type=int, default=1000, help='Generate samples every N steps.')
    parser.add_argument('--save-every', type=int, default=-1, help='Save model every N steps.')
    parser.add_argument('--log-every', type=int, default=1, help='Log training metrics every N steps.')
    args = parser.parse_args()
    user_config = vars(args).copy()

    # DDP Init
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this ddp run?
    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        ddp_master = ddp_rank == 0  # is this a master?
        device = f'cuda:{ddp_local_rank}'
        device_type = 'cuda'
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend='nccl', device_id=ddp_local_rank)  # device_id= to suppress barrier warning
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        ddp_master = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_type = device
    
    if ddp_master:
        print(f"Init: {ddp=} {ddp_rank=}, {ddp_local_rank=}, {ddp_world_size=}, {ddp_master=}, {device=}")

    # WandB Init
    if args.run is not None and ddp_master:
        wandb_logger = wandb.init(project="nanochat", name=args.run, config=user_config)
    else:
        wandb_logger = WandBDummy()
    
    # Autocast Context
    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == 'cuda' else nullcontext()

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
    if device_type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32" # uses tf32 instead of fp32 for matmuls

    ################################ EQUIVALENCE ###############################
    # Dissable TORCH.COMPILE for reproducibility non-DDP/DDP
    if args.deterministic:
        assert args.window_pattern == 'L', "In deterministinc mode window_pattern must be 'L' due to lack of support in upstream library"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        # torch.backends.cuda.enable_flash_sdp(False)
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch.backends.cuda.enable_math_sdp(True)
    ############################################################################

    # Hyperparameters
    vocab_size = tokenizer.n_vocab
    depth = args.depth
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim-1) // args.head_dim) * args.head_dim  # nudge up towards closest multiple of head_dim
    num_heads = model_dim // args.head_dim

    # Model
    block_size = args.max_seq_len
    model_config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    model = GPTModel(model_config)
    model.to(device)
    model.init_weights()
    orig_model = model
    if not args.deterministic:
        model = torch.compile(model)
    if ddp_master:
        print("Init: Model info:")
        print(f"  {block_size=}")
        print(f"  {vocab_size=}")
        print(f"  {depth=}")
        print(f"  {model_dim=}")
        print(f"  {num_heads=}")

    # (1) Scaling laws / transfer recipe
    # - target_param_data_ratio: at fixed FLOPs, sweep model size vs training horizon,
    #   find the compute-optimal tokens/param ratio
    # - choose d12 as the main reference tuning point
    # - at/around d12, sweep batch size
    # - at/around d12, sweep learning-rate-related hyperparameters
    # - sweep weight decay across several depths, fit a transfer rule
    # - then use paper-based / empirical scaling rules to map reference hyperparams
    #   from d12 to the actual target model
    param_counts: dict = model.number_scaling_params()
    scaling_params = param_counts['transformer_matrices'] + param_counts['lm_head']
    target_tokens = int(args.target_param_data_ratio * scaling_params)
    if ddp_master:
        print(f"Init: Scaling info:")
        print(f"  Scaling params (matrices + lm_head): {scaling_params:,}")
        print(f"  Target tokens (scaling_params * target_param_data_ratio): {target_tokens:,}")

    ref_d12_scaling_params = 135267456  # transformer_matrices + lm_head for d12 model, from nanochat
    ref_d12_target_tokens_D_REF = args.target_param_data_ratio * ref_d12_scaling_params
    ref_d12_batch_size_B_REF = 2**19    # 2**19=524288, measured empirically in nanochat for d12

    # (2) Batch size calculation
    if args.total_batch_size > 0:
        total_batch_size = args.total_batch_size
        if ddp_master:
            print(f"Init: Using user-provided total_batch_size={total_batch_size} without scaling.")
    else:
        # Power Lines paper (Bopt=D^0.383), https://arxiv.org/abs/2505.13738
        target_token_ratio = target_tokens / ref_d12_target_tokens_D_REF
        proposed_batch_size = ref_d12_batch_size_B_REF * target_token_ratio**0.383
        total_batch_size = 2 ** round(math.log2(proposed_batch_size))
        if ddp_master:
            print(f"Init: Calculated total_batch_size={total_batch_size} based on Power Lines scaling with target_token_ratio={target_token_ratio:.2f}. Proposed batch size before rounding: {proposed_batch_size:.2f}")

    # (3) Learning rate scaling
    # SGD - linear is standard
    # AdamW - sqrt scaling is standard (lr_scale = batch_ratio ** 0.5)
    # Muon - blindly use AdamW scaling in our case
    batch_ratio = total_batch_size / ref_d12_batch_size_B_REF
    batch_lr_scale = batch_ratio ** 0.5

    # (4) Weight decay scaling
    # T_epoch framework, https://arxiv.org/abs/2405.13698
    scaled_weight_decay = args.weight_decay * math.sqrt(total_batch_size / ref_d12_batch_size_B_REF) * (ref_d12_target_tokens_D_REF / target_tokens)
    if ddp_master:
        print(f"Init: Scaled weight decay: {args.weight_decay} -> {scaled_weight_decay}")

    # Training Hyperparameters
    micro_batch = args.device_batch_size
    assert total_batch_size % (block_size*micro_batch*ddp_world_size) == 0
    grad_accum = total_batch_size // (block_size*micro_batch*ddp_world_size)
    if ddp_master:
        print(f"Init: Training hyperparameters:")
        print(f"  {micro_batch=}")
        print(f"  {total_batch_size=}")
        print(f"  {grad_accum=}")

    # Optimizers
    params_matrix = list(model.transformer.h.parameters())
    params_embedding = list(model.transformer.wte.parameters())
    params_val_embds = list(model.value_embeds.parameters())
    params_lm_head = list(model.lm_head.parameters())
    params_resid = [model.resid_lambdas]
    params_x0 = [model.x0_lambdas]
    assert len(list(model.parameters())) == len(params_matrix) + len(params_embedding) + len(params_val_embds) + len(params_lm_head) + len(params_resid) + len(params_x0)

    unembedding_lr = args.unembedding_lr * batch_lr_scale
    embedding_lr = args.embedding_lr * batch_lr_scale
    matrix_lr = args.matrix_lr * batch_lr_scale
    adam_betas = (args.adam_beta1, args.adam_beta2)
    scalar_lr = args.scalar_lr * batch_lr_scale

    # Calc Max Steps
    if args.num_iterations > 0:
        max_steps = args.num_iterations
        if ddp_master:
            print(f"Init: Using user-provided num_iterations={max_steps} without scaling.")
    else:
        max_steps = target_tokens // total_batch_size  # floor the division
        if ddp_master:
            print(f"Init: Calculated max_steps={max_steps} based on scaling_params * target_param_data_ratio / total_batch_size")

    # WD for Optimizers
    def get_wd(step: int):
        return scaled_weight_decay * (1.0 - step / max_steps)  # linearly decay to 0

    # LR / Muon Scheduler functions
    def get_lr(step: int):
        warmup_steps = round(args.warmup_ratio * max_steps)
        warmdown_steps = round(args.warmdown_ratio * max_steps)
        if step < warmup_steps:
            return (step+1) / warmup_steps
        if step <= max_steps - warmdown_steps:
            return 1.0
        else:
            progress = (max_steps - step) / warmdown_steps
            return (progress * 1.0) + (1.0 - progress) * args.final_lr_frac

    def get_muon_momentum(step: int):
        muon_frac = min(step / 300, 1.0)
        muon_momentum = (1.0 - muon_frac) * 0.85 + muon_frac * 0.95
        return muon_momentum

    model_dim = model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    adam_groups = [
        {
            'params': params_lm_head,
            'lr': unembedding_lr * dmodel_lr_scale,
            'is_small': False,
        },
        {
            'params': params_embedding,
            'lr': embedding_lr * dmodel_lr_scale,
            'is_small': False,
        },
        {
            'params': params_val_embds,
            'lr': embedding_lr * dmodel_lr_scale,
            'is_small': False,
        },
        {
            'params': params_resid,
            'lr': scalar_lr * 0.01,
            'is_small': True,
        },
        {
            'params': params_x0,
            'lr': scalar_lr,
            'betas': (0.96, 0.95),
            'is_small': True,
        },
    ]
    adamw_factory = DistAdamW if ddp else torch.optim.AdamW
    adamw_optimizer = adamw_factory(
        adam_groups,
        betas=adam_betas,
        eps=1e-10,
        weight_decay=0.0,
        fused=True,
    )
    muon_groups = []
    for shape in sorted({p.shape for p in params_matrix}):
        group_params = [p for p in params_matrix if p.shape == shape]
        muon_groups.append({'params': group_params})

    muon_factory = DistMuon if ddp else Muon
    muon_optimizer = muon_factory(
        muon_groups,
        lr=matrix_lr,
        momentum=0.95,
        ns_steps=5,
        weight_decay=scaled_weight_decay,
    )
    
    optimizers = [adamw_optimizer, muon_optimizer]
    for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

    # Dataset
    folderpath = os.path.expanduser("~/.cache/nanochat/base_data")
    train_loader = DataLoader(
        folderpath=folderpath,
        split="train",
        batch_size=micro_batch,
        block_size=block_size,
        tokenizer=tokenizer,
        rank=ddp_rank,
        world_size=ddp_world_size,
    )
    if ddp_master:
        print(f"Init: Train dataloader initialised with shards {train_loader.first_shard} - {train_loader.last_shard}")

    assert args.eval_tokens % (micro_batch * block_size * ddp_world_size) == 0
    eval_steps = args.eval_tokens // (micro_batch * block_size * ddp_world_size)
    if ddp_master:
        print(f"Init: Eval BPB every {args.eval_every} steps, eval_steps={eval_steps}")
    eval_loader = DataLoader(
        folderpath=folderpath,
        split="val",
        batch_size=micro_batch,
        block_size=block_size,
        tokenizer=tokenizer,
        rank=ddp_rank,
        world_size=ddp_world_size,
    )
    if ddp_master:
        print(f"Init: Eval dataloader initialised with shards {eval_loader.first_shard} - {eval_loader.last_shard}")


    total_ntok = 0
    total_time = 0.0
    smooth_dt = 0.0
    smooth_train_loss = 0
    for step in range(max_steps+1):

        # BPB Evaluation
        # Always eval on step 0 to get memory allocation warmup (helps if GPU mem super tight)
        if step == 0 or (args.eval_every > 0 and (step % args.eval_every == 0 or step == max_steps)):
            model.eval()
            total_nats = torch.tensor(0.0, device=device, dtype=torch.float32)
            total_bytes = torch.tensor(0, device=device, dtype=torch.int64)
            eval_loader.reset()
            with torch.no_grad():
                for _ in range(eval_steps):
                    x, y = eval_loader.get_batch_bos()
                    assert (y >= 0).all()  # maskig with -1 not supported
                    x = x.to(device)
                    y = y.to(device)
                    with autocast_ctx:
                        _, loss_arr = model(x, y, reduction='none', return_logits=False)
                    bytes_arr = token_bytes[y.view(-1)]
                    loss_arr = loss_arr * (bytes_arr > 0)   # zero loss for tokens with 0 bytes (<bos> etc.)
                    total_nats += loss_arr.sum().item()
                    total_bytes += bytes_arr.sum().item()
            if ddp:
                torch.distributed.all_reduce(total_nats, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(total_bytes, op=torch.distributed.ReduceOp.SUM)
            total_nats = total_nats.item()
            total_bytes = total_bytes.item()
            bpb = float('inf')
            if total_bytes > 0:
                bpb = total_nats / (total_bytes * math.log(2))
            if ddp_master:
                print(f"BPB Eval {step} | BPB {bpb:.14f} | nats {total_nats:.1f} | bytes {total_bytes:.1f}")
            wandb_logger.log({
                'step': step,
                'total_training_time': total_time,
                'val/bpb': bpb,
            })
            model.train()

        # Core Metric
        if args.core_metric_every > 0 and step > 0 and (step % args.core_metric_every == 0 or step == max_steps):
            ts = time.time()
            model.eval()
            with autocast_ctx:
                bundle_path = os.path.expanduser("~/.cache/nanochat/eval_bundle")
                # Original model because shapes keep chaning
                results = evaluate_core_metric(bundle_path, orig_model, tokenizer, device, args.core_metric_max_per_task)
            core_metric = results['core_metric']
            accuracies = {task['label']: task['centered_accuracy'] for task in results['tasks']}
            if device.startswith('cuda'):
                torch.cuda.synchronize() # wait for the GPU to finish work
            dt = (time.time() - ts)
            if ddp_master:
                print(f"CORE {step} | core metric {core_metric:.14f} | dt {dt:.2f}s")
            wandb_logger.log({
                'step': step,
                'core_metric': core_metric,
                'centered_results': accuracies,
            })
            model.train()

        # Generate
        if ddp_master and args.sample_every > 0 and step > 0 and (step % args.sample_every == 0 or step == max_steps):
            model.eval()
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]

            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42)

            bos = tokenizer.encode_single_token('<|bos|>')
            for prompt in prompts:

                tokens =  [bos] + tokenizer.encode(prompt)
                idx = torch.tensor(tokens, dtype=torch.long, device=device)
                idx = idx.unsqueeze(0)  # B,T
                idx = generate(
                    orig_model,
                    autocast_ctx,
                    idx,
                    max_new_tokens=16,
                    temperature=0.0,
                    top_k=None,
                    sample_rng=sample_rng
                )  # B,T
                gen_text = tokenizer.decode(idx[0].tolist())
                print(gen_text)
            model.train()

        # Save Model
        if ddp_master and args.save_every > 0 and step > 0 and (step % args.save_every == 0 or step == max_steps):
            print("Saveing model...")
            models_path = os.path.dirname(__file__)+"/../models/"
            os.makedirs(models_path, exist_ok=True)
            model_data = model.state_dict()
            torch.save(model_data, models_path+f"model_{step:06d}.pt")
            # Calculate MD5 sum of saved file by running os command
            md5sum = os.popen(f"md5sum {models_path}model_{step:06d}.pt").read().split()[0]
            print(f"Saved model_{step:06d}.pt with MD5 sum: {md5sum}")
            
            metadata = {
                'step': step,
                'model_config': model_config.to_dict(),
                'user_config': user_config,
            }
            with open(models_path+f"meta_{step:06d}.json", "w") as f:
                json.dump(metadata, f)

        # Exit Condition
        if step == max_steps:
            break

        # Training
        model.train()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = time.time()
        loss_accum = 0.0
        for opt in optimizers:
            opt.zero_grad()
        for _ in range(grad_accum):
            x, y = train_loader.get_batch_bos()
            x = x.to(device)
            y = y.to(device)
            with autocast_ctx:
                _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
            loss = loss / grad_accum
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)

        # LR Scheduler
        lrm = get_lr(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = group['initial_lr'] * lrm
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_wd(step)
        for group in muon_optimizer.param_groups:
            group['momentum'] = muon_momentum
            group['weight_decay'] = muon_weight_decay

        # Optimizer Step
        for opt in optimizers:
            opt.step()

        # Sync & Time
        if device.startswith('cuda'):
            torch.cuda.synchronize()  # wait for the GPU to finish work
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        dt = (time.time() - ts)
        smooth_dt = 0.9 * smooth_dt + 0.1 * dt
        debiased_smooth_dt = smooth_dt / (1 - 0.9**(step+1))
        total_time += dt

        # Logs
        ntok = (micro_batch * block_size * grad_accum * ddp_world_size)
        total_ntok += ntok
        tps = int(ntok / dt)
        pct = (step) / max_steps * 100
        smooth_train_loss = 0.9 * smooth_train_loss + (1 - 0.9) * train_loss.item()
        debiased_smooth_train_loss = smooth_train_loss / (1 - 0.9**(step+1))
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
        remaining_steps = max_steps - step
        eta_seconds = debiased_smooth_dt * remaining_steps
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        if ddp_master:
            print(f"Step {step}/{max_steps} ({pct:.2f}%) | "
                  f"loss {debiased_smooth_train_loss:.16f} {loss_accum.item():.4f} | "
                  f"lrm {lrm} | dt {dt*1e3:.2f}ms {debiased_smooth_dt*1e3:.2f}ms | tps {tps:,} | mem {max_mem:.3f} GB | "
                  f"time {total_time_str} | eta {eta_str}")
        if step % args.log_every == 0:
            wandb_logger.log({
                'step': step,
                'total_training_time': total_time,
                'train/loss': debiased_smooth_train_loss,
                'train/lrm': lrm,
                'train/dt': dt,
                'train/tok_per_sec': tps,
            })

    if torch.cuda.is_available():
        for r in range(ddp_world_size):
            if r == ddp_rank:
                print(f"Mem rank {ddp_rank}: {torch.cuda.memory_allocated() / (1024**2):.1f}MiB, "
                    f"Res: {torch.cuda.memory_reserved() / (1024**2):.1f}MiB, "
                    f"Max: {torch.cuda.max_memory_allocated() / (1024**2):.1f}MiB")
            if ddp:
                torch.distributed.barrier()
    wandb_logger.finish()
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
