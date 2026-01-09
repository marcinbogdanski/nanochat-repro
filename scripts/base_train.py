import os
import math
import time
import json
import inspect
from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int


class CausalSelfAttentionMarcin(nn.Module):
    """Multiple self-attention heads"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # flag to scale proj into residual
        # self.register_buffer('bias', torch.tril(torch.ones((1, 1, config.block_size, config.block_size))))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)  # B, T, nh*hs
        q = q.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        k = k.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        v = v.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        q = q.transpose(1, 2)  # B,nh,T,hs
        k = k.transpose(1, 2)  # B,nh,T,hs
        v = v.transpose(1, 2)  # B,nh,T,hs

        # W_affin = q @ k.mT / k.shape[-1]**0.5  # B,nh,T,hs @ B,nh,hs,T -> B,nh,T,T
        # W_affin = W_affin.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        # W_affin = torch.softmax(W_affin, dim=-1)  # B,nh,T,T
        # y = W_affin @ v    # B,nh,T,T @ B,nh,T,hs -> B,nh,T,hs
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2)  # B,T,nh,hs
        y = y.contiguous()
        y = y.view(B,T,C)

        out = self.c_proj(y)
        return out

class MLP(nn.Module):
    """Linear transform and activation"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.act = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # flag to scale proj into residual
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionMarcin(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))        # B,T,E pre-norm
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Sharing
        self.transformer.wte.weight = self.lm_head.weight

        # Init Params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # note: wte/lm_head initialized twice, but that's ok
        if isinstance(module, nn.Linear):
            # 0.02 based on openai tensorflow source
            # 1/sqrt(768) = 0.036, 0.02 is "roughly reasonable"
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # each block project to residual 2x times: attention and MLP
                num_layers = 2 * self.config.n_layer
                std *= num_layers**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size
        
        # Embeddings
        pos = torch.arange(T, device=idx.device)  # T
        pos_emb = self.transformer.wpe(pos)       #   T,E <- T
        tok_emb = self.transformer.wte(idx)       # B,T,E <- B,T
        x = tok_emb + pos_emb                     # B,T,E

        # Transformer
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)   # B,T,V <- B,T,E

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits_ = logits.view(B*T, C)  # B*T, C
            targets_ = targets.view(B*T)   # B*T
            loss = F.cross_entropy(logits_, targets_)
            return logits, loss
        
    @classmethod
    def from_pretrained(cls, model_name):
        assert model_name == 'gpt2'  # for now
        from transformers import GPT2LMHeadModel
       
        # HF Model
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')

        # Our Model
        model = GPTModel(GPTConfig(
            block_size=1024,     # max context length, max len feed into the model,
            vocab_size=50257,    # 256 original, 50_000 merges, 1 <|end_of_doc|> token,
            n_layer=12,
            n_head=12,           # head size 384/6=64,
            n_embd=768,          # size of embeddings, i.e. 'first layer',
        ))
        model.eval()

        # Load weights from HF model
        our_sd = model.state_dict()
        our_keys = set(our_sd.keys())
        our_keys = set(k for k in our_keys if not k.endswith('.attn.masked_bias'))
        our_keys = set(k for k in our_keys if not k.endswith('.attn.bias'))
        hf_sd = model_hf.state_dict()
        hf_keys = set(hf_sd.keys())
        hf_keys = set(k for k in hf_keys if not k.endswith('.attn.masked_bias'))
        hf_keys = set(k for k in hf_keys if not k.endswith('.attn.bias'))
        assert our_keys == hf_keys
        for key in hf_keys:
            hf_val = hf_sd[key]
            hf_shape = hf_val.shape
            our_shape = our_sd[key].shape
            transpose = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
            if any(key.endswith(suffix) for suffix in transpose):
                assert hf_shape[::-1] == our_shape
                our_sd[key].copy_(hf_val.t().clone())
            else:
                assert hf_shape == our_shape
                our_sd[key].copy_(hf_val.clone())

        return model

def generate(model, idx, max_new_tokens, top_k=50, sample_rng=None):
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
            idx_tail = idx[:, -block_size:]    # B,T  sliding window
            context = torch.autocast(device_type=idx.device.type, dtype=torch.bfloat16) if idx.device.type == 'cuda' else nullcontext()
            with context:
                logits, _ = model(idx_tail)         # B,T,C <- B,T
            logits = logits[:, -1, :]          # B,C <- B,T,C  discard all but last
            probs = F.softmax(logits, dim=-1)  # B,C
            topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)  # B,k
            ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)  # B,1
            xcol = torch.gather(topk_indices, -1, ix)          # B,1
            idx = torch.cat((idx, xcol), dim=1)                # B,T+1  append
    
    model.train(is_training)
    return idx

class DataLoaderShakespeare:
    def __init__(self, data_path, batch_size, block_size, proc_rank, world_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.proc_rank = proc_rank
        self.world_size = world_size
        self.pos = self.batch_size * self.block_size * self.proc_rank

        with open(data_path, 'r') as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)

    def get_batch(self):
        buff = self.tokens[self.pos:self.pos+self.batch_size*self.block_size+1]
        x = buff[:-1].view(self.batch_size, self.block_size)
        y = buff[1:].view(self.batch_size, self.block_size)

        self.pos += self.batch_size * self.block_size * self.world_size
        if self.pos + self.batch_size * self.block_size * self.world_size + 1 > len(self.tokens):
            self.pos = self.batch_size * self.block_size * self.proc_rank

        return x, y

@dataclass
class DataLoaderState:
    current_shard: int
    pos: int

class DataLoader:
    def __init__(self, data_path, batch_size, block_size, proc_rank, world_size, split):
        self.data_path = data_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.proc_rank = proc_rank
        self.world_size = world_size
        assert split in ['train', 'val']
        self.split = split

        # Read shard file names
        self.shards = os.listdir(data_path)
        self.shards = sorted([s for s in self.shards if self.split in s])
        self.reset()

    def get_state(self):
        return DataLoaderState(
            current_shard=self.current_shard,
            pos=self.pos,
        )

    def load_shard(self, shard_idx):
        filepath = os.path.join(self.data_path, self.shards[shard_idx])
        tokens_np = np.load(filepath).astype(np.int32)
        return torch.tensor(tokens_np, dtype=torch.long)

    def reset(self):
        self.current_shard = 0
        self.tokens = self.load_shard(self.current_shard)
        self.pos = self.batch_size * self.block_size * self.proc_rank

    def get_batch(self):
        buff = self.tokens[self.pos:self.pos+self.batch_size*self.block_size+1]
        x = buff[:-1].view(self.batch_size, self.block_size)
        y = buff[1:].view(self.batch_size, self.block_size)

        self.pos += self.batch_size * self.block_size * self.world_size
        if self.pos + self.batch_size * self.block_size * self.world_size + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_shard(self.current_shard)
            self.pos = self.batch_size * self.block_size * self.proc_rank

        return x, y


class HellaSwagRenderer:
    def __init__(self):
        self.tok = tiktoken.get_encoding("gpt2")
    
    def render_example(self, example):
        ctx, ends, label = example["ctx"], example["endings"], example["label"]
        ctx_ids = self.tok.encode(ctx)
        ending_ids = [self.tok.encode(" " + ending) for ending in ends]  # " " because gpt2 tokenizer
        result_length = max(len(eids) for eids in ending_ids) + len(ctx_ids)
        tokens = torch.zeros((len(ends), result_length), dtype=torch.long)
        mask = torch.zeros((len(ends), result_length), dtype=torch.long)
        for i, eids in enumerate(ending_ids):
            tokens[i, :len(ctx_ids)] = torch.tensor(ctx_ids, dtype=torch.long)
            tokens[i, len(ctx_ids) : len(ctx_ids)+len(eids)] = torch.tensor(eids, dtype=torch.long)
            mask[i, len(ctx_ids) : len(ctx_ids)+len(eids)] = 1
        return tokens, mask, label


class LRScheduler:
    def __init__(self, max_lr, min_lr, warmup_steps, max_steps):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step+1) / self.warmup_steps
        elif self.warmup_steps <= step < self.max_steps:
            decay_ratio = (step-self.warmup_steps) / (self.max_steps-self.warmup_steps)
            cf = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.min_lr + cf * (self.max_lr - self.min_lr)
        else:
            return self.min_lr

def main():
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
    print(f"{ddp=} {ddp_rank=}, {ddp_local_rank=}, {ddp_world_size=}, {ddp_master=}, {device=}")

    # NOTE: Only affects perforamnce if autocast is *not* used
    # Option 1 - old API
    # torch.set_float32_matmul_precision("high")        # in video, causes deprecated warning
    # Option 2 - new API
    # Note = 'tf32' is the correct way as per torch 2.9 docs
    # assert torch.backends.cuda.matmul.fp32_precision    # check they exist
    # assert torch.backends.cudnn.conv.fp32_precision
    # torch.backends.cuda.matmul.fp32_precision = 'tf32'  # newer api
    # torch.backends.cudnn.conv.fp32_precision = 'tf32'
    # Option 3 - compatible with generation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Reproducibility
    # Model init relies on identical random seeds, will address later
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    ################################ EQUIVALENCE ###############################
    # Dissable TORCH.COMPILE for reproducibility non-DDP/DDP
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    ############################################################################

    # Batching
    total_batch_size = 524288    # 2**19, ~0.5M
    micro_batch = 16             # what fits in GPU
    block_size = 1024
    assert total_batch_size % (block_size*micro_batch*ddp_world_size) == 0
    grad_accum = total_batch_size // (block_size*micro_batch*ddp_world_size)
    if ddp_master:
        print(f"{total_batch_size=}, {block_size=}, {micro_batch=}, {ddp_world_size=}, {grad_accum=}")

    # Data Loader
    train_loader = DataLoader(
        data_path=os.path.dirname(__file__)+'/../data/fineweb-edu-sample-10BT',
        batch_size=micro_batch,
        block_size=block_size,
        proc_rank=ddp_rank,
        world_size=ddp_world_size,
        split='train',
    )
    val_loader = DataLoader(
        data_path=os.path.dirname(__file__)+'/../data/fineweb-edu-sample-10BT',
        batch_size=micro_batch,
        block_size=block_size,
        proc_rank=ddp_rank,
        world_size=ddp_world_size,
        split='val',
    )

    # Model
    # NOTE: because vocab_size is expanded model may in theory generate invalid tokens
    model = GPTModel(GPTConfig(
        block_size=1024,     # max context length, max len feed into the model,
        vocab_size=50304,    # 50304 is 'nicer', original was 50257
        n_layer=12,
        n_head=12,           # head size 768/12=64,
        n_embd=768,          # size of embeddings, i.e. 'first layer',
    ))
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    ################################ EQUIVALENCE ###############################
    # - CUDA_VISIBLE_DEVICES=-1 python train_gpt2.py
    # - python train_gpt2.py
    # - torchrun --standalone --nproc_per_node=2 train_gpt2.py
    # torch.set_printoptions(precision=10, sci_mode=True, linewidth=200)
    # model_raw = model.module if ddp else model
    # for r in range(ddp_world_size):
    #     if ddp_local_rank == r:
    #         print(f"--- [rank {ddp_local_rank}] ---", flush=True)
    #         print("Init:")
    #         print(model_raw.transformer.wte.weight[0, :10].detach().cpu())
    #         print(model_raw.transformer.h[0].ln_1.weight[:10].detach().cpu())
    #         print(model_raw.transformer.h[0].attn.c_attn.weight[0, :10].detach().cpu())
    #         print(model_raw.transformer.h[0].mlp.c_proj.weight[0, :10].detach().cpu())
    #         print(model_raw.transformer.ln_f.weight[:10].detach().cpu())
    #         print(model_raw.lm_head.weight[0, :10].detach().cpu())
    #         print(f"--- ----------------------- ---", flush=True)
    #     if ddp:
    #         torch.distributed.barrier()
    ############################################################################

    # LR Scheduler
    max_lr = 6e-4              # params from GPT-3 paper, 124M model
    min_lr = max_lr * 0.1
    warmup_steps = 715         # 375M tokens / 2**19 tok = 715 steps
    max_steps = 19073          # 10B tokens / 2**19 tok = 19073 - 1 epoch
    lr_scheduler = LRScheduler(max_lr, min_lr, warmup_steps=warmup_steps, max_steps=max_steps)

    # Optimizer
    weight_decay = 0.1
    # All params that require grad
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # weight decay 2D params (matmul, embd), skip 1D (biases, layernorms)
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    if ddp_master:
        print(f"Decay {len(decay_params)} tensors with {num_decay} params")
        print(f"Nodecay {len(nodecay_params)} tensors with {num_nodecay} params")
    # Check if fused adam
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    if ddp_master:
        print(f"Using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )

    # Eval Params
    eval_loss_every = 250  # steps
    eval_accum_steps = 20   # steps
    # Checkpoint
    checkpoint_every = 5000  # steps
    assert checkpoint_every % eval_loss_every == 0

    # Generate Params
    gen_samples_every = 250  # steps
    tok = tiktoken.get_encoding("gpt2")
    prompt = "Hello, I'm a language model,"  # 8 tokens
    max_new_tokens = 24
    num_generate = 4

    # HellaSwag Stuff
    # Read all lines from the validation set
    hellaswag_every = 250  # steps
    hellaswag_renderer = HellaSwagRenderer()
    hellaswag_fp = os.path.dirname(__file__) + "/../data/hellaswag/hellaswag_val.jsonl"
    with open(hellaswag_fp, "r") as f:
        lines = f.readlines()
    hellaswag_examples = [json.loads(line) for line in lines]

    # Logging
    logfile = "log.txt"
    with open(logfile, 'w') as f:
        pass  # clear logfile

    total_tok = 0
    for i in range(max_steps):
        
        ########################################
        # Evaluate validation loss
        if (i % eval_loss_every) == 0 or (i == max_steps-1):
            model.eval()
            val_loader.reset()
            ts = time.time()
            loss_val_accum = 0.0    
            with torch.no_grad():
                for ii in range(eval_accum_steps):
                    x, y = val_loader.get_batch()
                    x, y = x.to(device), y.to(device)
                    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == 'cuda' else nullcontext()
                    with autocast_ctx:
                        _, loss = model(x, y)
                    loss = loss / eval_accum_steps
                    loss_val_accum += loss.detach()
            if ddp:
                torch.distributed.all_reduce(loss_val_accum, op=torch.distributed.ReduceOp.AVG)
                    
            # Logs
            if device.startswith('cuda'):
                torch.cuda.synchronize() # wait for the GPU to finish work
            dt = (time.time() - ts)
            tps = (micro_batch * block_size * eval_accum_steps * ddp_world_size) / dt
            if ddp_master:
                print(f"Eval at {i:4d}:, L={loss_val_accum.item():.6f}, dt={dt*1e3:.2f}ms, tps={tps:.2f}")
                with open(logfile, 'a') as f:
                    f.write(f"val,{i},{total_tok},{loss_val_accum.item():.6f}\n")
                if (i > 0 and (i % checkpoint_every) == 0) or (i == max_steps-1):
                    model_raw = model.module if ddp else model
                    ckpt_path = f"ckpt_step_{i:06d}.pt"
                    checkpoint = {
                        'model_state_dict': model_raw.state_dict(),
                        'model_config': model_raw.config,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loader_state': train_loader.get_state(),
                        'val_loader_state': val_loader.get_state(),
                        'step': i,
                        'total_tok': total_tok,
                        'val_loss': loss_val_accum.item(),
                    }
                    torch.save(checkpoint, ckpt_path)

        ########################################
        # HellaSwag Evaluation
        if (i % hellaswag_every) == 0 or (i == max_steps-1):
            model.eval()
            num_total = 0
            num_correct = 0
            for j in range(len(hellaswag_examples)):
                if j % ddp_world_size != ddp_rank:
                    continue  # only do i-th example
                example = hellaswag_examples[j]
                tokens, mask, label = hellaswag_renderer.render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == 'cuda' else nullcontext()
                    with autocast_ctx:
                        logits, _ = model(tokens)
                    logits_shifted = logits[:, :-1, :].contiguous()
                    tokens_shifted = tokens[:, 1:].contiguous()
                    mask_shifted = mask[:, 1:].contiguous()
                    B, T_1, V = logits_shifted.shape
                    losses_shifted = F.cross_entropy(
                        logits_shifted.view(B*T_1, V),
                        tokens_shifted.view(B*T_1),
                        reduction='none'
                    ).view(B, T_1)
                    losses_shifted_masked = losses_shifted * mask_shifted
                    losses_avg = losses_shifted_masked.sum(dim=1) / mask_shifted.sum(dim=1)
                    model_label = torch.argmin(losses_avg).item()
                    num_total += 1
                    if model_label == label:
                        num_correct += 1
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct = torch.tensor(num_correct, dtype=torch.long, device=device)
                torch.distributed.all_reduce(num_total, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(num_correct, op=torch.distributed.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct = num_correct.item()
            acc_norm = num_correct / num_total
            if ddp_master:
                print(f"HellaSwag acc={acc_norm:.4f} correct={num_correct} total={num_total}")
                with open(logfile, 'a') as f:
                    f.write(f"hella,{i},{total_tok},{acc_norm:.6f}\n")


        ########################################
        # Generate samples
        if (i > 0 and (i % gen_samples_every) == 0) or (i == max_steps-1):
            model.eval()

            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)

            tokens = tok.encode(prompt)
            idx = torch.tensor(tokens, dtype=torch.long, device=device)
            idx = idx.unsqueeze(0).repeat(num_generate, 1)  # B,T
            idx = generate(model, idx, max_new_tokens, top_k=50, sample_rng=sample_rng) # B,T

            for r in range(ddp_world_size):
                if ddp_local_rank == r:
                    for b in range(idx.shape[0]):
                        gen_text = tok.decode(idx[b].tolist())
                        print(f"  {r}:{b} > {gen_text}", flush=True)
                        if ddp_master:
                            with open(logfile, 'a') as f:
                                f.write(f"gen,{i},{total_tok},{r},{b},{gen_text}\n")
                if ddp:
                    torch.distributed.barrier()

        ########################################
        # Training step
        model.train()
        ts = time.time()

        # Calc gradient
        loss_accum = 0.0
        optimizer.zero_grad()
        for ii in range(grad_accum):
            x, y = train_loader.get_batch()
            x, y = x.to(device), y.to(device)
            # autocast to bfloat16 hangs in backward() on CPU
            # https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == 'cuda' else nullcontext()
            with autocast_ctx:
                _, loss = model(x, y)
            loss /= grad_accum
            loss_accum += loss.detach()
            # Sync only if DDP and last backward step
            context = model.no_sync() if (ddp and ii < grad_accum - 1) else nullcontext()
            with context:
                loss.backward()
        if ddp:
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
            
        # Gradient norm clip
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        lr = lr_scheduler.get_lr(i)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()

        # Logs
        if device.startswith('cuda'):
            torch.cuda.synchronize() # wait for the GPU to finish work
        dt = (time.time() - ts)
        ntok = (micro_batch * block_size * grad_accum* ddp_world_size)
        total_tok += ntok
        tps = ntok / dt
        if ddp_master:
            pct = (i+1) / max_steps * 100
            cs = train_loader.current_shard
            cp = train_loader.pos
            print(f"{i:4d} ({pct:.2f}%) [{cs};{cp:,}]:, L={loss_accum.item():.6f}, lr={lr:.4e} norm={norm:.4f}, dt={dt*1e3:.2f}ms, tps={tps:.2f}")
            with open(logfile, 'a') as f:
                f.write(f"train,{i},{total_tok},{loss_accum.item():.6f},{lr:.6e},{norm:.6f},{dt*1e3:.2f},{tps:.2f}\n")
        


    ################################ EQUIVALENCE ###############################
    # - CUDA_VISIBLE_DEVICES=-1 python train_gpt2.py          <- different
    # - CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_gpt2.py
    # - CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc_per_node=2 train_gpt2.py
    # EQUIVALENCE CHECK: WEIGHT INIT
    # torch.set_printoptions(precision=10, sci_mode=True, linewidth=200)
    # model_raw = model.module if ddp else model
    # for r in range(ddp_world_size):
    #     if ddp_local_rank == r:
    #         print(f"--- [rank {ddp_local_rank}] ---", flush=True)
    #         print("After:")
    #         print(model_raw.transformer.wte.weight[0, :10].detach().cpu())
    #         print(model_raw.transformer.h[0].ln_1.weight[:10].detach().cpu())
    #         print(model_raw.transformer.h[0].attn.c_attn.weight[0, :10].detach().cpu())
    #         print(model_raw.transformer.h[0].mlp.c_proj.weight[0, :10].detach().cpu())
    #         print(model_raw.transformer.ln_f.weight[:10].detach().cpu())
    #         print(model_raw.lm_head.weight[0, :10].detach().cpu())
    #         print(f"--- ----------------------- ---", flush=True)
    #     if ddp:
    #         torch.distributed.barrier()
    ############################################################################

    # Avg dt: 534.84  Agv tps: 15364.54 - base fp32
    # Avg dt: 405.77  Agv tps: 20282.09 - use tf32
    # Avg dt: 303.13  Agv tps: 27361.53 - add autocast to bf16 in fwd/loss calc
    # Avg dt: 183.24  Agv tps: 44706.74 - add torch.compile()
    # Avg dt: 139.38  Agv tps: 58773.61 - fused flash attention
    # Avg dt: 136.42  Agv tps: 60050.99 - switch vocab size to 50304
    # Avg dt: 134.80  Agv tps: 60773.48 - fused AdamW

    # total_batch_size=524288, block_size=1024, micro_batch=16, ddp_world_size=1, grad_accum=32
    # 9:, L=7.341308, lr=6.0000e-04 norm=1.8631, dt=8139.71ms, tps=64411.14

    # total_batch_size=524288, block_size=1024, micro_batch=16, ddp_world_size=1, grad_accum=32
    # 9:, L=7.331692, lr=6.0000e-04 norm=1.9200, dt=7618.34ms, tps=68819.22

    # total_batch_size=524288, block_size=1024, micro_batch=16, ddp_world_size=2, grad_accum=16
    # 8:, L=7.696642, lr=5.4000e-04 norm=2.3113, dt=4870.42ms, tps=107647.40
    if ddp:
        torch.distributed.destroy_process_group()
    print("Bye")


if __name__ == "__main__":
    main()

