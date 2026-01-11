import os
import math
import time
import inspect
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

class GPTConfig:
    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd, dropout=0.0):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout  # 0.0, Karpathy doesn't use dropout in the video



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
        our_keys = set(k for k in our_sd.keys() if not k.endswith('.attn.bias'))
        hf_sd = model_hf.state_dict()
        hf_keys = set(hf_sd.keys())
        assert our_keys == hf_keys
        for key, hf_val in hf_sd.items():
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

def generate(model, idx, max_new_tokens, top_k=50):
    """Generate max_tokens starting from idx[B,T]"""
    assert isinstance(idx, torch.Tensor)
    assert idx.dtype == torch.long
    assert len(idx.shape) == 2  # B,T
    assert isinstance(max_new_tokens, int)
    
    model.eval()
    block_size = model.config.block_size
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_tail = idx[:, -block_size:]    # B,T  sliding window
            logits, _ = model(idx_tail)         # B,T,C <- B,T
            logits = logits[:, -1, :]          # B,C <- B,T,C  discard all but last
            probs = F.softmax(logits, dim=-1)  # B,C
            topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)  # B,k
            ix = torch.multinomial(topk_probs, num_samples=1)  # B,1
            xcol = torch.gather(topk_indices, -1, ix)          # B,1
            idx = torch.cat((idx, xcol), dim=1)                # B,T+1  append
    return idx

class DataLoader:
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

        # TODO: needs per-epoch shuffling deterministic across DDP
        # TODO: Position reset logic is subject to slightly different behavior between:
        # - no DDP, large batch with gradient accumlation - boundry checked every mini_batch
        # - DDP, large batch spread across GPUs, no grad accum - boundry checked every full batch
        self.pos += self.batch_size * self.block_size * self.world_size
        if self.pos + self.batch_size * self.block_size * self.world_size + 1 > len(self.tokens):
            self.pos = self.batch_size * self.block_size * self.proc_rank

        return x, y

# [          page 0           |           page 1          ]
# [   accum 0   |   accum 1   |   accum 0   |   accum 1   ]
# [ dds0 | dds1 | dds0 | dds1 | dds0 | dds1 | dds0 | dds1 ]
class DataLoaderPaged:
    def __init__(self, data_path, batch_size, block_size, proc_rank, world_size, grad_accum):
        self.data_path = data_path
        self.batch_size = batch_size
        self.block_size = block_size
        self.proc_rank = proc_rank
        self.world_size = world_size
        self.grad_accum = grad_accum
        self.page_idx = 0
        self.accum_idx = 0
        self.page_size = self.batch_size * self.block_size * self.world_size * self.grad_accum

        with open(data_path, 'r') as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)

        self.last_batch = None  # Debug

    def get_batch(self):
        buff_start = \
            self.page_idx * self.page_size + \
            self.accum_idx * (self.batch_size * self.block_size * self.world_size) + \
            self.proc_rank * (self.batch_size * self.block_size)
        buff_end = buff_start + (self.batch_size * self.block_size)
        buff = self.tokens[buff_start:buff_end+1]  # +1 to grab last target

        x = buff[:-1].view(self.batch_size, self.block_size)
        y = buff[1:].view(self.batch_size, self.block_size)
        # TODO: needs per-epoch shuffling deterministic across DDP
        self.accum_idx += 1
        if self.accum_idx >= self.grad_accum:
            self.accum_idx = 0
            self.page_idx += 1
            if self.page_idx * self.page_size + self.page_size + 1  > len(self.tokens):
                self.page_idx = 0

        return x, y


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

    # Note = 'tf32' is the correct way as per torch 2.9 docs
    # torch.set_float32_matmul_precision("high")        # in video, causes deprecated warning
    assert torch.backends.cuda.matmul.fp32_precision    # check they exist
    assert torch.backends.cudnn.conv.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = 'tf32'  # newer api
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

    # Reproducibility
    # TODO: Model init relies on identical random seeds, will address later
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
    total_batch_size = 524288 // 2   # 2**19, ~0.5M
    micro_batch = 16             # what fits in GPU
    block_size = 1024
    assert total_batch_size % (block_size*micro_batch*ddp_world_size) == 0
    grad_accum = total_batch_size // (block_size*micro_batch*ddp_world_size)
    if ddp_master:
        print(f"{total_batch_size=}, {block_size=}, {micro_batch=}, {ddp_world_size=}, {grad_accum=}")

    # Data Loader
    data_loader = DataLoaderPaged(
        data_path=os.path.dirname(__file__)+'/../data/tinyshakespeare.txt',
        batch_size=micro_batch,
        block_size=block_size,
        proc_rank=ddp_rank,
        world_size=ddp_world_size,
        grad_accum=grad_accum)

    # Model
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
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 50
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

    dt_list, tps_list = [], []
    model.train()
    for i in range(max_steps):
        ts = time.time()

        # Calc gradient
        loss_accum = 0.0
        optimizer.zero_grad()
        for ii in range(grad_accum):
            x, y = data_loader.get_batch()
            x, y = x.to(device), y.to(device)
            # autocast to bfloat16 hangs in backward() on CPU
            # https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == 'cuda' else nullcontext()
            with autocast_ctx:
                logits, loss = model(x, y)
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
        tps = (micro_batch * block_size * grad_accum* ddp_world_size) / dt
        if i != 0:  # skip compile
            dt_list.append(dt), tps_list.append(tps)
        if ddp_master:
            print(f"{i:4d}:, L={loss_accum.item():.6f}, lr={lr:.4e} norm={norm:.4f}, dt={dt*1e3:.2f}ms, tps={tps:.2f}")

    if ddp_master and dt_list:
        print(f"Avg dt: {sum(dt_list)/len(dt_list)*1e3:.2f}  Agv tps: {sum(tps_list)/len(tps_list):.2f}")


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

