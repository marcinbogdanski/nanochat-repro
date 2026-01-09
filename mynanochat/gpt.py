from dataclasses import dataclass
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F


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
