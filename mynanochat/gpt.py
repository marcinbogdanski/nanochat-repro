import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd


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
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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
        x = self.transformer.wte(idx)             # B,T,E <- B,T

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
