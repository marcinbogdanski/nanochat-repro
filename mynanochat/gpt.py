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


class CausalSelfAttentionRoPE(nn.Module):
    """Multiple self-attention heads"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def _apply_rope(self, q, cos, sin):
        B, T, nh, hs = q.size()
        # Trim sin, cos to T and add batch dim
        sin = sin[:T, :].view(1, T, 1, hs//2)     # 1,T,1,hs/2
        cos = cos[:T, :].view(1, T, 1, hs//2)
        # Split x/y
        q_x, q_y = q[..., :hs//2], q[..., hs//2:]  # B,T,nh,hs/2
        # Apply rotation
        q_x_rot = cos * q_x + sin * q_y
        q_y_rot = -sin * q_x + cos * q_y
        # Combine back
        q_rot = torch.cat([q_x_rot, q_y_rot], dim=-1)        # B,T,nh,hs
        return q_rot
    
    @classmethod
    def precalculate_cos_sin(cls, seq_len, head_size, base=10_000):
        # Compute exponent for the RoPE frequencies
        theta = torch.arange(0, head_size, step=2)
        theta = base**-(theta/head_size)     # head_size//2
        pos = torch.arange(0, seq_len)       # seq_len
        tmp = torch.outer(pos, theta)        # seq_len, head_size//2
        sin, cos = torch.sin(tmp), torch.cos(tmp)
        cos, sin = cos.bfloat16(), sin.bfloat16()
        return cos, sin

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q = self.c_q(x)    # B, T, nh*hs
        k = self.c_k(x)    # B, T, nh*hs
        v = self.c_v(x)    # B, T, nh*hs
        q = q.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        k = k.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        v = v.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs

        q_rot = self._apply_rope(q, cos, sin)
        k_rot = self._apply_rope(k, cos, sin)

        # Normalize q,k
        q_rot = F.rms_norm(q_rot, (q_rot.size(-1),))
        k_rot = F.rms_norm(k_rot, (k_rot.size(-1),))

        q = q_rot.transpose(1, 2)  # B,nh,T,hs
        k = k_rot.transpose(1, 2)  # B,nh,T,hs
        v = v.transpose(1, 2)  # B,nh,T,hs

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
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=False)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttentionRoPE(config)
        self.mlp = MLP(config)

    def _norm(self, x):
        return F.rms_norm(x, (x.size(-1),))

    def forward(self, x, cos, sin):
        x = x + self.attn(self._norm(x), cos, sin)        # B,T,E pre-norm
        x = x + self.mlp(self._norm(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        cos, sin = CausalSelfAttentionRoPE.precalculate_cos_sin(
            config.block_size, config.n_embd // config.n_head
        )
        self.register_buffer("cos", cos, persistent=False)  # don't save to checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        """Initialization following NanoChat

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Cast to bfloat16 to align with NanoChat
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

        

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size
        
        # Embeddings
        x = self.transformer.wte(idx)             # B,T,E <- B,T
        x = F.rms_norm(x, (x.size(-1),))

        # Transformer
        for block in self.transformer.h:
            x = block(x, self.cos, self.sin)
        x = F.rms_norm(x, (x.size(-1),))

        # Logits
        softcap = 15
        logits = self.lm_head(x)   # B,T,V <- B,T,E
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits_ = logits.view(B*T, C)  # B*T, C
            targets_ = targets.view(B*T)   # B*T
            loss = F.cross_entropy(logits_, targets_)
            return logits, loss
