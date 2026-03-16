import torch
import torch.nn as nn
import torch.nn.functional as F

# Flash Attention 3, source wheel with 3090 support
from kernels import get_kernel
flash_attn = get_kernel('kernels-community/flash-attn3').flash_attn_interface

class GPTConfig:
    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd, window_pattern):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.window_pattern = window_pattern

    def to_dict(self):
        return {
            'block_size': self.block_size,
            'vocab_size': self.vocab_size,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_embd': self.n_embd,
            'window_pattern': self.window_pattern,
        }

class CausalSelfAttentionRoPE(nn.Module):
    """Multiple self-attention heads"""
    def __init__(self, config, ve_enable):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # VE Gate
        self.ve_gate_size = 32
        self.ve_gate = nn.Linear(32, config.n_head, bias=False) if ve_enable else None

    def _apply_rope(self, q, cos, sin):
        B, T, nh, hs = q.size()
        # Trim sin, cos to T and add batch dim
        #sin = sin[:T, :].view(1, T, 1, hs//2)     # 1,T,1,hs/2       ### MARCIN - my version
        #cos = cos[:T, :].view(1, T, 1, hs//2)                        ### MARCIN - my version
        sin = sin[:, :T, :, :]     # 1,T,1,hs/2                       ### MARCIN - nanochat version
        cos = cos[:, :T, :, :]     # 1,T,1,hs/2                       ### MARCIN - nanochat version
        # Split x/y
        q_x, q_y = q[..., :hs//2], q[..., hs//2:]  # B,T,nh,hs/2
        # Apply rotation
        q_x_rot = cos * q_x + sin * q_y
        q_y_rot = -sin * q_x + cos * q_y
        # Combine back
        q_rot = torch.cat([q_x_rot, q_y_rot], dim=-1)        # B,T,nh,hs
        return q_rot
    
    @classmethod
    def precalculate_cos_sin(cls, seq_len, head_size, base=10_000, device=None):             ### MARCIN added device=
        # Compute exponent for the RoPE frequencies
        theta = torch.arange(0, head_size, step=2, dtype=torch.float32, device=device)       ### MARCIN added dtype= device=
        # theta = base**-(theta/head_size)       # head_size//2                              ### MARCIN my version
        theta = 1.0 / (base**(theta/head_size))  # head_size//2                              ### MARCIN nanochat version
        pos = torch.arange(0, seq_len, dtype=torch.float32, device=device)       # seq_len   ### MARCIN added dtype= device=
        tmp = torch.outer(pos, theta)        # seq_len, head_size//2
        sin, cos = torch.sin(tmp), torch.cos(tmp)
        sin, cos = sin[None, :, None, :], cos[None, :, None, :]  # 1,seq_len,1,head_size//2  ### MARCIN different sclicing
        cos, sin = cos.bfloat16(), sin.bfloat16()
        return cos, sin

    def forward(self, x, ve, cos, sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x)    # B, T, nh*hs
        k = self.c_k(x)    # B, T, nh*hs
        v = self.c_v(x)    # B, T, nh*hs
        q = q.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        k = k.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
        v = v.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs

        if self.ve_gate is not None:
            ve = ve.view(B, T, self.n_head, C//self.n_head)  # B,T,nh,hs
            gate = 2.0 * F.sigmoid(self.ve_gate(x[..., :self.ve_gate_size]))  # B, T, nh
            v = v + gate.unsqueeze(-1) * ve

        q_rot = self._apply_rope(q, cos, sin)
        k_rot = self._apply_rope(k, cos, sin)

        # Normalize q,k
        q_rot = F.rms_norm(q_rot, (q_rot.size(-1),))
        k_rot = F.rms_norm(k_rot, (k_rot.size(-1),))

        # q = q_rot.transpose(1, 2)  # B,nh,T,hs
        # k = k_rot.transpose(1, 2)  # B,nh,T,hs
        # v = v.transpose(1, 2)  # B,nh,T,hs

        y = flash_attn.flash_attn_func(q_rot, k_rot, v, causal=True, window_size=window_size, deterministic=True)

        # y = y.transpose(1, 2)  # B,T,nh,hs
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
    def __init__(self, config, ve_enable):
        super().__init__()
        self.attn = CausalSelfAttentionRoPE(config, ve_enable)
        self.mlp = MLP(config)

    def _norm(self, x):
        return F.rms_norm(x, (x.size(-1),))

    def forward(self, x, ve, cos, sin, window_size):
        x = x + self.attn(self._norm(x), ve, cos, sin, window_size)        # B,T,E pre-norm
        x = x + self.mlp(self._norm(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, self._has_ve(i, config.n_layer)) for i in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Params for merging x0 across network and blending residual stream
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Value embeddings for each layer
        self.value_embeds = nn.ModuleDict({
            str(i) : nn.Embedding(config.vocab_size, config.n_embd) for i in range(config.n_layer) if self._has_ve(i, config.n_layer)
        })

        cos, sin = CausalSelfAttentionRoPE.precalculate_cos_sin(
            config.block_size * 10, config.n_embd // config.n_head
        )
        self.register_buffer("cos", cos, persistent=False)  # don't save to checkpoint
        self.register_buffer("sin", sin, persistent=False)

        # Pre-calculate window size tuples (context_length, 0) for each layer
        self.window_sizes = self._calc_window_sizes(self.config)

    def _calc_window_sizes(self, config):
        chat_to_window_type = {
            'L': (config.block_size, 0),
            'S': (config.block_size//2, 0),
        }
        window_sizes = []
        for layer_idx in range(config.n_layer):
            window_type = config.window_pattern[layer_idx % len(config.window_pattern)]
            window_sizes.append(chat_to_window_type[window_type])
        window_sizes[-1] = (config.block_size, 0)  # Last layer always full attention
        return window_sizes
    
    def _has_ve(self, layer_idx, n_layer):
        # Every other layer, last always included
        return layer_idx % 2 == (n_layer-1) % 2

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

        torch.nn.init.constant_(self.resid_lambdas, 1.0)
        torch.nn.init.constant_(self.x0_lambdas, 0.1)

        # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                # Init to zero, so sigmoid(0) -> 0.5, 2*0.5 = 1, i.e. enabled neutral at the start
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        self.cos, self.sin = CausalSelfAttentionRoPE.precalculate_cos_sin(          ### MARCIN - init here as well as in constructor (remove?)
            self.config.block_size * 10, self.config.n_embd // self.config.n_head,
            device=self.transformer.wte.weight.device
        )

        # Cast to bfloat16 to align with NanoChat
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)



    def forward(self, idx, targets=None, reduction='mean', return_logits=True):
        B, T = idx.shape
        assert T <= self.cos.size(1), "Cannot forward, model block size is exhausted."
        assert idx.device == self.cos.device, "Input device does not match model device."
        assert self.cos.dtype == torch.bfloat16, "Model buffers are not in bfloat16."

        # Embeddings
        x = self.transformer.wte(idx)             # B,T,E <- B,T
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        # Transformer
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if self._has_ve(i, self.config.n_layer) else None
            x = block(x, ve, self.cos, self.sin, self.window_sizes[i])
        x = F.rms_norm(x, (x.size(-1),))

        # Logits
        softcap = 15
        logits = self.lm_head(x)   # B,T,V <- B,T,E
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is None:
            assert return_logits, "If targets is None, return_logits must be True."
            return logits, None
        else:
            B, T, C = logits.shape
            logits_ = logits.view(B*T, C)  # B*T, C
            targets_ = targets.view(B*T)   # B*T
            loss = F.cross_entropy(logits_, targets_, reduction=reduction)
            if return_logits:
                return logits, loss
            else:
                return None, loss
