import torch
import torch.nn as nn
import torch.nn.functional as F
from mynanochat.fp8 import LinearFP8
from mynanochat.moe import MoE
from mynanochat.flash_attention import sdpa_attn_func, fa3_attn_func

class GPTConfig:
    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd, window_pattern, moe_enable, moe_n_experts, moe_top_k):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.window_pattern = window_pattern
        self.moe_enable = moe_enable
        self.moe_n_experts = moe_n_experts
        self.moe_top_k = moe_top_k

    def to_dict(self):
        return {
            'block_size': self.block_size,
            'vocab_size': self.vocab_size,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_embd': self.n_embd,
            'window_pattern': self.window_pattern,
            'moe_enable': self.moe_enable,
            'moe_experts': self.moe_n_experts,
            'moe_top_k': self.moe_top_k,
        }

class CausalSelfAttentionRoPE(nn.Module):
    """Multiple self-attention heads"""
    def __init__(self, config, ve_enable, enable_fa3):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert isinstance(enable_fa3, bool)
        self.n_head = config.n_head
        self.block_size = config.block_size
        self.enable_fa3 = enable_fa3

        self.c_q = LinearFP8(config.n_embd, config.n_embd, bias=False)
        self.c_k = LinearFP8(config.n_embd, config.n_embd, bias=False)
        self.c_v = LinearFP8(config.n_embd, config.n_embd, bias=False)
        self.c_proj = LinearFP8(config.n_embd, config.n_embd, bias=False)

        # VE Gate
        self.ve_gate_size = 12
        self.ve_gate = LinearFP8(self.ve_gate_size, config.n_head, bias=False) if ve_enable else None

    def _apply_rope(self, q, cos, sin):
        B, T, nh, hs = q.size()
        # Trim sin, cos to T and add batch dim
        sin = sin[:, :T, :, :]     # 1,T,1,hs/2
        cos = cos[:, :T, :, :]     # 1,T,1,hs/2
        # Split x/y
        q_x, q_y = q[..., :hs//2], q[..., hs//2:]  # B,T,nh,hs/2
        # Apply rotation
        q_x_rot = cos * q_x + sin * q_y
        q_y_rot = -sin * q_x + cos * q_y
        # Combine back
        q_rot = torch.cat([q_x_rot, q_y_rot], dim=-1)        # B,T,nh,hs
        return q_rot
    
    @classmethod
    def precalculate_cos_sin(cls, seq_len, head_size, base=100_000, device=None, dtype=None):
        # Compute exponent for the RoPE frequencies
        theta = torch.arange(0, head_size, step=2, dtype=torch.float32, device=device)
        # theta = base**-(theta/head_size)       # head_size//2
        theta = 1.0 / (base**(theta/head_size))  # head_size//2
        pos = torch.arange(0, seq_len, dtype=torch.float32, device=device)       # seq_len
        tmp = torch.outer(pos, theta)        # seq_len, head_size//2
        sin, cos = torch.sin(tmp), torch.cos(tmp)
        sin, cos = sin[None, :, None, :], cos[None, :, None, :]  # 1,seq_len,1,head_size//2
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
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
            gate = 3.0 * F.sigmoid(self.ve_gate(x[..., :self.ve_gate_size]))  # B, T, nh
            v = v + gate.unsqueeze(-1) * ve

        q_rot = self._apply_rope(q, cos, sin)
        k_rot = self._apply_rope(k, cos, sin)

        # Normalize q,k
        q_rot = F.rms_norm(q_rot, (q_rot.size(-1),))
        k_rot = F.rms_norm(k_rot, (k_rot.size(-1),))

        # Sharper attention
        q_rot = q_rot * 1.2
        k_rot = k_rot * 1.2

        if self.enable_fa3:
            # Flash Attention 3
            y = fa3_attn_func(q_rot, k_rot, v, causal=True, window_size=window_size)
        else:
            # SDPA fallback
            y = sdpa_attn_func(q_rot, k_rot, v, causal=True, window_size=window_size)

        y = y.contiguous()
        y = y.view(B,T,C)

        out = self.c_proj(y)
        return out

class MLP(nn.Module):
    """Linear transform and activation"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = LinearFP8(config.n_embd, 4*config.n_embd, bias=False)
        self.c_proj = LinearFP8(4*config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, ve_enable, enable_fa3):
        super().__init__()
        self.attn = CausalSelfAttentionRoPE(config, ve_enable, enable_fa3)
        self.moe_enable = config.moe_enable
        if not config.moe_enable:
            self.mlp = MLP(config)
        else:
            self.moe = MoE(dim=config.n_embd, n_routed_experts=config.moe_n_experts, top_k=config.moe_top_k)

    def _norm(self, x):
        return F.rms_norm(x, (x.size(-1),))

    def forward(self, x, ve, cos, sin, window_size):
        x = x + self.attn(self._norm(x), ve, cos, sin, window_size)        # B,T,E pre-norm
        if not self.moe_enable:
            x = x + self.mlp(self._norm(x))
        else:
            x = x + self.moe(self._norm(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config, compute_dtype, enable_fa3, fp8_training):
        """Initialize to default device/dtype here, cast to compute_dtype in init_weights.
        
        Type handling:
        - all inputs (RoPE/WTE/VE) are in compute_dtype to begin with
        - in forward pass:
          - model parameters (float32) are cast to input dtype (compute_dtype)
          - operations/modules carry over input dtype into output
          - this way compute_dtype and propagates in forward() through the stack
        - at the end we cast logits back to float32 before softmax        
        """
        assert isinstance(compute_dtype, torch.dtype)
        assert isinstance(enable_fa3, bool)
        assert isinstance(fp8_training, bool)
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.fp8_training = fp8_training

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, self._has_ve(i, config.n_layer), enable_fa3) for i in range(config.n_layer)]),
        ))
        self.lm_head = LinearFP8(config.n_embd, config.vocab_size, bias=False)

        # Params for merging x0 across network and blending residual stream
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # Value embeddings for each layer
        self.value_embeds = nn.ModuleDict({
            str(i) : nn.Embedding(config.vocab_size, config.n_embd) for i in range(config.n_layer) if self._has_ve(i, config.n_layer)
        })

        cos, sin = CausalSelfAttentionRoPE.precalculate_cos_sin(
            seq_len=config.block_size * 10,
            head_size=config.n_embd // config.n_head,
            base=100_000,
        )
        self.register_buffer("cos", cos, persistent=False)  # don't save to checkpoint
        self.register_buffer("sin", sin, persistent=False)

        # Pre-calculate window size tuples (context_length, 0) for each layer
        self.window_sizes = self._calc_window_sizes(self.config)

    def number_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Counted params do not match total params"
        moe_inactive = sum(
            block.moe.num_expert_params()['inactive'] for block in self.transformer.h if block.moe_enable
        )
        result = {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'active_transformer_matrices': transformer_matrices - moe_inactive,
            'scalars': scalars,
            'moe_inactive': moe_inactive,
            'total': total,
            'active_total': total - moe_inactive,
        }
        return result

    def _calc_window_sizes(self, config):
        long_window = config.block_size
        short_window = -(-long_window // 4 // 128) * 128  # Nearest multiple of 128 that is at least 1/4 of long_window
        chat_to_window_type = {
            'L': (long_window, 0),
            'S': (short_window, 0),
        }
        window_sizes = []
        for layer_idx in range(config.n_layer):
            window_type = config.window_pattern[layer_idx % len(config.window_pattern)]
            window_sizes.append(chat_to_window_type[window_type])
        window_sizes[-1] = (long_window, 0)  # Last layer always full attention
        return window_sizes
    
    def _has_ve(self, layer_idx, n_layer):
        # Every other layer, last always included
        return layer_idx % 2 == (n_layer-1) % 2

    def init_weights(self):
        """Initialize weights/buffers, cast RoPE/WTE/VE to compute_dtype.

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
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            if not self.config.moe_enable:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s*0.4, s*0.4)  # smaller init for feedforward
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            else:
                torch.nn.init.uniform_(block.moe.router.gate.weight, -s, s)
                torch.nn.init.uniform_(block.moe.experts.w_up, -s, s)
                torch.nn.init.zeros_(block.moe.experts.w_down)
                torch.nn.init.uniform_(block.moe.shared_expert.w_up.weight, -s, s)
                torch.nn.init.zeros_(block.moe.shared_expert.w_down.weight)
                torch.nn.init.zeros_(block.moe.router.expert_bias)
                torch.nn.init.zeros_(block.moe.router.tokens_per_expert_counter)

        # Per layer scalars
        n_layer = self.config.n_layer
        for i in range(n_layer):
            # Linearly interpolate from 1.15 to 1.05 across layers,
            # earlier layers benefit more from the sharper attention
            init_val = 1.15 - (0.10 * i / max(n_layer-1, 1))
            self.resid_lambdas.data[i] = init_val
        for i in range(n_layer):
            # Linearly interpolate from 0.2 to 0.05 across layers
            # earlier layers get more x0 blending
            init_val = 0.20 - (0.15 * i / max(n_layer-1, 1))
            self.x0_lambdas.data[i] = init_val

        # VE embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                # Zero init would mean sigmoid(0)->0.5, 2*0.5=1, i.e. neutral at the start
                # Small positive init means it is slightly above neutral at the start
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # RoPE buffers in compute dtype
        self.cos, self.sin = CausalSelfAttentionRoPE.precalculate_cos_sin(
            seq_len=self.config.block_size * 10,
            head_size=self.config.n_embd // self.config.n_head,
            base=100_000,
            device=self.transformer.wte.weight.device,
            dtype=self.compute_dtype
        )

        # Cast to compute dtype
        self.transformer.wte.to(dtype=self.compute_dtype)
        for ve in self.value_embeds.values():
            ve.to(dtype=self.compute_dtype)

    def train(self, mode=True):
        if self.fp8_training:
            fp8_mode = 'fp8' if mode else 'native'
            for module in self.modules():
                if isinstance(module, LinearFP8):
                    module.switch_mode_if_legal(fp8_mode)
        return super().train(mode)

    def update_moe_balancing(self):
        for block in self.transformer.h:
            if block.moe_enable:
                block.moe.update_expert_bias()

    def zero_moe_counters(self):
        for block in self.transformer.h:
            if block.moe_enable:
                block.moe.zero_token_counters()

    def forward(self, idx, targets=None, reduction='mean', return_logits=True):
        B, T = idx.shape
        assert T <= self.cos.size(1), "Cannot forward, model block size is exhausted."
        assert idx.device == self.cos.device, "Input device does not match model device."
        assert self.cos.dtype == self.compute_dtype, "Model buffers are not in compute_dtype."

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
