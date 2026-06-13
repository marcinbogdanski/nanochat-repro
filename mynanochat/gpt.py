import torch
import torch.nn as nn
import torch.nn.functional as F
from mynanochat.fp8 import LinearFP8
from mynanochat.moe import MoE
from mynanochat.flash_attention import sdpa_attn_func, fa3_attn_func
from mynanochat.adamw import AdamW, DistAdamW
from mynanochat.muon import Muon, DistMuon

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
    def __init__(self, config, ve_enable, enable_fa3, enable_metrics=False):
        super().__init__()
        self.attn = CausalSelfAttentionRoPE(config, ve_enable, enable_fa3)
        self.moe_enable = config.moe_enable
        self.enable_metrics = enable_metrics
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
        if self.enable_metrics:
            sq_sum_t = x.detach().float().square().sum()  # keep as tensor so we don't break graph
            num_el = x.numel()
            return x, sq_sum_t, num_el
        return x, None, None


class GPTModel(nn.Module):
    def __init__(self, config, compute_dtype, enable_fa3, fp8_training, enable_metrics=False):
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
        self.enable_metrics = enable_metrics

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([
                Block(config, self._has_ve(i, config.n_layer), enable_fa3, enable_metrics) for i in range(config.n_layer)
            ]),
        ))
        self.lm_head = LinearFP8(config.n_embd, config.vocab_size, bias=False)

        # Params for merging x0 across network and blending residual stream
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Smear gate to mix previous token's embeddings into current token (bigram-like)
        self.smear_gate = LinearFP8(24, 1, bias=False)  # used as linear only
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout, Nanochat says: "subtract cached mid-layer residual before final norm to remove low-level features"
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

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
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)  # similar to VE gate init)
        torch.nn.init.constant_(self.smear_lambda, 0.0)
        torch.nn.init.constant_(self.backout_lambda, 0.2)

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


    def collect_metrics(self, fwd_metrics, opt_metrics):
        """Collect metrics, flat list of (block, tensor_name, surface='grad', stat='sq_sum'/'num_el', value)
        
        We opt to iterate over known params explicitly, trying to iterate self.parameters() is messy. Just go through each param and collect metrics.
        """
        metrics = []  # flat list of (block, tensor_name, surface='grad', stat='sq_sum'/'num_el', value)

        # Forward activation metrics, _lt='list of tensors', _l='list', _t='tensor'
        # fwd_metrics = {
        #     'resid_post_sq_sum_lt': metrics_resid_post_sq_sum,  # list of tensors, one per block
        #     'resid_post_num_el_l': metrics_resid_post_num_el,   # list of ints, one per block
        #     'logits_sq_sum_t': metrics_logits_sq_sum,           # tensor
        #     'logits_num_el': metrics_logits_num_el,
        #     'probs_max_sum_t': metrics_probs_max_sum,           # tensor
        #     'probs_max_count': metrics_probs_max_count,
        #     'entropy_sum_t': metrics_entropy_sum,               # tensor
        #     'entropy_count': metrics_entropy_count,
        # }
        for block_n in range(self.config.n_layer):
            entry_resid_post_sq_sum = {'block': block_n, 'tensor_name': 'resid_post', 'surface': 'fwd', 'stat': 'sq_sum', 'value': 0.0}
            entry_resid_post_num_el = {'block': block_n, 'tensor_name': 'resid_post', 'surface': 'fwd', 'stat': 'num_el', 'value': 0}
            for ga_idx in range(len(fwd_metrics)):  # over grad accum steps
                if fwd_metrics[ga_idx] is not None:
                    sq_sum_t_list, num_el_list = fwd_metrics[ga_idx]['resid_post_sq_sum_lt'], fwd_metrics[ga_idx]['resid_post_num_el_l']
                    entry_resid_post_sq_sum['value'] += sq_sum_t_list[block_n].item()
                    entry_resid_post_num_el['value'] += num_el_list[block_n]
            metrics.append(entry_resid_post_sq_sum)
            metrics.append(entry_resid_post_num_el)
        entry_logits_sq_sum =     {'block': None, 'tensor_name': 'logits',     'surface': 'fwd', 'stat': 'sq_sum', 'value': 0.0}
        entry_logits_num_el =     {'block': None, 'tensor_name': 'logits',     'surface': 'fwd', 'stat': 'num_el', 'value': 0}
        entry_probs_max_sum =     {'block': None, 'tensor_name': 'probs_max',  'surface': 'fwd', 'stat': 'sum',    'value': 0.0}
        entry_probs_max_count =   {'block': None, 'tensor_name': 'probs_max',  'surface': 'fwd', 'stat': 'count',  'value': 0}
        entry_entropy_sum =       {'block': None, 'tensor_name': 'entropy',    'surface': 'fwd', 'stat': 'sum',    'value': 0.0}
        entry_entropy_count =     {'block': None, 'tensor_name': 'entropy',    'surface': 'fwd', 'stat': 'count',  'value': 0}
        for ga_idx in range(len(fwd_metrics)):
            entry_logits_sq_sum['value'] += fwd_metrics[ga_idx]['logits_sq_sum_t'].item()
            entry_logits_num_el['value'] += fwd_metrics[ga_idx]['logits_num_el']
            entry_probs_max_sum['value'] += fwd_metrics[ga_idx]['probs_max_sum_t'].item()
            entry_probs_max_count['value'] += fwd_metrics[ga_idx]['probs_max_count']
            entry_entropy_sum['value'] += fwd_metrics[ga_idx]['entropy_sum_t'].item()
            entry_entropy_count['value'] += fwd_metrics[ga_idx]['entropy_count']
        metrics.append(entry_logits_sq_sum)
        metrics.append(entry_logits_num_el)
        metrics.append(entry_probs_max_sum)
        metrics.append(entry_probs_max_count)
        metrics.append(entry_entropy_sum)
        metrics.append(entry_entropy_count)

        # Gradient, params and update metrics        
        def extend_metrics(tensor, block, tensor_name):
            sq_sum = 0.0 if tensor.grad is None else tensor.grad.detach().float().square().sum().item()
            num_el = 0 if tensor.grad is None else tensor.grad.numel()
            metrics.append({'block': block, 'tensor_name': tensor_name, 'surface': 'grad', 'stat': 'sq_sum', 'value': sq_sum})
            metrics.append({'block': block, 'tensor_name': tensor_name, 'surface': 'grad', 'stat': 'num_el', 'value': num_el})
            # opt_metrics has 3x keys: 'update_sq_sum', 'params_sq_sum', 'params_num_el'
            # currently all params are in some opt group, so no need to check if tensor is in opt_metrics
            metrics.append({'block': block, 'tensor_name': tensor_name, 'surface': 'params', 'stat': 'sq_sum', 'value': opt_metrics[tensor]['params_sq_sum']})
            metrics.append({'block': block, 'tensor_name': tensor_name, 'surface': 'params', 'stat': 'num_el', 'value': opt_metrics[tensor]['params_num_el']})
            metrics.append({'block': block, 'tensor_name': tensor_name, 'surface': 'update', 'stat': 'sq_sum', 'value': opt_metrics[tensor]['update_sq_sum']})
            metrics.append({'block': block, 'tensor_name': tensor_name, 'surface': 'update', 'stat': 'num_el', 'value': opt_metrics[tensor]['params_num_el']}) # use params_num_el

        def extend_metrics_scalars(tensor, tensor_name):
            if tensor.ndim == 0:
                value = tensor.detach().float().cpu().item()
                metrics.append({'block': None, 'tensor_name': tensor_name, 'surface': 'params', 'stat': 'value', 'value': value})
            else:
                values = tensor.detach().float().cpu().tolist()
                for i, val in enumerate(values):
                    metrics.append({'block': i, 'tensor_name': tensor_name, 'surface': 'params', 'stat': 'value', 'value': val})  # reuse block for indexing

        extend_metrics(self.transformer.wte.weight, block=None, tensor_name='wte.weight')
        extend_metrics(self.lm_head.weight, block=None, tensor_name='lm_head.weight')
        for i, block in enumerate(self.transformer.h):
            extend_metrics(block.attn.c_q.weight, block=i, tensor_name='attn.c_q.weight')
            extend_metrics(block.attn.c_k.weight, block=i, tensor_name='attn.c_k.weight')
            extend_metrics(block.attn.c_v.weight, block=i, tensor_name='attn.c_v.weight')
            extend_metrics(block.attn.c_proj.weight, block=i, tensor_name='attn.c_proj.weight')
            if not self.config.moe_enable:
                extend_metrics(block.mlp.c_fc.weight, block=i, tensor_name='mlp.c_fc.weight')
                extend_metrics(block.mlp.c_proj.weight, block=i, tensor_name='mlp.c_proj.weight')
            else:
                extend_metrics(block.moe.router.gate.weight, block=i, tensor_name='moe.router.gate.weight')
                extend_metrics(block.moe.experts.w_up, block=i, tensor_name='moe.experts.w_up')
                extend_metrics(block.moe.experts.w_down, block=i, tensor_name='moe.experts.w_down')
                extend_metrics(block.moe.shared_expert.w_up.weight, block=i, tensor_name='moe.shared_expert.w_up.weight')
                extend_metrics(block.moe.shared_expert.w_down.weight, block=i, tensor_name='moe.shared_expert.w_down.weight')
        extend_metrics_scalars(self.resid_lambdas, tensor_name='resid_lambdas')  # for scalars we just log the value, no grad/update metrics
        extend_metrics_scalars(self.x0_lambdas, tensor_name='x0_lambdas')
        extend_metrics(self.smear_gate.weight, block=None, tensor_name='smear_gate.weight')
        extend_metrics_scalars(self.smear_lambda, tensor_name='smear_lambda')
        extend_metrics_scalars(self.backout_lambda, tensor_name='backout_lambda')
        for i, ve in self.value_embeds.items():
            extend_metrics(ve.weight, block=int(i), tensor_name='value_embed.weight')
        for i, block in enumerate(self.transformer.h):
            if block.attn.ve_gate is not None:
                extend_metrics(block.attn.ve_gate.weight, block=i, tensor_name='attn.ve_gate.weight')
        return metrics


    def setup_optimizer(self, embedding_lr, matrix_lr, unembedding_lr, scalar_lr, weight_decay, enable_metrics=False):
        """Prepare param groups and setup optimizers. Scale learning rates based on parameter counts"""
        ddp = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1

        # Separate parameters into groups for different optimizers and learning rates
        params_matrix = list(self.transformer.h.parameters())
        params_embedding = list(self.transformer.wte.parameters())
        params_val_embds = list(self.value_embeds.parameters())
        params_lm_head = list(self.lm_head.parameters())
        params_resid = [self.resid_lambdas]
        params_x0 = [self.x0_lambdas]
        smear_backout_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(params_matrix) + len(params_embedding) + len(params_val_embds) + len(params_lm_head) + len(params_resid) + len(params_x0) + len(smear_backout_params)

        # Apply learning rate scaling based on parameter counts, similar to Chinchilla scaling
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5

        # AdamW for dense params
        adam_groups = [
            dict(params=params_lm_head, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), weight_decay=0.01, is_small=False),
            dict(params=params_embedding, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), weight_decay=0.001, is_small=False),
            dict(params=params_val_embds, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), weight_decay=0.01, is_small=False),
            dict(params=params_resid, lr=scalar_lr * 0.01, betas=(0.8, 0.95), weight_decay=0.05, is_small=True),
            dict(params=params_x0, lr=scalar_lr, betas=(0.96, 0.95), weight_decay=0.0, is_small=True),
            dict(params=smear_backout_params, lr=0.2, betas=(0.8, 0.95), weight_decay=0.0, is_small=True),
        ]
        adamw_factory = DistAdamW if ddp else AdamW
        adamw_optimizer = adamw_factory(adam_groups, eps=1e-10, weight_decay=0.0, enable_metrics=enable_metrics)

        # Muon for large matrix params
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
            beta2=0.9,
            weight_decay=weight_decay,
            compute_dtype=self.compute_dtype,
            enable_metrics=enable_metrics,
        )
        
        # Set initial_lr in param groups for proper LR scaling
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
                for group in opt.param_groups:
                    group["initial_lr"] = group["lr"]
        
        # [0] is AdamW, [1] is Muon
        return optimizers


    def estimate_flops_per_token(self):
        """Estimate FLOPs per token, for the forward and backward pass."""
        # Param Matmuls
        # each matmul is: 2 flops forward per param (multiply and add), 2 matmuls per backward (4 flops), total 2+4=6
        num_params = self.number_scaling_params()
        matmul_flops = 6 * (num_params['transformer_active'] + num_params['lm_head'])
        # Attention FLOPs
        # Two extra fwd matmuls in attention (Q @ K.T and attn @ V) - 2 * 6 = 12
        attn_flops = 0
        head_size = self.config.n_embd // self.config.n_head
        for layer_idx in range(self.config.n_layer):
            window_size, _ = self.window_sizes[layer_idx]  # (left, right), we only use left for causal attention
            effective_seq_len = min(window_size, self.config.block_size)
            attn_flops += 12 * self.config.n_head * effective_seq_len * head_size
        return matmul_flops + attn_flops


    def number_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Counted params do not match total params"
        moe_inactive = sum(
            block.moe.num_expert_params()['inactive'] for block in self.transformer.h if block.moe_enable
        )
        result = {
            'wte': wte,                            # word token embedding
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_all': transformer_matrices,
            'transformer_active': transformer_matrices - moe_inactive,
            'transformer_inactive': moe_inactive,
            'scalars': scalars,
            'total': total,
            'total_active': total - moe_inactive,
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

    def _apply_smear(self, x):
        """Mix previous token's embedding into current token (bigram-like)"""
        x_gate_input = x[:, 1:, :24]                                   # B,T-1,24
        x_score = torch.sigmoid(self.smear_gate(x_gate_input))         # B,T-1,1
        x_score = self.smear_lambda.to(x.dtype) * x_score              # B,T-1,1
        outputs = torch.cat([x[:,:1], x[:,1:] + x_score*x[:,:-1]], dim=1)
        assert outputs.dtype == x.dtype        
        return outputs

    def forward(self, idx, targets=None, reduction='mean', return_logits=True):
        B, T = idx.shape
        assert T <= self.cos.size(1), "Cannot forward, model block size is exhausted."
        assert idx.device == self.cos.device, "Input device does not match model device."
        assert self.cos.dtype == self.compute_dtype, "Model buffers are not in compute_dtype."

        # Embeddings
        x = self.transformer.wte(idx)             # B,T,E <- B,T
        x = F.rms_norm(x, (x.size(-1),))

        # Smear
        x = self._apply_smear(x)

        # Transformer
        x0 = x
        backout_layer = self.config.n_layer // 2  # backout in middle of network
        x_backout = None
        if self.enable_metrics:
            metrics_resid_post_sq_sum, metrics_resid_post_num_el = [], []
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if self._has_ve(i, self.config.n_layer) else None
            x, sq_sum_t, num_el = block(x, ve, self.cos, self.sin, self.window_sizes[i])
            if i == backout_layer:
                x_backout = x
            if self.enable_metrics:
                metrics_resid_post_sq_sum.append(sq_sum_t)
                metrics_resid_post_num_el.append(num_el)

        # Final backout blending
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = F.rms_norm(x, (x.size(-1),))

        # Logits
        softcap = 15
        logits = self.lm_head(x)   # B,T,V <- B,T,E
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        metrics = None
        if self.enable_metrics:
            metrics_logits_sq_sum = logits.detach().float().square().sum()  # .item() here would break the graph
            metrics_logits_num_el = logits.numel()
            probs = F.softmax(logits.detach(), dim=-1)
            probs_max = probs.amax(dim=-1)
            metrics_probs_max_sum = probs_max.sum()
            metrics_probs_count = probs_max.numel()
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            metrics_entropy_sum = entropy.sum()
            metrics_entropy_count = entropy.numel()
            metrics = {
                'resid_post_sq_sum_lt': metrics_resid_post_sq_sum,  # _lt = list of tensors, one per block
                'resid_post_num_el_l': metrics_resid_post_num_el,   # _l = list of ints, one per block
                'logits_sq_sum_t': metrics_logits_sq_sum,           # _t = tensor
                'logits_num_el': metrics_logits_num_el,
                'probs_max_sum_t': metrics_probs_max_sum,           # _t = tensor
                'probs_max_count': metrics_probs_count,
                'entropy_sum_t': metrics_entropy_sum,               # _t = tensor
                'entropy_count': metrics_entropy_count,
            }

        if targets is None:
            assert return_logits, "If targets is None, return_logits must be True."
            return logits, None, metrics
        else:
            B, T, C = logits.shape
            logits_ = logits.view(B*T, C)  # B*T, C
            targets_ = targets.view(B*T)   # B*T
            loss = F.cross_entropy(logits_, targets_, reduction=reduction)
            if return_logits:
                return logits, loss, metrics
            else:
                return None, loss, metrics
