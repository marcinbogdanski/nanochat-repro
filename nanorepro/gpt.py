from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanorepro.fp8 import LinearFP8
from nanorepro.moe import MoE
from nanorepro.flash_attention import sdpa_attn_func, fa3_attn_func, sdpa_attn_with_kvcache, fa3_attn_with_kvcache
from nanorepro.adamw import AdamW, DistAdamW
from nanorepro.muon import Muon, DistMuon
from nanorepro.backward_scheduler import BackwardScheduler
from nanorepro.nsight_trace import clone_boundary

class GPTConfig:
    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd, window_pattern, moe_enable, moe_experts, moe_top_k):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.window_pattern = window_pattern
        self.moe_enable = moe_enable
        self.moe_experts = moe_experts
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
            'moe_experts': self.moe_experts,
            'moe_top_k': self.moe_top_k,
        }

class CausalSelfAttentionRoPE(nn.Module):
    """Multiple self-attention heads"""
    def __init__(self, config, layer_idx, ve_enable, enable_fa):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert isinstance(enable_fa, bool)
        self.n_head = config.n_head
        self.block_size = config.block_size
        self.layer_idx = layer_idx
        self.enable_fa = enable_fa

        self.c_q = LinearFP8(config.n_embd, config.n_embd, bias=False)
        self.c_k = LinearFP8(config.n_embd, config.n_embd, bias=False)
        self.c_v = LinearFP8(config.n_embd, config.n_embd, bias=False)
        self.c_proj = LinearFP8(config.n_embd, config.n_embd, bias=False)

        # VE Gate
        self.ve_gate_size = 12
        self.ve_gate = LinearFP8(self.ve_gate_size, config.n_head, bias=False) if ve_enable else None

    def _apply_rope(self, q, cos, sin):
        B, T, nh, hs = q.size()
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

    def forward(self, x, ve, cos, sin, window_size, kv_cache):
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

        if self.enable_fa:
            # Flash Attention
            if kv_cache is None:
                y = fa3_attn_func(q_rot, k_rot, v, causal=True, window_size=window_size)
            else:
                y = fa3_attn_with_kvcache(
                    q=q_rot,
                    k_cache=kv_cache.k_cache[self.layer_idx],  # writes in-place
                    v_cache=kv_cache.v_cache[self.layer_idx],  # writes in-place
                    k=k_rot,
                    v=v,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    window_size=window_size
                )
        else:
            # SDPA fallback
            if kv_cache is None:
                y = sdpa_attn_func(q_rot, k_rot, v, causal=True, window_size=window_size)
            else:
                y = sdpa_attn_with_kvcache(
                    q=q_rot,
                    k_cache=kv_cache.k_cache[self.layer_idx],  # writes in-place
                    v_cache=kv_cache.v_cache[self.layer_idx],  # writes in-place
                    k=k_rot,
                    v=v,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    window_size=window_size
                )

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
    def __init__(self, config, layer_idx, ve_enable, enable_fa, enable_metrics=False):
        super().__init__()
        self.attn = CausalSelfAttentionRoPE(config, layer_idx, ve_enable, enable_fa)
        self.enable_metrics = enable_metrics
        if not config.moe_enable:
            self.mlp = MLP(config)
        else:
            # keep the same interface as MLP for the forward pass
            self.mlp = MoE(dim=config.n_embd, n_routed_experts=config.moe_experts, top_k=config.moe_top_k)

    def _norm(self, x):
        return F.rms_norm(x, (x.size(-1),))

    def forward(self, x, ve, cos, sin, window_size, kv_cache):
        x = x + self.attn(self._norm(x), ve, cos, sin, window_size, kv_cache)        # B,T,E pre-norm
        x = x + self.mlp(self._norm(x))  # MoE, if enabled
        if self.enable_metrics:
            sq_sum_t = x.detach().float().square().sum()  # keep as tensor so we don't break graph
            num_el = x.numel()
            return x, sq_sum_t, num_el
        return x, None, None


class GPTModel(nn.Module):
    def __init__(self, config, compute_dtype, enable_fa, fp8_training, enable_metrics=False):
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
        assert isinstance(enable_fa, bool)
        assert isinstance(fp8_training, bool)
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.fp8_training = fp8_training
        self.enable_metrics = enable_metrics

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([
                Block(config, i, self._has_ve(i, config.n_layer), enable_fa, enable_metrics) for i in range(config.n_layer)
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

        # Compiled regions, if used
        self._compiled_layer_regions = None  # split regions, use on last grad_accum only (no point splitting if no comms to overlap)
        self._compiled_output_region = None
        self._compiled_whole_transformer_region = None

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
            if isinstance(block.mlp, MLP):
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s*0.4, s*0.4)  # smaller init for feedforward
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            elif isinstance(block.mlp, MoE):
                torch.nn.init.uniform_(block.mlp.router.gate.weight, -s, s)
                torch.nn.init.uniform_(block.mlp.experts.w_up, -s*0.4, s*0.4)
                torch.nn.init.zeros_(block.mlp.experts.w_down)
                torch.nn.init.uniform_(block.mlp.shared_expert.w_up.weight, -s*0.4, s*0.4)
                torch.nn.init.zeros_(block.mlp.shared_expert.w_down.weight)
                torch.nn.init.zeros_(block.mlp.router.expert_bias)
                torch.nn.init.zeros_(block.mlp.router.tokens_per_expert_counter)
            else:
                raise ValueError(f"Unknown MLP type: {type(block.mlp)}")

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

        The main metrics we want to reconstruct from the data we collect are RMS of fwd activations, gradients and param updates.
        Note that code below collects only sq_sum and num_el for the elements represented by this rank only. There is an additional
        step required later in post processing to combine the metrics across the ranks to get the RMS across all elements.
        This is possible because RMS is calculated as `sqrt(sum(sq(all_elements_across_all_ranks)) / num_el(all_elements_across_all_ranks))`
        Which can be decomposed into `sqrt(sum_across_ranks(sum(sq(all_elements_on_this_rank))) / sum_across_ranks(num_el(all_elements_on_this_rank)))`
        Since `sq_sum = sum(sq(all_elements_on_this_rank))` and `num_el = num_el(all_elements_on_this_rank)`, we can later sum the sq_sum and num_el
        across ranks in post-processing to get the global RMS.
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
            # opt_metrics has 4x keys: 'grad_sq_sum', 'update_sq_sum', 'params_sq_sum', 'params_num_el'
            # currently all params are in some opt group, so no need to check if tensor is in opt_metrics
            metrics.append({'block': block, 'tensor_name': tensor_name, 'surface': 'grad', 'stat': 'sq_sum', 'value': opt_metrics[tensor]['grad_sq_sum']})
            metrics.append({'block': block, 'tensor_name': tensor_name, 'surface': 'grad', 'stat': 'num_el', 'value': opt_metrics[tensor]['params_num_el']})
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
            if isinstance(block.mlp, MLP):
                extend_metrics(block.mlp.c_fc.weight, block=i, tensor_name='mlp.c_fc.weight')
                extend_metrics(block.mlp.c_proj.weight, block=i, tensor_name='mlp.c_proj.weight')
            elif isinstance(block.mlp, MoE):
                extend_metrics(block.mlp.router.gate.weight, block=i, tensor_name='moe.router.gate.weight')
                extend_metrics(block.mlp.experts.w_up, block=i, tensor_name='moe.experts.w_up')
                extend_metrics(block.mlp.experts.w_down, block=i, tensor_name='moe.experts.w_down')
                extend_metrics(block.mlp.shared_expert.w_up.weight, block=i, tensor_name='moe.shared_expert.w_up.weight')
                extend_metrics(block.mlp.shared_expert.w_down.weight, block=i, tensor_name='moe.shared_expert.w_down.weight')
            else:
                raise ValueError(f"Unknown MLP type: {type(block.mlp)}")
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


    def _build_backward_collectives(self, params_matrix, muon_params_per_bucket):
        """Manually build comms buckets in backward pass order."""

        # Construct the list of non-small params in backward order.
        # I omit small params because as they are.. well, small and i expect not much to gain from backward overlapping them. Did not test.
        backward_params = [self.lm_head.weight]
        for layer_idx in reversed(range(self.config.n_layer)):
            block = self.transformer.h[layer_idx]
            if isinstance(block.mlp, MLP):
                backward_params.extend([block.mlp.c_proj.weight, block.mlp.c_fc.weight])
            elif isinstance(block.mlp, MoE):
                backward_params.extend([
                    block.mlp.shared_expert.w_down.weight,
                    block.mlp.experts.w_down,
                    block.mlp.shared_expert.w_up.weight,
                    block.mlp.experts.w_up,
                    block.mlp.router.gate.weight,
                ])
            else:
                raise ValueError(f"Unknown MLP type: {type(block.mlp)}")
            backward_params.extend([
                block.attn.c_proj.weight,
                block.attn.c_v.weight,
                block.attn.c_k.weight,
                block.attn.c_q.weight,
            ])
            if block.attn.ve_gate is not None:
                backward_params.append(block.attn.ve_gate.weight)
            if str(layer_idx) in self.value_embeds:
                backward_params.append(self.value_embeds[str(layer_idx)].weight)
        backward_params.append(self.transformer.wte.weight)

        params_matrix = set(params_matrix)
        muon_params_in_bwd_order = [p for p in backward_params if p in params_matrix]
        muon_buckets_by_last_param = {}
        for shape in sorted({p.shape for p in muon_params_in_bwd_order}):
            shape_params = [p for p in muon_params_in_bwd_order if p.shape == shape]
            if muon_params_per_bucket != -1:
                for i in range(0, len(shape_params), muon_params_per_bucket):
                    bucket_params = shape_params[i:i + muon_params_per_bucket]
                    muon_buckets_by_last_param[bucket_params[-1]] = bucket_params
            else:
                muon_buckets_by_last_param[shape_params[-1]] = shape_params

        collectives = []
        for param in backward_params:
            if param not in params_matrix:
                collectives.append(('adamw', [param]))
            elif param in muon_buckets_by_last_param:
                collectives.append(('muon', muon_buckets_by_last_param[param]))

        return collectives


    def setup_optimizer(self, embedding_lr, matrix_lr, unembedding_lr, scalar_lr, router_lr, smear_backout_lr, weight_decay, backward_overlap=False, muon_params_per_bucket=-1, enable_metrics=False):
        """Prepare param groups and setup optimizers. Scale learning rates based on parameter counts"""
        ddp = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        world_size = torch.distributed.get_world_size() if ddp else 1
        assert muon_params_per_bucket == -1 or (muon_params_per_bucket > 0 and muon_params_per_bucket % world_size == 0)  # must be divisible by world_size

        # Separate parameters into groups for different optimizers and learning rates
        params_matrix = list(self.transformer.h.parameters())
        params_embedding = list(self.transformer.wte.parameters())
        params_val_embds = list(self.value_embeds.parameters())
        params_lm_head = list(self.lm_head.parameters())
        params_resid = [self.resid_lambdas]
        params_x0 = [self.x0_lambdas]
        smear_backout_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        # Move MoE router params from params_matrix to a separate AdamW group (if MoE disabled, params_router will be empty)
        params_router = [block.mlp.router.gate.weight for block in self.transformer.h if isinstance(block.mlp, MoE)]
        router_params_ids = {id(p) for p in params_router}
        params_matrix = [p for p in params_matrix if id(p) not in router_params_ids]
        assert len(list(self.parameters())) == len(params_matrix) + len(params_embedding) + len(params_val_embds) + len(params_lm_head) + len(params_resid) + len(params_x0) + len(smear_backout_params) + len(params_router)

        # Apply learning rate scaling based on parameter counts, similar to Chinchilla scaling
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5

        # AdamW for dense params
        adam_groups = [
            dict(params=params_lm_head, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), weight_decay=0.01, is_small=False),
            dict(params=params_embedding, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), weight_decay=0.001, is_small=False),
            dict(params=params_val_embds, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), weight_decay=0.01, is_small=False),
            dict(params=params_resid, lr=scalar_lr * 0.01, betas=(0.8, 0.95), weight_decay=0.05, is_small=True),
            dict(params=params_x0, lr=scalar_lr, betas=(0.96, 0.95), weight_decay=0.0, is_small=True),
            dict(params=smear_backout_params, lr=smear_backout_lr, betas=(0.8, 0.95), weight_decay=0.0, is_small=True),
        ]
        if params_router:
            # No weight decay for MoE to prevent drift towards sigmoid(0.0)=0.5
            adam_groups.append(dict(params=params_router, lr=router_lr * dmodel_lr_scale, betas=(0.8, 0.96), weight_decay=0.0, is_small=False))
        adamw_factory = DistAdamW if ddp else AdamW
        adamw_optimizer = adamw_factory(adam_groups, eps=1e-10, weight_decay=0.0, enable_metrics=enable_metrics)

        # Muon for large matrix params
        # backward_collectives = [('adamw', [param]), ('muon', [param, param], ...]
        backward_collectives = self._build_backward_collectives(params_matrix, muon_params_per_bucket)
        scheduled_params = [param for _, bucket_params in backward_collectives for param in bucket_params]
        expected_params = params_matrix + params_lm_head + params_embedding + params_val_embds + params_router
        assert len(scheduled_params) == len(expected_params) and set(scheduled_params) == set(expected_params)

        # Muon Groups
        if muon_params_per_bucket == -1:
            muon_groups = []
            for shape in sorted({p.shape for p in params_matrix}):
                group_params = [p for p in params_matrix if p.shape == shape]
                muon_groups.append({'params': group_params})
        else:
            muon_groups = [{'params': group_params} for optim_type, group_params in backward_collectives if optim_type == 'muon']  # new way

        # Muon Optimizer
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

        comm_launchers = []
        if ddp:
            adamw_param_to_idx = {param: bucket_idx for bucket_idx, (group, param) in enumerate(adamw_optimizer.buckets)}
            muon_param_to_idx = {param: group_idx for group_idx, group in enumerate(muon_optimizer.param_groups) for param in group['params']}
            for param_bucket in backward_collectives:
                optim_type, bucket_params = param_bucket
                if optim_type == 'adamw':
                    param = bucket_params[0]  # single param per adamw bucket
                    bucket_idx = adamw_param_to_idx[param]
                    launcher = partial(adamw_optimizer.launch_reduce, bucket_idx)
                    comm_launchers.append(launcher)
                elif optim_type == 'muon':
                    group_idx = muon_param_to_idx[bucket_params[0]]
                    launcher = partial(muon_optimizer.launch_reduce, group_idx)
                    comm_launchers.append(launcher)

        param_buckets = [bucket_params for _, bucket_params in backward_collectives]
        backward_scheduler = BackwardScheduler(
            param_buckets=param_buckets,
            comm_launchers=comm_launchers,
            backward_overlap=backward_overlap and ddp,  # whole class becomes no-op if False
        )
        
        # Set initial_lr in param groups for proper LR scaling
        for opt in [adamw_optimizer, muon_optimizer]:
                for group in opt.param_groups:
                    group["initial_lr"] = group["lr"]

        return adamw_optimizer, muon_optimizer, backward_scheduler


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
            block.mlp.num_expert_params()['inactive'] for block in self.transformer.h if isinstance(block.mlp, MoE)
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
            if isinstance(block.mlp, MoE):
                block.mlp.update_expert_bias()

    def zero_moe_counters(self):
        for block in self.transformer.h:
            if isinstance(block.mlp, MoE):
                block.mlp.zero_token_counters()

    def _apply_smear(self, x, kv_cache):
        """Mix previous token's embedding into current token (bigram-like)"""
        B, T, E = x.shape
        if kv_cache is None:
            # No KV cache path
            x_gate_input = x[:, 1:, :24]                                   # B,T-1,24
            x_score = torch.sigmoid(self.smear_gate(x_gate_input))         # B,T-1,1
            x_score = self.smear_lambda.to(x.dtype) * x_score              # B,T-1,1
            outputs = torch.cat([x[:,:1], x[:,1:] + x_score*x[:,:-1]], dim=1)
        else:
            # KV Cache path
            if kv_cache.previous_embd is None:
                # Prefill
                x_gate_input = x[:, 1:, :24]                                    # B,T-1,24
                x_score = torch.sigmoid(self.smear_gate(x_gate_input))          # B,T-1,1
                x_score = self.smear_lambda.to(x.dtype) * x_score               # B,T-1,1
                outputs = torch.cat([x[:,:1], x[:,1:] + x_score*x[:,:-1]], dim=1)
                kv_cache.previous_embd = x[:,-1:,:]   # B,1,E, store for later, view ok since no grads in inference mode
            else:
                # Generation
                assert T==1
                x_gate_input = x[:, :, :24]                                     # B,1,24
                x_score = torch.sigmoid(self.smear_gate(x_gate_input))          # B,1,1
                x_score = self.smear_lambda.to(x.dtype) * x_score               # B,1,1
                outputs = x + x_score * kv_cache.previous_embd
                kv_cache.previous_embd = x            # B,1,E, store for later, view ok since no grads in inference mode
        return outputs

    def get_device(self):
        return next(self.parameters()).device

    def compile_layer_regions(self, layers_per_region):
        """Compile the transformer layers into regions for optimized execution.

        This by itself does not switch on compiled path, just makes it available. To enable it, pass use_compiled_if_available=True to forward()
        - enable: training and evaluation with stable input/target shapes, e.g. training froward and BPB
        - don't enable: variable-shape inference, e.g. CORE, sampling, free-form generation
        It is completely safe to compile regions and not use them.
        """
        assert layers_per_region == -1 or layers_per_region > 0

        if layers_per_region == -1:
            layers_per_region = len(self.transformer.h)

        num_region_variants = 0

        # Split transformer into compiled regions
        self._compiled_layer_regions = []
        for start_layer in range(0, len(self.transformer.h), layers_per_region):
            end_layer = min(start_layer + layers_per_region, len(self.transformer.h))
            region = partial(self._fwd_layer_regions, start_layer, end_layer)
            compiled_region = torch.compile(region, dynamic=False)
            self._compiled_layer_regions.append(compiled_region)
            num_region_variants += 1

        # Compile the whole transformer as one region
        if len(self._compiled_layer_regions) == 1:
            self._compiled_whole_transformer_region = self._compiled_layer_regions[0]
        else:
            transformer_region = partial(self._fwd_layer_regions, 0, len(self.transformer.h))
            self._compiled_whole_transformer_region = torch.compile(transformer_region, dynamic=False)
            num_region_variants += 1

        # Compile the output region
        self._compiled_output_region = torch.compile(self._fwd_output_region, dynamic=False)
        num_region_variants += 1

        # Raise dynamo limit, required on PyTorch 2.9
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, num_region_variants)

    def forward(self, idx, targets=None, kv_cache=None, reduction='mean', return_logits=True, use_compiled_if_available=False, split_compiled_regions=False):
        B, T = idx.shape
        assert T <= self.cos.size(1), "Cannot forward, model block size is exhausted."
        assert idx.device == self.cos.device, "Input device does not match model device."
        assert self.cos.dtype == self.compute_dtype, "Model buffers are not in compute_dtype."
        assert kv_cache is None or not torch.is_grad_enabled()     # if kv_cache, then ensure no_grad (training with KV cache not supported)
        assert kv_cache is None or not use_compiled_if_available   # if kv_cache, then ensure no compiled regions (compiled path doesn't support KV caching)
        use_compiled_regions = use_compiled_if_available and self._compiled_layer_regions is not None

        # Offset sin/cos
        offset = 0 if kv_cache is None else kv_cache.cache_seqlens[0].item()  # assume seqlens are equal across the batch
        cos = self.cos[:, offset:offset+T, :, :]
        sin = self.sin[:, offset:offset+T, :, :]

        # Setup
        x = x0 = x_backout = None
        metrics_resid_post_sq_sum, metrics_resid_post_num_el = [], []

        # Transformer
        if use_compiled_regions:
            compiled_regions = self._compiled_layer_regions if split_compiled_regions else [self._compiled_whole_transformer_region]
            for region in compiled_regions:
                x, x0, x_backout, sq_sum_list, num_el_list = region(idx, x, x0, x_backout, cos, sin, kv_cache)
                if self.enable_metrics:
                    metrics_resid_post_sq_sum.extend(sq_sum_list)
                    metrics_resid_post_num_el.extend(num_el_list)
        else:
            x, x0, x_backout, sq_sum_list, num_el_list = self._fwd_layer_regions(0, len(self.transformer.h), idx, x, x0, x_backout, cos, sin, kv_cache)
            if self.enable_metrics:
                metrics_resid_post_sq_sum.extend(sq_sum_list)
                metrics_resid_post_num_el.extend(num_el_list)

        # Advance kv_cache seqlens
        if kv_cache is not None:
            kv_cache.cache_seqlens.add_(T)

        if use_compiled_regions:
            return self._compiled_output_region(x, x_backout, targets, reduction, return_logits, metrics_resid_post_sq_sum, metrics_resid_post_num_el)
        return self._fwd_output_region(x, x_backout, targets, reduction, return_logits, metrics_resid_post_sq_sum, metrics_resid_post_num_el)

    def _fwd_layer_regions(self, start_layer, end_layer, idx, x, x0, x_backout, cos, sin, kv_cache):
        """Forward a region of multiple transformer layers, from start_layer to end_layer (exclusive)."""

        # Embeddings, Smear
        if start_layer == 0:
            x = self.transformer.wte(idx)             # B,T,E <- B,T
            x = F.rms_norm(x, (x.size(-1),))
            x = self._apply_smear(x, kv_cache)  # smear
            x = clone_boundary(x, left=None, right="block_0")     # mark start of block_0
            x0 = x

        # Iterate Transformer Layers
        sq_sum_list, num_el_list = [], []
        for i in range(start_layer, end_layer):
            # Residuals and value embeddings
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if self._has_ve(i, self.config.n_layer) else None

            # Transformer Block
            x, sq_sum_t, num_el = self.transformer.h[i](x, ve, cos, sin, self.window_sizes[i], kv_cache)
            sq_sum_list.append(sq_sum_t)
            num_el_list.append(num_el)

            # Mark end of block_{i} and start of block_{i+1} (apart from last)
            is_last_block = (i == len(self.transformer.h) - 1)
            right = f"block_{i+1}" if is_last_block is False else "output"
            x = clone_boundary(x, left=f"block_{i}", right=right)

            # Backout in the middle of the network:
            if i == self.config.n_layer // 2:
                x_backout = x

        return x, x0, x_backout, sq_sum_list, num_el_list

    def _fwd_output_region(self, x, x_backout, targets, reduction, return_logits, metrics_resid_post_sq_sum, metrics_resid_post_num_el):
        """Forward the output region of the transformer, return logits/loss/metrics as requested."""

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
            loss = F.cross_entropy(logits_, targets_, ignore_index=-1, reduction=reduction)
            loss = clone_boundary(loss, left="output", right=None)  # only used in training, so it's ok not to cover 'targets is None' case
            if return_logits:
                return logits, loss, metrics
            else:
                return None, loss, metrics

    @torch.inference_mode()
    def sample_one_token(self, logits, temperature=1.0, top_k=None, sample_rng=None):
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
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None, sample_rng=None):
        """Generate max_tokens starting from idx[B,T]"""
        assert isinstance(idx, torch.Tensor)
        assert idx.dtype == torch.long
        assert len(idx.shape) == 2  # B,T
        assert isinstance(max_new_tokens, int)
        
        is_training = self.training
        self.eval()

        block_size = self.config.block_size
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_tail = idx[:, -block_size:]      # B,T  sliding window
                logits, _, _ = self(idx_tail, use_compiled_if_available=False)      # B,T,C <- B,T  don't use compiled, shapes change
                logits = logits[:, -1, :]            # B,C <- B,T,C  discard all but last
                xcol = self.sample_one_token(logits, temperature=temperature, top_k=top_k, sample_rng=sample_rng)  # B,1
                idx = torch.cat((idx, xcol), dim=1)  # B,T+1  append
        
        self.train(is_training)
        return idx
