"""Mixture of Experts (MoE) layer for MyNanochat

This is a drop-in replacement for MLP layer in transformer blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, dim, n_routed_experts, top_k):
        """Initialize the MoE layer.
        
        Args:
            dim: Input and output dimension of the MoE layer.
            n_routed_experts: Number of routed experts (not counting the shared expert).
            top_k: Subset of routed experts to use for each token.
        """
        super().__init__()
        C, E, K = dim, n_routed_experts, top_k
        self.K = K
        active_experts = 1 + K  # 1 shared expert + K routed experts
        # Router - keep names consistent with Nanochat for save compatibility
        self.router = nn.Module()
        self.router.gate = nn.Linear(C, E, bias=False)
        self.router.register_buffer('expert_bias', torch.zeros(E))
        self.router.register_buffer('tokens_per_expert_counter', torch.zeros(E))
        # Routed experts
        self.experts = nn.Module()
        hidden_dim = round(C*4/active_experts/128) * 128  # Nearest multiple of 128
        self.experts.w_up = nn.Parameter(torch.randn(E, hidden_dim, C)*C**-0.5)
        self.experts.w_down = nn.Parameter(torch.zeros(E, C, hidden_dim))
        # Shared expert
        self.shared_expert = nn.Module()
        self.shared_expert.w_up = nn.Linear(C, hidden_dim, bias=False)
        self.shared_expert.w_down = nn.Linear(hidden_dim, C, bias=False)

    def num_expert_params(self):
        """Number of parameters in the experts, total, active, and inactive."""
        E = self.experts.w_up.size(0)
        single_expert_params = self.shared_expert.w_up.weight.numel() + self.shared_expert.w_down.weight.numel()
        expert_params_active = (1 + self.K) * single_expert_params
        expert_params_total = (1 + E) * single_expert_params
        expert_params_inactive = expert_params_total - expert_params_active
        return {
            'total': expert_params_total,
            'active': expert_params_active,
            'inactive': expert_params_inactive,
        }
    
    @torch.compiler.disable  # Dynamic slicing breaks the torch.compile
    def _exec_experts_loop(self, x_flat_sorted_weighted, sel_experts_flat, x_dtype):
        """Execute the experts using a loop - fallback when torch._grouped_mm is not available (e.g. on CPU)"""
        start_idx = 0
        outs = []
        E = self.experts.w_up.size(0)
        for i in range(E):
            num_expert = (sel_experts_flat==i).sum().item()
            end_idx = start_idx + num_expert
            h = x_flat_sorted_weighted[start_idx:end_idx] @ self.experts.w_up[i].T
            z = F.relu(h).square()
            o = z @ self.experts.w_down[i].T
            outs.append(o)
            start_idx += num_expert
        out_flat_stacked_flat_sorted = torch.cat(outs)   # B*T*K, C
        return out_flat_stacked_flat_sorted

    def _exec_experts_grouped_mm(self, x_flat_sorted_weighted, sel_experts_flat, x_dtype):
        """Execute the experts using grouped matrix multiplication - CUDA only"""
        E = self.experts.w_up.size(0)
        expert_ids = torch.arange(E, device=sel_experts_flat.device).unsqueeze(-1)
        expert_mask = expert_ids == sel_experts_flat.unsqueeze(0)
        expert_offsets = expert_mask.sum(dim=1).cumsum(dim=0).to(torch.int32)  # E

        expert_w_up = self.experts.w_up
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_gpu_dtype()
            x_flat_sorted_weighted = x_flat_sorted_weighted.to(autocast_dtype)
            expert_w_up = self.experts.w_up.to(autocast_dtype)
        h_experts = torch._grouped_mm(
            input=x_flat_sorted_weighted,
            mat2=expert_w_up.mT,
            offs=expert_offsets,
        )
        z_experts = F.relu(h_experts).square()
        expert_w_down = self.experts.w_down
        if torch.is_autocast_enabled():
            z_experts = z_experts.to(autocast_dtype)
            expert_w_down = self.experts.w_down.to(autocast_dtype)
        out_experts = torch._grouped_mm(
            input=z_experts,
            mat2=expert_w_down.mT,
            offs=expert_offsets,
        )
        out_flat_stacked_flat_sorted = out_experts.to(x_dtype)
        return out_flat_stacked_flat_sorted

    def forward(self, x):
        B, T, C = x.shape
        K = self.K
        E = self.experts.w_up.size(0)
        x_flat = x.reshape(-1, C)          # B*T, C

        # Routed expert path start
        # Bias the expert selection, but *not* weighting (Nanochat, DeepSeekV3)
        logits = self.router.gate(x_flat)       # B*T, E
        scores = torch.sigmoid(logits.float())    # B*T, E
        scores_biased = scores + self.router.expert_bias   # B*T, E
        _, sel_experts = torch.topk(scores_biased, K, dim=-1, sorted=False)   # B*T, K
        sel_scores = torch.gather(scores, dim=-1, index=sel_experts)   # B*T, K
        sel_experts_flat = sel_experts.reshape(-1)                   # B*T*K
        sel_experts_flat_sorted_idx = torch.argsort(sel_experts_flat, stable=True)  # B*T*K
        # vv
        # There are two equivalent formulations, the simpler one required duplicating the x_flat
        # Te second formulation we are actually using is slightly better on memory
        # x_flat_stacked = torch.stack([x_flat]*K, dim=1)      # B*T, K, C
        # x_flat_stacked_flat = x_flat_stacked.reshape(-1, C)  # B*T*K, C
        # x_flat_sorted = x_flat_stacked_flat[sel_experts_flat_sorted_idx]  # B*T*K, C
        # --
        token_ids = sel_experts_flat_sorted_idx // K         # B*T*K
        x_flat_sorted = x_flat[token_ids]       # B*T*K, C
        # ^^
        values_flat = sel_scores.reshape(-1)                     # B*T*K
        values_flat_sorted = values_flat[sel_experts_flat_sorted_idx]  # B*T*K
        x_flat_sorted_weighted = x_flat_sorted.float() * values_flat_sorted.unsqueeze(-1)  # B*T*K, C
        x_flat_sorted_weighted = x_flat_sorted_weighted.to(x.dtype)

        # Shared expert path
        h_shared = self.shared_expert.w_up(x_flat)
        z_shared = F.relu(h_shared).square()
        out_flat_shared = self.shared_expert.w_down(z_shared)

        # Update expert token counts for load balancing
        # This probably should be disabled during evaluation
        expert_ids = torch.arange(E, device=sel_experts.device)
        indices_column = sel_experts_flat.unsqueeze(1)                           # B*T*K, 1
        num_tokens_per_expert = (indices_column == expert_ids).sum(dim=0)
        num_tokens_per_expert = num_tokens_per_expert.to(self.router.tokens_per_expert_counter.dtype)
        self.router.tokens_per_expert_counter += num_tokens_per_expert
        
        if x.is_cuda:
            out_flat_sorted = self._exec_experts_grouped_mm(x_flat_sorted_weighted, sel_experts_flat, x.dtype)
        else:
            out_flat_sorted = self._exec_experts_loop(x_flat_sorted_weighted, sel_experts_flat, x.dtype)

        # Combine routed experts and project back
        out_flat_stacked_flat = torch.zeros(B*T*K, C, device=x.device, dtype=out_flat_sorted.dtype)
        out_flat_stacked_flat[sel_experts_flat_sorted_idx] = out_flat_sorted   # B*T*K, C
        out_flat_stacked = out_flat_stacked_flat.reshape(B*T, K, C)
        out_flat = out_flat_stacked.sum(dim=1)   # B*T, C

        # Combine shared and routed expert paths
        outputs = out_flat + out_flat_shared   # B*T, C
        outputs = outputs.reshape(B, T, C)
        return outputs

    def update_expert_bias(self, coeff=1e-3):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.router.tokens_per_expert_counter)
        # Push experts with low token counts up, and vice-versa
        token_count_mean = self.router.tokens_per_expert_counter.mean()
        token_count_centered = self.router.tokens_per_expert_counter - token_count_mean
        self.router.expert_bias -= coeff * torch.sign(token_count_centered)
        self.router.expert_bias = self.router.expert_bias - self.router.expert_bias.mean()

    def zero_token_counters(self):
        self.router.tokens_per_expert_counter.zero_()
