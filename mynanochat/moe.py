import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, C, E, K):
        super().__init__()
        self.K = K
        self.router = nn.Module()
        self.router.gate = nn.Linear(C, E, bias=False)
        self.router.register_buffer('expert_bias', torch.zeros(E))
        self.router.register_buffer('tokens_per_expert_counter', torch.zeros(E))
        self.experts = nn.Module()
        scale = C**-0.5  # Xavier-style 1/sqrt(fan_in), overridden by init_weights in gpt.py
        active_experts = 1 + K  # 1 shared expert + K routed experts
        hidden_dim = round(C*4/active_experts/128) * 128  # Nearest multiple of 128
        self.experts.w_up = nn.Parameter(torch.randn(E, hidden_dim, C)*scale)
        self.experts.w_down = nn.Parameter(torch.zeros(E, C, hidden_dim))
        self.shared_expert = nn.Module()
        self.shared_expert.w_up = nn.Linear(C, hidden_dim, bias=False)
        self.shared_expert.w_down = nn.Linear(hidden_dim, C, bias=False)

    @torch.compiler.disable  # Dynamic slicing breaks the torch.compile
    def forward(self, x):
        B, T, C = x.shape
        K = self.K
        E = self.experts.w_up.size(0)
        x_flat = x.reshape(-1, C)          # B*T, C

        # Routed expert path start
        # Bias the expert selection, but *not* weighting (Nanochat, DeepSeekV3)
        logits = self.router.gate(x_flat)       # B*T, E
        weights = torch.sigmoid(logits.float())    # B*T, E
        weights_biased = weights + self.router.expert_bias   # B*T, E
        _, indices = torch.topk(weights_biased, K, dim=-1, sorted=False)   # B*T, K
        values = torch.gather(weights, dim=-1, index=indices)   # B*T, K
        x_flat_stacked = torch.stack([x_flat]*K, dim=1)      # B*T, K, C
        x_flat_stacked_flat = x_flat_stacked.reshape(-1, C)  # B*T*K, C
        indices_flat = indices.reshape(-1)                   # B*T*K
        values_flat = values.reshape(-1)                     # B*T*K
        indices_flat_sorted_indices = torch.argsort(indices_flat, stable=True)  # B*T*K
        x_flat_stacked_flat_sorted = x_flat_stacked_flat[indices_flat_sorted_indices]  # B*T*K, C
        values_flat_sorted = values_flat[indices_flat_sorted_indices]  # B*T*K
        x_flat_stacked_flat_sorted_weighted = x_flat_stacked_flat_sorted * values_flat_sorted.unsqueeze(-1)  # B*T*K, C

        # Shared expert path
        h_shared = self.shared_expert.w_up(x_flat)
        z_shared = F.relu(h_shared).square()
        out_flat_shared = self.shared_expert.w_down(z_shared)

        # Update expert token counts for load balancing
        # This probably should be disabled during evaluation
        expert_ids = torch.arange(E, device=indices.device)
        indices_column = indices_flat.unsqueeze(1)                           # B*T*K, 1
        num_tokens_per_expert = (indices_column == expert_ids).sum(dim=0).to(self.router.tokens_per_expert_counter.dtype)
        self.router.tokens_per_expert_counter += num_tokens_per_expert
        
        # Loop over routed experts
        start_idx = 0
        outs = []
        for i in range(E):
            num_expert = (indices==i).sum().item()
            end_idx = start_idx + num_expert
            h = x_flat_stacked_flat_sorted_weighted[start_idx:end_idx] @ self.experts.w_up[i].T
            z = F.relu(h).square()
            o = z @ self.experts.w_down[i].T
            outs.append(o)
            start_idx += num_expert
        
        # Combine routed experts and project back
        out_flat_stacked_flat_sorted = torch.cat(outs)   # B*T*K, C
        out_flat_stacked_flat = torch.zeros(B*T*K, C, device=x.device, dtype=out_flat_stacked_flat_sorted.dtype)
        out_flat_stacked_flat[indices_flat_sorted_indices] = out_flat_stacked_flat_sorted   # B*T*K, C
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
