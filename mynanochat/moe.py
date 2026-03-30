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
        self.experts.w_ups = nn.ParameterList([nn.Parameter(torch.empty(C*4//K, C)) for _ in range(E)])
        self.experts.w_downs = nn.ParameterList([nn.Parameter(torch.empty(C, C*4//K)) for _ in range(E)])

    @torch.compiler.disable  # Dynamic slicing breaks the torch.compile
    def forward(self, x):
        B, T, C = x.shape
        K = self.K
        E = len(self.experts.w_ups)
        x_flat = x.reshape(-1, C)          # B*T, C
        logits = self.router.gate(x_flat)       # B*T, E
        weights = torch.sigmoid(logits.float())    # B*T, E
        values, indices = torch.topk(weights, K, dim=-1)     # B*T, K
        x_flat_stacked = torch.stack([x_flat]*K, dim=1)      # B*T, K, C
        x_flat_stacked_flat = x_flat_stacked.reshape(-1, C)  # B*T*K, C
        indices_flat = indices.reshape(-1)                   # B*T*K
        values_flat = values.reshape(-1)                     # B*T*K
        indices_flat_sorted_indices = torch.argsort(indices_flat, stable=True)  # B*T*K
        x_flat_stacked_flat_sorted = x_flat_stacked_flat[indices_flat_sorted_indices]  # B*T*K, C
        values_flat_sorted = values_flat[indices_flat_sorted_indices]  # B*T*K
        x_flat_stacked_flat_sorted_weighted = x_flat_stacked_flat_sorted * values_flat_sorted.unsqueeze(-1)  # B*T*K, C
        
        start_idx = 0
        outs = []
        for i in range(E):
            num_expert = (indices==i).sum().item()
            end_idx = start_idx + num_expert
            h = x_flat_stacked_flat_sorted_weighted[start_idx:end_idx] @ self.experts.w_ups[i].T
            z = F.relu(h).square()
            o = z @ self.experts.w_downs[i].T
            outs.append(o)
            start_idx += num_expert
        out_flat_stacked_flat_sorted = torch.cat(outs)   # B*T*K, C
        
        out_flat_stacked_flat = torch.zeros(B*T*K, C, device=x.device, dtype=out_flat_stacked_flat_sorted.dtype)
        out_flat_stacked_flat[indices_flat_sorted_indices] = out_flat_stacked_flat_sorted   # B*T*K, C
        out_flat_stacked = out_flat_stacked_flat.reshape(B*T, K, C)
        out_flat = out_flat_stacked.sum(dim=1)   # B*T, C
        outputs = out_flat.reshape(B, T, C)
        return outputs
