
import torch
import torch.nn as nn
import torch.nn.functional as F
from dev.moe_reference import MoE  # Nanochat version

# Run like this
# python -m dev.microbenchmark_moe

class SmallModelMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32, 128, bias=False)
        self.linear2 = nn.Linear(128, 32, bias=False)
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
    
class SmallModelMOE(nn.Module):
    def __init__(self, C, E, K):
        super().__init__()
        self.K = K
        self.router = nn.Linear(C, E, bias=False)
        self.experts_up = nn.ModuleList([nn.Linear(C, C*4//K, bias=False) for _ in range(E)])
        self.experts_down = nn.ModuleList([nn.Linear(C*4//K, C, bias=False) for _ in range(E)])

    @torch.compiler.disable  # Dynamic slicing breaks the torch.compile
    def forward(self, x):
        B, T, C = x.shape
        K = self.K
        E = len(self.experts_up)
        x_flat = x.reshape(-1, C)          # B*T, C
        logits = self.router(x_flat)       # B*T, E
        weights = torch.sigmoid(logits)    # B*T, E
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
            h = self.experts_up[i](x_flat_stacked_flat_sorted_weighted[start_idx:end_idx])
            z = F.relu(h).square()
            o = self.experts_down[i](z)
            outs.append(o)
            start_idx += num_expert
        out_flat_stacked_flat_sorted = torch.cat(outs)   # B*T*K, C
        
        out_flat_stacked_flat = torch.zeros(B*T*K, C, device=x.device, dtype=x.dtype)
        out_flat_stacked_flat[indices_flat_sorted_indices] = out_flat_stacked_flat_sorted   # B*T*K, C
        out_flat_stacked = out_flat_stacked_flat.reshape(B*T, K, C)
        out_flat = out_flat_stacked.sum(dim=1)   # B*T, C
        outputs = out_flat.reshape(B, T, C)
        return outputs
    
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

B, T, C = 8, 2048, 768
E = 8  # num experts
K = 2  # top_k

model_m = SmallModelMOE(C=C, E=E, K=K)
model_k = MoE(n_embd=C, num_experts=E, top_k=K)
with torch.no_grad():
    for (name_m, p_m), (name_k, p_k) in zip(model_m.named_parameters(), model_k.named_parameters()):
        p_k.data.copy_(p_m.data)


x = torch.randn(B, T, C)  # B,T,C

with torch.no_grad():
    out_m = model_m(x)
    out_k = model_k(x)
    loss_m = out_m.float().square().mean()
    loss_k = out_k.float().square().mean()
#loss_m.backward()
#loss_k.backward()

print(out_m[0, :5, :5])
print(out_k[0, :5, :5])

print(f"(out_m-out_k).abs().max().item(): {(out_m-out_k).abs().max().item()}")  # 0.0

print("Bye")
