
import torch
import torch.nn as nn


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
    def __init__(self):
        super().__init__()
        self.experts1 = nn.ModuleList([nn.Linear(32, 128, bias=False) for _ in range(4)])
        self.experts2 = nn.ModuleList([nn.Linear(128, 32, bias=False) for _ in range(4)])

    def call_experts_loop(self, x, experts, expert_offsets):
        assert len(experts) == len(expert_offsets)
        offsets_start = [0] + expert_offsets[:-1]
        outputs = []
        for expert, off_start, off_end in zip(experts, offsets_start, expert_offsets):
            out = expert(x[off_start:off_end])
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        B, T, C = x.shape
        assert outputs.shape == (B, T, C*4)
        return outputs
    
    def forward(self, x):
        expert_offsets = [4, 8, 12, 16]  # index of end of group, non-inclusive, pass directly to grouped_mm()
        x = self.call_experts_loop(x, self.experts1, expert_offsets=expert_offsets)
        x = torch.relu(x)
        x = self.call_experts_loop(x, self.experts2, expert_offsets=expert_offsets)
        return x
    
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

model = SmallModelMOE().cuda()
x = torch.randn(2, 16, 32, device="cuda")  # B,T,C

out = model(x)
loss = out.float().square().mean()
loss.backward()

print("Bye")
