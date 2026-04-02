import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mynanochat.moe import MoE

# Run like this
# python -m dev.microbenchmark_moe

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

B, T, C = 8, 2048, 768
E = 8  # num experts
K = 2  # top_k

model = MoE(dim=C, n_routed_experts=E, top_k=K).cuda()
# Init weights weights so we can see non-zero results, down projections are zero by default
for param in model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)

x = torch.randn(B, T, C).cuda()  # B,T,C
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    out = model(x)
loss = out.float().square().mean()
loss.backward()
print(loss.item())
print(out[0, :5, :5])

model = torch.compile(model)

torch.cuda.synchronize()
start_time = time.time()
for i in range(100):
    x = torch.randn(B, T, C).cuda()  # B,T,C
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(x)
    loss = out.float().square().mean()
    loss.backward()
    model.zero_grad()
torch.cuda.synchronize()
end_time = time.time()
print(f"Time taken for 100 iterations: {end_time - start_time} seconds")

print("Bye")
