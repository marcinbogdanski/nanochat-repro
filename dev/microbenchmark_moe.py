import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mynanochat.moe import MoE
from dev.moe_reference import MoE as MoE_ref  # Nanochat version

# Run like this
# python -m dev.microbenchmark_moe

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

B, T, C = 8, 2048, 768
E = 8  # num experts
K = 2  # top_k

model_m = MoE(C=C, E=E, K=K).cuda()
model_k = MoE_ref(n_embd=C, num_experts=E, top_k=K).cuda()
with torch.no_grad():
    for (name_m, p_m), (name_k, p_k) in zip(model_m.named_parameters(), model_k.named_parameters()):
        p_k.data.copy_(p_m.data)


x = torch.randn(B, T, C).cuda()  # B,T,C
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    out_m = model_m(x)
    out_k = model_k(x)
loss_m = out_m.float().square().mean()
loss_k = out_k.float().square().mean()
loss_m.backward()
loss_k.backward()

print(out_m[0, :5, :5])
print(out_k[0, :5, :5])
print(f"(out_m-out_k).abs().max().item(): {(out_m-out_k).abs().max().item()}")  # 0.0

# Likewise compare gradients
for (name_m, p_m), (name_k, p_k) in zip(model_m.named_parameters(), model_k.named_parameters()):
    print(f"Param: {name_m} | max abs diff in grad: {(p_m.grad - p_k.grad).abs().max().item():.16f}")

model_m = model_m.cuda()
model_m = torch.compile(model_m)
model_k = model_k.cuda()
model_k = torch.compile(model_k)

torch.cuda.synchronize()
start_time = time.time()
for i in range(100):
    x = torch.randn(B, T, C).cuda()  # B,T,C
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model_m(x)
    loss = out.float().square().mean()
    loss.backward()
    model_m.zero_grad()
torch.cuda.synchronize()
end_time = time.time()
print(f"Time taken for 100 iterations: {end_time - start_time} seconds")

torch.cuda.synchronize()
start_time = time.time()
for i in range(100):
    x = torch.randn(B, T, C).cuda()  # B,T,C
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model_k(x)
    loss = out.float().square().mean()
    loss.backward()
    model_k.zero_grad()
torch.cuda.synchronize()
end_time = time.time()
print(f"Time taken for 100 iterations: {end_time - start_time} seconds")


print("Bye")
