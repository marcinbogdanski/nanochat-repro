import os
if os.environ.get("RANK", "0") == "0":  # set on rank 0 only
    os.environ["TORCH_LOGS"] = "graph_breaks,recompiles"
else:
    os.environ.pop("TORCH_LOGS", None)
from contextlib import nullcontext
import torch
import torch.nn as nn

# Run like this:
# TMPDIR=/workspace/my-nanochat/debug_tmp TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+inductor" uv run python -m dev.example_debug
# Notes:
# TMPDIR *must* be specified, and must exist, and must be empty - some lock/cache mechanism stops dump from working otherwise


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 128, bias=False)
        self.linear2 = nn.Linear(128, 32, bias=False)
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x



torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
x = torch.randn(16, 64, device="cuda")

model = SmallModel().cuda()
model = torch.compile(model)


# autocast_ctx = nullcontext()
autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)

with autocast_ctx:
    out = model(x)
loss = out.float().square().mean()
loss.backward()
model.zero_grad()


torch.cuda.synchronize()
with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:

    with autocast_ctx:
        out = model(x)
    loss = out.float().square().mean()
    #loss.backward()
    #model.zero_grad()

    torch.cuda.synchronize()




print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace(f"trace.json")

#print('---')
#for i, e in enumerate(prof.events()[:150]):
#    print(i, e.name, e.device_type, e.self_cpu_time_total)

