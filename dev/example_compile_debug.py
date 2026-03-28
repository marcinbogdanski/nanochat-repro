import os
import shutil

# Create an empty TMPDIR, set TORCH_COMPILE_DEBUG, TORCH_LOGS
# TMPDIR *must* be specified, and must exist, and must be empty - some lock/cache mechanism stops dump from working otherwise
dirname = os.path.dirname(os.path.abspath(__file__))
tmp_path = os.path.join(dirname, "../debug_tmp")
shutil.rmtree(tmp_path, ignore_errors=True)
os.makedirs(tmp_path)
os.environ["TMPDIR"] = tmp_path
os.environ["TORCH_COMPILE_DEBUG"] = "1"
# os.environ["TORCH_LOGS"] = "graph_breaks,recompiles,+inductor"
import torch
import torch.nn as nn

# Run like this:
# uv run python -m dev.example_debug

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

x = torch.randn(16, 64, device="cuda")

model = SmallModel().cuda()
model = torch.compile(model)


with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    out = model(x)
loss = out.float().square().mean()
loss.backward()

shutil.rmtree(tmp_path, ignore_errors=True)
