import os
import time
import torch
import torch.nn as nn

# Run like this
# python -m dev.microbenchmark_fp8

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 128, bias=False)
        self.linear2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

# Graph capture function
captured_graphs = []
def capture_backend(gm, example_inputs):
    captured_graphs.append(gm)
    return gm.forward  # no compile
    # return inductor(gm, example_inputs)  # with compile



# Create model
model = SmallModel().cuda()
compiled = torch.compile(model, backend=capture_backend)

# Forward / backward pass
x = torch.randn(8, 64, device="cuda")
out = compiled(x)
loss = out.float().square().mean()
loss.backward()



# Extract the graph
gm = captured_graphs[0]

def get_dtype_and_shape(meta):
    val = meta.get("example_value", meta.get("val"))
    if val is not None and hasattr(val, "dtype") and hasattr(val, "shape"):
        return val.dtype, tuple(val.shape)
    return "—", "—"

# Print tabular with dtype and shape
def get_dtype_and_shape(node):
    val = node.meta.get("example_value", node.meta.get("val"))
    if val is not None and hasattr(val, "dtype") and hasattr(val, "shape"):
        return str(val.dtype), tuple(val.shape)
    return "—", "—"
print(f"{'name':42} {'op':13} {'target':55} {'dtype':15} shape")
print("-"*42 + " " + "-"*13 + " " + "-"*55 + " " + "-"*15 + " " + "-"*20)
for node in gm.graph.nodes:
    dtype, shape = get_dtype_and_shape(node)
    print(f"{node.name:42} {node.op:13} {str(node.target):55} {dtype:15} {shape}")
