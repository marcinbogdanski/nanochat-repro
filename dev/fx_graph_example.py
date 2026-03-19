import os
os.environ["TORCH_LOGS"] = "graph_breaks,recompiles"
import torch
import torch.nn as nn


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
    return gm.forward

# Create model, capture the graph
model = SmallModel().cuda()
compiled = torch.compile(model, backend=capture_backend)

# Forward / backward pass
x = torch.randn(8, 64, device="cuda")
out = compiled(x)
loss = out.sum()
loss.backward()

# Extract the graph
gm = captured_graphs[0]

# Print tabular
print("=" * 80)
print("RAW GRAPH")
print("=" * 80)
gm.graph.print_tabular()
