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

def get_dtype_and_shape(meta):
    val = meta.get("example_value", meta.get("val"))
    if val is not None and hasattr(val, "dtype") and hasattr(val, "shape"):
        return val.dtype, tuple(val.shape)
    return "—", "—"

# Print tabular
print("=" * 80)
print("RAW GRAPH")
print("=" * 80)
gm.graph.print_tabular()

print("\n" + "=" * 80)
print("DETAILED NODE INFO")
print("=" * 80)
for node in gm.graph.nodes:
    meta = node.meta

    dtype, shape = get_dtype_and_shape(meta)

    nn_stack = meta.get("nn_module_stack", {})
    module_origin = ""
    if nn_stack:
        last = list(nn_stack.values())[-1]
        module_origin = (
            f"{last[0]}({last[1].__name__})"
            if isinstance(last, tuple)
            else str(last)
        )

    source_fn = meta.get("source_fn_stack", [])
    source = source_fn[-1] if source_fn else "—"

    print(f"\n  Node: {node.name}")
    print(f"    Op:      {node.op} -> {node.target}")
    print(f"    Dtype:   {dtype}")
    print(f"    Shape:   {shape}")
    print(f"    Module:  {module_origin or '—'}")
    print(f"    Source:  {source}")

from torch.fx.passes.graph_drawer import FxGraphDrawer
drawer = FxGraphDrawer(gm, "small_model")
dot = drawer.get_dot_graph()

node_meta = {node.name: node.meta for node in gm.graph.nodes}
for dot_node in dot.get_nodes():
    name = dot_node.get_name().strip('"')
    meta = node_meta.get(name)
    if not meta:
        continue

    dtype, shape = get_dtype_and_shape(meta)
    if dtype == "—":
        continue

    old_label = dot_node.get("label").strip('"')
    if old_label.startswith("{") and old_label.endswith("}"):
        new_label = f"{old_label[:-1]}|dtype={dtype}|shape={shape}}}"
    else:
        new_label = f"{old_label}\\n{dtype}\\n{shape}"
    dot_node.set("label", new_label)

dot.write_svg("dev/fx_graph.svg")
print(f"\nGraph saved to dev/fx_graph.svg")
