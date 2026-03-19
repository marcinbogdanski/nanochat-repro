import os
os.environ["TORCH_LOGS"] = "graph_breaks,recompiles"
import torch
import torch.nn as nn
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch._dynamo.backends.inductor import inductor


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

# Graph capture functions
forward_graphs = []
backward_graphs = []

def capture_forward_backend(gm, example_inputs):
    forward_graphs.append(gm)
    return make_boxed_func(gm.forward)  # no compile
    # return make_boxed_func(inductor(gm, example_inputs))  # with compile

def capture_backward_backend(gm, example_inputs):
    backward_graphs.append(gm)
    return make_boxed_func(gm.forward)  # no compile
    # return make_boxed_func(inductor(gm, example_inputs))  # with compile

# Create model, capture the graph
model = SmallModel().cuda()
compiled = torch.compile(
    model,
    backend=aot_autograd(
        fw_compiler=capture_forward_backend,
        bw_compiler=capture_backward_backend,
    ),
)

# Forward / backward pass
x = torch.randn(8, 64, device="cuda")
out = compiled(x)
loss = out.sum()
loss.backward()

def get_dtype_and_shape(meta):
    val = meta.get("example_value", meta.get("val"))
    if val is not None and hasattr(val, "dtype") and hasattr(val, "shape"):
        return val.dtype, tuple(val.shape)
    return "—", "—"

def dump_graph(gm, title, svg_path):
    print("=" * 80)
    print(f"{title} RAW GRAPH")
    print("=" * 80)
    gm.graph.print_tabular()

    print("\n" + "=" * 80)
    print(f"{title} DETAILED NODE INFO")
    print("=" * 80)
    for node in gm.graph.nodes:
        meta = node.meta

        dtype, shape = get_dtype_and_shape(meta)

        nn_stack = meta.get("nn_module_stack", meta.get("fwd_nn_module_stack", {}))
        module_origin = ""
        if nn_stack:
            last = list(nn_stack.values())[-1]
            module_origin = (
                f"{last[0]}({last[1].__name__})"
                if isinstance(last, tuple)
                else str(last)
            )

        source_fn = meta.get("source_fn_stack", meta.get("fwd_source_fn_stack", []))
        source = source_fn[-1] if source_fn else "—"

        print(f"\n  Node: {node.name}")
        print(f"    Op:      {node.op} -> {node.target}")
        print(f"    Dtype:   {dtype}")
        print(f"    Shape:   {shape}")
        print(f"    Module:  {module_origin or '—'}")
        print(f"    Source:  {source}")

    drawer = FxGraphDrawer(gm, title.lower().replace(" ", "_"))
    dot = drawer.get_dot_graph()
    dot.write_svg(svg_path)
    print(f"\nGraph saved to {svg_path}")


dump_graph(forward_graphs[0], "FORWARD", "fx_graph_fw.svg")
dump_graph(backward_graphs[0], "BACKWARD", "fx_graph_bw.svg")
