import os
import time
import torch
import torch.nn as nn

# Run like this
# python -m dev.microbenchmark_fp8

class FXGraphCapture:
    def __init__(self):
        self.captured_graphs = []

    def capture_backend(self, gm, example_inputs):
        self.captured_graphs.append(gm)
        return gm.forward  # no compile
        # return inductor(gm, example_inputs)  # with compile

    def _get_dtype_and_shape(self, meta):
        val = meta.get("example_value", meta.get("val"))
        if val is not None and hasattr(val, "dtype") and hasattr(val, "shape"):
            return val.dtype, tuple(val.shape)
        return "-", "-"

    def print_captured_graph(self):
        # Extract the graph
        gm = self.captured_graphs[0]
        rows = [
            ("---", "---", "---", "---", "---"),
            ("NAME", "OP", "TARGET", "DTYPE", "SHAPE"),
            ("---", "---", "---", "---", "---")
        ]
        for node in gm.graph.nodes:
            dtype, shape = self._get_dtype_and_shape(node.meta)
            rows.append((node.name, node.op, str(node.target), str(dtype), str(shape)))
        w = [max(map(len, col)) for col in zip(*rows)]  # widths for each column
        for r in rows:
            print(f"{r[0]:{w[0]}} {r[1]:{w[1]}} {r[2]:{w[2]}} {r[3]:{w[3]}} {r[4]:{w[4]}}")

    def save_dot_graph(self, filename):
        from torch.fx.passes.graph_drawer import FxGraphDrawer
        gm = self.captured_graphs[0]
        drawer = FxGraphDrawer(gm, "small_model")
        dot = drawer.get_dot_graph()
        node_meta = {node.name: node.meta for node in gm.graph.nodes}
        for dot_node in dot.get_nodes():
            name = dot_node.get_name().strip('"')
            meta = node_meta.get(name)
            if not meta:
                continue
            dtype, shape = self._get_dtype_and_shape(meta)
            if dtype == "-":
                continue
            old_label = dot_node.get("label").strip('"')
            if old_label.startswith("{") and old_label.endswith("}"):
                new_label = f"{old_label[:-1]}|dtype={dtype}|shape={shape}}}"
            else:
                new_label = f"{old_label}\\n{dtype}\\n{shape}"
            dot_node.set("label", new_label)
        dot.write_svg(filename)




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

# FX Capture
fx_capture = FXGraphCapture()
# Create model
model = SmallModel().cuda()
compiled = torch.compile(model, backend=fx_capture.capture_backend)
# Forward / backward pass
x = torch.randn(8, 64, device="cuda")
out = compiled(x)
loss = out.float().square().mean()
loss.backward()
# Print captured graph
print()
fx_capture.print_captured_graph()
# fx_capture.save_dot_graph("small_1.svg")



from mynanochat.fp8 import FP8Linear

class SmallModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = FP8Linear(64, 128, bias=False)
        self.linear2 = FP8Linear(128, 10, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

# FX Capture
fx_capture = FXGraphCapture()
# Create model
model = SmallModel2().cuda()
compiled = torch.compile(model, backend=fx_capture.capture_backend)
# Forward / backward pass
x = torch.randn(8, 64, device="cuda")
out = compiled(x)
loss = out.float().square().mean()
loss.backward()
# Print captured graph
print()
fx_capture.print_captured_graph()
# fx_capture.save_dot_graph("small_2.svg")
