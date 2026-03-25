import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mynanochat.fp8 import FP8Linear
from mynanochat.fp32_temp import FP32Linear


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




#-------------------------------------------------------------------------------
# Models
class SmallModelPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 128, bias=False)
        self.linear2 = nn.Linear(128, 32, bias=False)
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

class SmallModelFP32(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = FP32Linear(64, 128, bias=False)
        self.linear2 = FP32Linear(128, 32, bias=False)
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

class SmallModelFP8(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = FP8Linear(64, 128, bias=False)
        self.linear2 = FP8Linear(128, 32, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

#-------------------------------------------------------------------------------
# Init
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
x = torch.randn(16, 64, device="cuda")
ref1 = nn.Linear(64, 128, bias=False)
ref2 = nn.Linear(128, 32, bias=False)

# FX Capture
fx_capture_pt = FXGraphCapture()
fx_capture_fp32 = FXGraphCapture()
fx_capture_fp8 = FXGraphCapture()

# Create models
model_pt = SmallModelPT().cuda()
with torch.no_grad():
    model_pt.linear1.weight.copy_(ref1.weight)
    model_pt.linear2.weight.copy_(ref2.weight)
compiled_pt = torch.compile(model_pt, backend=fx_capture_pt.capture_backend)

model_fp32 = SmallModelFP32().cuda()
with torch.no_grad():
    model_fp32.linear1.weight.copy_(ref1.weight)
    model_fp32.linear2.weight.copy_(ref2.weight)
compiled_fp32 = torch.compile(model_fp32, backend=fx_capture_fp32.capture_backend)

model_fp8 = SmallModelFP8().cuda()
with torch.no_grad():
    model_fp8.linear1.weight.copy_(ref1.weight)
    model_fp8.linear2.weight.copy_(ref2.weight)
compiled_fp8 = torch.compile(model_fp8, backend=fx_capture_fp8.capture_backend)

#-------------------------------------------------------------------------------
# Forward / backward pass

out_pt = compiled_pt(x)
loss_pt = out_pt.float().square().mean()
loss_pt.backward()

out_fp32 = compiled_fp32(x)
loss_fp32 = out_fp32.float().square().mean()
loss_fp32.backward()

out_fp8 = compiled_fp8(x)
loss_fp8 = out_fp8.float().square().mean()
loss_fp8.backward()


def check_diff(title, t1, t2):
    print(f"--- {title} ---")
    abs_diff = (t1 - t2).abs()
    rel_diff = abs_diff / t1.abs().clamp_min(1e-6)
    cos_sim = F.cosine_similarity(t1.flatten(), t2.flatten(), dim=0)
    norm_ratio = t2.norm() / t1.norm()
    print( f"{abs_diff.max()=}" )
    print( f"{abs_diff.mean()=}" )
    print( f"{rel_diff.max()=}")
    print( f"{rel_diff.mean()=}")
    print( f"{cos_sim=}")
    print( f"{norm_ratio=}")

check_diff("out_pt, out_fp8", out_pt, out_fp8)
check_diff("pt-fp8, linear1.weight.grad", model_pt.linear1.weight.grad, model_fp8.linear1.weight.grad)
check_diff("pt-fp8, linear2.weight.grad", model_pt.linear2.weight.grad, model_fp8.linear2.weight.grad)


#-------------------------------------------------------------------------------
# Print captured graphs
print(); fx_capture_pt.print_captured_graph()
print(); fx_capture_fp32.print_captured_graph()
print(); fx_capture_fp8.print_captured_graph()
# fx_capture_pt.save_dot_graph("small_1.svg")
# fx_capture_fp32.save_dot_graph("small_2.svg")
# fx_capture_fp8.save_dot_graph("small_3.svg")
