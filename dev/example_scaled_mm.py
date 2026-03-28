import torch


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

def to_fp8(x, dtype):
    fp8_max = torch.finfo(dtype).max
    # Compute scale in higher precision
    x_max = x.float().abs().amax()
    # Go to fp63 so eager/compiled paths are numerically identical
    scale = (fp8_max / x_max.double().clamp(min=1e-12)).float()
    x_scaled = x * scale
    # Explicit clamp, protect against small numerical error if scale is imperfect
    x_scaled_clipped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_scaled_clipped.to(dtype)
    scale_inv = scale.reciprocal()
    return x_fp8, scale_inv

input = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
print('input', input)
weight = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
print('weight', weight)
input_fp8, input_scale_inv = to_fp8(input, torch.float8_e4m3fn)
print('input_fp8', input_fp8)
print('input_scale_inv', input_scale_inv)
weight_fp8, weight_scale_inv = to_fp8(weight, torch.float8_e4m3fn)
print('weight_fp8', weight_fp8)
print('weight_scale_inv', weight_scale_inv)

output = torch.mm(input, weight.t())
output_fp8 = torch._scaled_mm(
    input=input_fp8,
    mat2=weight_fp8.t(),
    scale_a=input_scale_inv,
    scale_b=weight_scale_inv,
    out_dtype=torch.float32,
    use_fast_accum=True
)

print('---')
print(output)
print('---')
print(output_fp8)
print('---')

abs_diff = (output - output_fp8).abs()
rel_diff = abs_diff / output.abs().clamp_min(1e-6)

print( f"{abs_diff.max()=}" )
print( f"{abs_diff.mean()=}" )
print( f"{rel_diff.max()=}")
print( f"{rel_diff.mean()=}")
