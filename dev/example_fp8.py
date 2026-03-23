import torch


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

def to_fp8(x, dtype):
    print('vvvvv')
    fp8_max = torch.finfo(dtype).max
    print('fp8_max', fp8_max)
    x_max = x.abs().amax()
    print('x_max', x_max)
    scale = fp8_max / x_max
    print('scale', scale)
    x_scaled = x * scale
    print('x_scaled', x_scaled)
    x_scaled_clipped = x_scaled.clamp(-fp8_max, fp8_max)
    print('x_scaled_clipped', x_scaled_clipped)
    x_fp8 = x_scaled_clipped.to(dtype)
    print('x_fp8', x_fp8)
    print('^^^^^')
    return x_fp8, scale.reciprocal()


input = torch.randn(16, 64, device="cuda")
print('input', input)
weight = torch.randn(128, 64, device="cuda")
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

print( (output - output_fp8).abs().max() )

