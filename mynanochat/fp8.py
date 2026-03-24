import torch

def to_fp8(x, dtype):
    """Convert to one of fp8 dtypes"""
    assert dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

    fp8_max = torch.finfo(dtype).max
    # Compute scale in higher precision
    x_max = x.float().abs().amax()
    # Go to fp63 so eager/compiled paths are numerically identical
    scale = (fp8_max / x_max.double().clamp(min=1e-12)).float()
    x_scaled = x * scale
    # Expclict clamp, protect agains small numercial error if scale is imperfect
    x_scaled_clipped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_scaled_clipped.to(dtype)
    scale_inv = scale.reciprocal()
    return x_fp8, scale_inv


class FP8Matmul(torch.autograd.Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(ctx, input, weight, bias=None):
        # The forward pass can use ctx.
        ctx.save_for_backward(input, weight, bias)


        input_fp8, input_scale_inv = to_fp8(input, torch.float8_e4m3fn)
        weight_fp8, weight_scale_inv = to_fp8(weight, torch.float8_e4m3fn)
        output = torch._scaled_mm(
            input=input_fp8,
            mat2=weight_fp8.t(),
            scale_a=input_scale_inv,
            scale_b=weight_scale_inv,
            out_dtype=input.dtype,
            use_fast_accum=True
        )


        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class FP8Linear(torch.nn.Linear):
    def forward(self, input):
        return FP8Matmul.apply(input, self.weight, self.bias)
