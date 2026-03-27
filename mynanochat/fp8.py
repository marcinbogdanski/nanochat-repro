import torch
import torch.nn.functional as F

@torch.no_grad()
def to_fp8(x, dtype):
    """Convert to one of fp8 dtypes"""
    assert dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

    fp8_max = torch.finfo(dtype).max
    # Compute scale in higher precision
    x_max = x.float().abs().max()
    # Go to fp63 so eager/compiled paths are numerically identical
    scale = fp8_max / x_max.double().clamp(min=1e-12)
    scale = scale.float()
    x_scaled = x.float() * scale
    # Explicit clamp, protect against small numerical error if scale is imperfect
    x_scaled_clipped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_scaled_clipped.to(dtype)
    scale_inv = scale.reciprocal()
    return x_fp8, scale_inv

# Allow treating as black box
@torch._dynamo.allow_in_graph
class MatmulFP8(torch.autograd.Function):
    # ctx is the first argument to forward
    @staticmethod
    def forward(ctx, input, weight):
        # The forward pass can use ctx.
        ctx.save_for_backward(input, weight)

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
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_out_fp8, grad_out_scale_inv = to_fp8(grad_output, torch.float8_e5m2)

        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight)
            weight_fp8, weight_scale_inv = to_fp8(weight, torch.float8_e4m3fn)
            weight_fp8_cont = weight_fp8.t().contiguous().t()
            grad_input = torch._scaled_mm(
                input=grad_out_fp8,
                mat2=weight_fp8_cont,
                scale_a=grad_out_scale_inv,
                scale_b=weight_scale_inv,
                #out_dtype=input.dtype,  # AI says this is more correct, but nanochat uses grad_output
                out_dtype=grad_output.dtype,
                use_fast_accum=False
            )
        
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.t().mm(input)
            input_fp8, input_scale_inv = to_fp8(input, torch.float8_e4m3fn)
            input_fp8_cont = input_fp8.t().contiguous().t()
            grad_weight = torch._scaled_mm(
                input=grad_out_fp8.t().contiguous(),
                mat2=input_fp8_cont,
                scale_a=grad_out_scale_inv,
                scale_b=input_scale_inv,
                #out_dtype=weight.dtype,  # AI says this is more correct, but nanochat uses grad_output
                out_dtype=grad_output.dtype,
                use_fast_accum=False
            )

        return grad_input, grad_weight


class LinearFP8(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=False, mode='native'):
        super().__init__(in_features, out_features, bias=bias)
        assert mode in ['fp8', 'native']
        if mode == 'fp8':
            assert in_features % 16 == 0
            assert out_features % 16 == 0
        self.mode = mode

    def is_fp8_legal(self):
        # Input dim also matters, but we can't check ahead of time - we trust user to handle it
        return self.weight.size(0) % 16 == 0 and self.weight.size(1) % 16 == 0

    def switch_mode_if_legal(self, mode):
        """Switch mode. For fp8 only switch if dims allow, otherwise quietly do nothing."""
        if mode == 'native':
            self.mode = mode
        elif mode == 'fp8':
            if self.is_fp8_legal():
                self.mode = mode
        else:
            raise ValueError("Mode must be 'native' or 'fp8'")

    def forward(self, input):
        if self.mode == 'native':
            return F.linear(input, self.weight, self.bias)

        if torch.is_autocast_enabled():
            input = input.to(torch.get_autocast_gpu_dtype())
        input_2d = input.reshape(-1, input.shape[-1])
        output_2d = MatmulFP8.apply(input_2d, self.weight)
        output = output_2d.reshape(*input.shape[:-1], output_2d.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output
