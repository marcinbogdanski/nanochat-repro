import torch

class FP32Matmul(torch.autograd.Function):
    """Drop-in nn.Linear replacemnt for testing how custom Function behaves."""
    
    @staticmethod
    # ctx is the first argument to forward
    def forward(ctx, input, weight):
        # torch.mm() *is* autocast aware, so we don't need to do anything in forward pass.
        # E.g.:
        # - first layer: in=fp32, w=fp32 -> both 'secretly' downcast to bf16. out=bf16
        # - second layer: in=bf16, w=fp32 -> w 'secretly' downcast to bf16, out=bf16
        # All works 'magically'        
        ctx.save_for_backward(input, weight)
        output = input.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # torch.mm() automatic autocast is *disabled* in backward pass, so this needs handling.
        # So in backward pass:
        # - the op should run in dtype defined by grad_output
        # - the results should match original input/weight dtype
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight.to(grad_output.dtype))
            grad_input = grad_input.to(input.dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input.to(grad_output.dtype))
            grad_weight = grad_weight.to(weight.dtype)
        return grad_input, grad_weight


class FP32Linear(torch.nn.Linear):
    def forward(self, input):
        output = FP32Matmul.apply(input, self.weight)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output
