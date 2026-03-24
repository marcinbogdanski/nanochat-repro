import torch

class FP32Matmul(torch.autograd.Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(ctx, input, weight):
        # The forward pass can use ctx.
        ctx.save_for_backward(input, weight)
        output = input.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        return grad_input, grad_weight


class FP32Linear(torch.nn.Linear):
    def forward(self, input):
        output = FP32Matmul.apply(input, self.weight)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output
