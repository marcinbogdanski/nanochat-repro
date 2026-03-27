import torch

@torch.compile(dynamic=False, fullgraph=True)
def fused_adamw_step(
    params,
    grad,
    exp_avg,
    exp_avg_sq,
    step,
    lr,
    beta1,
    beta2,
    eps,
    wd,
):
    # Weight Decay
    # p = p - lr * weight_decay * p
    params.mul_(1 - lr * wd)

    # Update v
    # v = B1 * v + (1-B1) * g
    exp_avg.lerp_(grad, 1-beta1)

    # Update s
    # s = B2 * s + (1-B2) * g**2
    exp_avg_sq.lerp_(grad.square(), 1-beta2)
    
    # Correction
    # Somewhat convoluted way to do:
    # v_corrected = v / (1-B1**t)
    # s_corrected = s / (1-B2**t)
    # p = p - lr * v_corrected / (sqrt(s_corrected)+eps)
    bias1 = 1-beta1**step
    bias2 = 1-beta2**step
    denom = (exp_avg_sq / bias2).sqrt().add_(eps)
    update = exp_avg.div(denom).mul_(lr / bias1)
    params.add_(update, alpha=-1.0)


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer
    
    Algorithm:
        v = B1 * v + (1 - B1) * g          # first moment (direction)
        s = B2 * s + (1 - B2) * g^2        # second moment (scaling)

        v_corrected = v / (1 - B1^t)       # bias correction
        s_corrected = s / (1 - B2^t)

        p = p - lr * v_corrected / (sqrt(s_corrected) + eps)
        p = p - lr * wd * p                # AdamW: decoupled weight decay
    """
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for params in group['params']:
                if params.grad is None:
                    continue
                # Lazy Init
                if params not in self.state:
                    self.state[params] = {
                        'step': 0,
                        'exp_avg': torch.zeros_like(params),
                        'exp_avg_sq': torch.zeros_like(params),
                    }
                self.state[params]['step'] += 1

                grad = params.grad
                exp_avg = self.state[params]['exp_avg']
                exp_avg_sq = self.state[params]['exp_avg_sq']

                step = torch.tensor(self.state[params]['step'], device='cpu', dtype=torch.float32)
                lr = torch.tensor(group['lr'], device='cpu', dtype=torch.float32)
                beta1 = torch.tensor(group['betas'][0], device='cpu', dtype=torch.float32)
                beta2 = torch.tensor(group['betas'][1], device='cpu', dtype=torch.float32)
                eps = torch.tensor(group['eps'], device='cpu', dtype=torch.float32)
                wd = torch.tensor(group['weight_decay'], device='cpu', dtype=torch.float32)

                fused_adamw_step(
                    params=params,
                    grad=grad,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    step=step,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    wd=wd,
                )




class DistAdamW(torch.optim.Optimizer):
    """ZeRO-2 version of AdamW optimizer"""
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        temp_buffers = {}

        for i, group in enumerate(self.param_groups):
            for j, params in enumerate(group['params']):
                if params.grad is None:
                    continue
                # Lazy Init
                if group['is_small']:
                    # Don't slice
                    slice_width = params.size(0)
                    slice_start = 0
                    slice_end = params.size(0)
                else:
                    assert params.size(0) % world_size == 0
                    slice_width = params.size(0) // world_size
                    slice_start = rank * slice_width
                    slice_end = slice_start + slice_width

                if params not in self.state:
                    self.state[params] = {
                        'step': 0,
                        'exp_avg': torch.zeros_like(params[:slice_width]),
                        'exp_avg_sq': torch.zeros_like(params[:slice_width]),
                    }
                self.state[params]['step'] += 1

                # Sync point 1
                grad_slice = torch.empty_like(params.grad[:slice_width])
                if group['is_small']:
                    # Don't slice
                    future = torch.distributed.all_reduce(
                        params.grad, op=torch.distributed.ReduceOp.AVG, async_op=True
                    ).get_future()
                    grad_slice = params.grad
                    params_slice = params
                else:
                    future = torch.distributed.reduce_scatter_tensor(
                        grad_slice, params.grad, op=torch.distributed.ReduceOp.AVG, async_op=True
                    ).get_future()
                    params_slice = params[slice_start:slice_end]

                temp_buffers[(i,j)] = {
                    'future': future,
                    'grad_slice': grad_slice,
                    'params_slice': params_slice
                }

        for i, group in enumerate(self.param_groups):
            for j, params in enumerate(group['params']):
                temp_buffers[(i,j)].pop('future').wait()
                grad_slice = temp_buffers[(i,j)].pop('grad_slice')
                params_slice = temp_buffers[(i,j)].pop('params_slice')

                exp_avg = self.state[params]['exp_avg']
                exp_avg_sq = self.state[params]['exp_avg_sq']

                step = torch.tensor(self.state[params]['step'], device='cpu', dtype=torch.float32)
                lr = torch.tensor(group['lr'], device='cpu', dtype=torch.float32)
                beta1 = torch.tensor(group['betas'][0], device='cpu', dtype=torch.float32)
                beta2 = torch.tensor(group['betas'][1], device='cpu', dtype=torch.float32)
                eps = torch.tensor(group['eps'], device='cpu', dtype=torch.float32)
                wd = torch.tensor(group['weight_decay'], device='cpu', dtype=torch.float32)

                fused_adamw_step(
                    params=params_slice,
                    grad=grad_slice,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    step=step,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    wd=wd,
                )

                # Sync point 2
                if not group['is_small']:
                    future2 = torch.distributed.all_gather_into_tensor(
                        params, params_slice, async_op=True
                    ).get_future()
                    temp_buffers[(i,j)]['future2'] = future2
        
        for i, group in enumerate(self.param_groups):
            for j, params in enumerate(group['params']):
                if not group['is_small']:
                    temp_buffers[(i,j)].pop('future2').wait()
        


