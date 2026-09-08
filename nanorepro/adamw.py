import torch
from nanorepro.nsight_trace import record_event

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
    metrics=False,  # effectively a compile-time flag to enable metric calculation
):
    # Metrics
    # We want ||delta_W||/||W|| aggregated by transformer block. For that we need
    # sum(delta_W**2) and sum(W**2) for the whole block, before we can reduce and divide.
    # This is why we return per-tensor sums here. Then later on, in GPT class,
    # we can aggregate them by block and calculate the final ratio.
    grad_sum_squares, update_sum_squares, params_sum_squares = None, None, None
    if metrics:
        grad_sum_squares = grad.float().square().sum()
        params_sum_squares = params.float().square().sum()

    # FP32 math is a MPS compatibility fix from Nanochat - technically on CUDA equivalent implicit impl.
    params_fp32 = params.float()
    exp_avg_fp32 = exp_avg.float()
    exp_avg_sq_fp32 = exp_avg_sq.float()
    grad_fp32 = grad.float()

    # Weight Decay
    # p = p - lr * weight_decay * p
    wd_update = params_fp32 * lr * wd
    params_fp32.mul_(1 - lr * wd)

    # Update v
    # v = B1 * v + (1-B1) * g
    exp_avg_fp32.lerp_(grad_fp32, 1-beta1)

    # Update s
    # s = B2 * s + (1-B2) * g**2
    exp_avg_sq_fp32.lerp_(grad_fp32.square(), 1-beta2)
    
    # Correction
    # Somewhat convoluted way to do:
    # v_corrected = v / (1-B1**t)
    # s_corrected = s / (1-B2**t)
    # p = p - lr * v_corrected / (sqrt(s_corrected)+eps)
    bias1 = 1-beta1**step
    bias2 = 1-beta2**step
    denom = (exp_avg_sq_fp32 / bias2).sqrt().add_(eps)
    optim_update = exp_avg_fp32.div(denom).mul_(lr / bias1)

    # Modify in-place
    params_fp32.add_(optim_update, alpha=-1.0)

    # Copy back to original dtype
    params.copy_(params_fp32)
    exp_avg.copy_(exp_avg_fp32)
    exp_avg_sq.copy_(exp_avg_sq_fp32)

    if metrics:
        final_update = -(optim_update + wd_update)
        update_sum_squares = final_update.float().square().sum()

    return grad_sum_squares, update_sum_squares, params_sum_squares


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
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, enable_metrics=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.enable_metrics = enable_metrics
        self.debug_stats = {}    # metrics, if enabled

    def get_metrics(self):
        return self.debug_stats
    
    @torch.no_grad()
    def step(self):
        self.debug_stats = {}  # clear every step
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                # Lazy Init
                if param not in self.state:
                    self.state[param] = {
                        'step': 0,
                        'exp_avg': torch.zeros_like(param),
                        'exp_avg_sq': torch.zeros_like(param),
                    }
                self.state[param]['step'] += 1

                grad = param.grad
                exp_avg = self.state[param]['exp_avg']
                exp_avg_sq = self.state[param]['exp_avg_sq']

                step = torch.tensor(self.state[param]['step'], device='cpu', dtype=torch.float32)
                lr = torch.tensor(group['lr'], device='cpu', dtype=torch.float32)
                beta1 = torch.tensor(group['betas'][0], device='cpu', dtype=torch.float32)
                beta2 = torch.tensor(group['betas'][1], device='cpu', dtype=torch.float32)
                eps = torch.tensor(group['eps'], device='cpu', dtype=torch.float32)
                wd = torch.tensor(group['weight_decay'], device='cpu', dtype=torch.float32)

                grad_sum_squares, update_sum_squares, param_sum_squares = fused_adamw_step(
                    params=param,
                    grad=grad,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    step=step,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    wd=wd,
                    metrics=self.enable_metrics,
                )
                if self.enable_metrics:
                    grad_sum_squares = grad_sum_squares.item() if grad_sum_squares is not None else None
                    update_sum_squares = update_sum_squares.item() if update_sum_squares is not None else None
                    param_sum_squares = param_sum_squares.item() if param_sum_squares is not None else None
                    self.debug_stats[param] = {
                        'grad_sq_sum': grad_sum_squares,
                        'update_sq_sum': update_sum_squares,
                        'params_sq_sum': param_sum_squares,
                        'params_num_el': grad.numel(),
                    }




class DistAdamW(torch.optim.Optimizer):
    """ZeRO-2 version of AdamW optimizer"""
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, enable_metrics=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.enable_metrics = enable_metrics
        self.debug_stats = {}    # metrics, if enabled

    def get_metrics(self):
        return self.debug_stats

    @torch.no_grad()
    def step(self):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        self.debug_stats = {}  # clear every step

        temp_buffers = {}

        for i, group in enumerate(self.param_groups):
            for j, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                # Lazy Init
                if group['is_small']:
                    # Don't slice
                    slice_width = param.size(0)
                    slice_start = 0
                    slice_end = param.size(0)
                else:
                    assert param.size(0) % world_size == 0
                    slice_width = param.size(0) // world_size
                    slice_start = rank * slice_width
                    slice_end = slice_start + slice_width

                if param not in self.state:
                    self.state[param] = {
                        'step': 0,
                        'exp_avg': torch.zeros_like(param[:slice_width]),
                        'exp_avg_sq': torch.zeros_like(param[:slice_width]),
                    }
                self.state[param]['step'] += 1

                # Sync point 1
                grad_slice = torch.empty_like(param.grad[:slice_width])
                shape_str = "x".join(map(str, param.shape))
                event_suffix = "_rs" if not group['is_small'] else "_ar"  # _rs for reduce_scatter, _ar for all_reduce
                event_name = f"adamw_g{i}_p{j}_{shape_str}"   # g=group, p=param
                record_event(event_name + event_suffix + ".begin")  # spans launch-to-completion, includes waiting, not just NCCL comms
                if group['is_small']:
                    # Don't slice
                    future = torch.distributed.all_reduce(
                        param.grad, op=torch.distributed.ReduceOp.AVG, async_op=True
                    ).get_future()
                    grad_slice = param.grad
                    param_slice = param
                else:
                    future = torch.distributed.reduce_scatter_tensor(
                        grad_slice, param.grad, op=torch.distributed.ReduceOp.AVG, async_op=True
                    ).get_future()
                    param_slice = param[slice_start:slice_end]

                temp_buffers[(i,j)] = {
                    'future': future,
                    'grad_slice': grad_slice,
                    'param_slice': param_slice,
                    'event_name': event_name,
                }

        for i, group in enumerate(self.param_groups):
            for j, param in enumerate(group['params']):
                temp_buffers[(i,j)].pop('future').wait()
                grad_slice = temp_buffers[(i,j)].pop('grad_slice')
                param_slice = temp_buffers[(i,j)].pop('param_slice')
                event_name = temp_buffers[(i,j)]['event_name']
                event_suffix = "_rs" if not group['is_small'] else "_ar"
                record_event(event_name + event_suffix + ".end")

                exp_avg = self.state[param]['exp_avg']
                exp_avg_sq = self.state[param]['exp_avg_sq']

                step = torch.tensor(self.state[param]['step'], device='cpu', dtype=torch.float32)
                lr = torch.tensor(group['lr'], device='cpu', dtype=torch.float32)
                beta1 = torch.tensor(group['betas'][0], device='cpu', dtype=torch.float32)
                beta2 = torch.tensor(group['betas'][1], device='cpu', dtype=torch.float32)
                eps = torch.tensor(group['eps'], device='cpu', dtype=torch.float32)
                wd = torch.tensor(group['weight_decay'], device='cpu', dtype=torch.float32)

                if not group['is_small']:
                    record_event(event_name + "_fused.begin")
                grad_sum_squares, update_sum_squares, param_sum_squares = fused_adamw_step(
                    params=param_slice,
                    grad=grad_slice,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    step=step,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    wd=wd,
                    metrics=self.enable_metrics,
                )
                if self.enable_metrics:
                    grad_sum_squares = grad_sum_squares.item() if grad_sum_squares is not None else None
                    update_sum_squares = update_sum_squares.item() if update_sum_squares is not None else None
                    param_sum_squares = param_sum_squares.item() if param_sum_squares is not None else None
                    param_num_el = grad_slice.numel()
                    if rank != 0 and group['is_small']:
                        # For small param, rank 0 has the full param and grad, zero other ranks to avoid duplication
                        grad_sum_squares = 0.0
                        update_sum_squares = 0.0
                        param_sum_squares = 0.0
                        param_num_el = 0
                    self.debug_stats[param] = {
                        'grad_sq_sum': grad_sum_squares,
                        'update_sq_sum': update_sum_squares,
                        'params_sq_sum': param_sum_squares,
                        'params_num_el': param_num_el,
                    }
                if not group['is_small']:
                    record_event(event_name + "_fused.end")

                # Sync point 2
                if not group['is_small']:
                    record_event(event_name + "_ag.begin")
                    future2 = torch.distributed.all_gather_into_tensor(
                        param, param_slice, async_op=True
                    ).get_future()
                    temp_buffers[(i,j)]['future2'] = future2
        
        for i, group in enumerate(self.param_groups):
            for j, param in enumerate(group['params']):
                if not group['is_small']:
                    temp_buffers[(i,j)].pop('future2').wait()
                    event_name = temp_buffers[(i,j)].pop('event_name')
                    record_event(event_name + "_ag.end")
