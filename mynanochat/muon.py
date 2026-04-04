import torch

# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def fused_muon_step(
    params,
    grad,
    momentum_buffer,
    momentum_buffer2,
    momentum,
    lr,
    wd,
    beta2,
    steps,  # 5
    compute_dtype,
):

    # Update v: v = B1 * v + (1-B) * g
    v = momentum_buffer
    v.lerp_(grad, 1 - momentum)
    # Nesterov look-ahead: vv = B*v + (1-B)*g
    grad = grad.lerp(v, momentum)

    ###################################
    # Polar express orthogonalization
    # https://arxiv.org/pdf/2505.16932
    X = grad.to(compute_dtype)
    # Ensure spectral norm is at most 1 (with 2% safety factor)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if grad.size(-2) > grad.size(-1):
        for i in range(steps):
            a, b, c = polar_express_coeffs[i]
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for i in range(steps):
            a, b, c = polar_express_coeffs[i]
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    grad = X

    ################################################
    # Similar to NorMuon per row variance reduction
    # https://arxiv.org/pdf/2510.05491
    reduction_dim = -2 if momentum_buffer2.size(-2) == 1 else -1
    reduction_dim_size = grad.size(reduction_dim)
    # Per row variance
    s_squared_column = grad.float().square().mean(dim=reduction_dim, keepdim=True)
    # Current norm
    norm_current = s_squared_column.sum(dim=(-2,-1), keepdim=True) * reduction_dim_size
    norm_current = norm_current.sqrt()
    # EMA momentum_buffer2
    beta2 = beta2.to(grad.dtype)
    momentum_buffer2.lerp_(s_squared_column.to(dtype=momentum_buffer2.dtype), 1-beta2)
    # Compute scaling factor
    step_size_column = momentum_buffer2.clamp_min(1e-10).rsqrt()
    xx = (s_squared_column * reduction_dim_size) * step_size_column.float().square()
    norm_new = xx.sum(dim=(-2,-1), keepdim=True).sqrt()
    # Final scale
    final_scale = step_size_column * (norm_current / norm_new.clamp_min(1e-10))
    update = grad.mul(final_scale.to(grad.dtype))

    ##################################
    # Decoupled Cautious Weight Decay
    lr = lr.to(update.dtype)
    wd = wd.to(update.dtype)
    mask = (update * params) >= 0
    params.sub_(lr * update + lr * wd * params * mask)


class Muon(torch.optim.Optimizer):
    """Muon optimizer
    
    Algorithm:
        p = p - lr * wd * p              # decoupled weight decay
        v = B * v + (1-B) * g            # momentum 
        vv = B * v + (1-B) * g           # optional, Nesterov look-ahead (just lerp again)
        U = newton_schulz(vv)            # orthogonalize
        lr_adj = lr * sqrt(max(1, m/n))  # adjust for aspect ratio
        p = p - lr * U                   # update weights
    """
    def __init__(self, params, lr=0.01, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=0.1, compute_dtype=torch.bfloat16):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, beta2=beta2, weight_decay=weight_decay)
        self.compute_dtype = compute_dtype
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        assert all(p.grad is not None for group in self.param_groups for p in group["params"])

        for group in self.param_groups:
            # First dim is stack size, i.e. num params in group
            p = group['params'][0]  # keep buffers i group['params'][0] for each param_group
            stacked_params = torch.stack([p for p in group['params']])
            stacked_grads = torch.stack([p.grad for p in group['params']])

            # Create buffers
            if 'momentum_buffer' not in self.state[p]:
                self.state[p]['momentum_buffer'] = torch.zeros_like(stacked_params)
                if p.size(-2) >= p.size(-1):
                    self.state[p]['momentum_buffer2'] = torch.zeros_like(stacked_grads[..., :1])
                else:
                    self.state[p]['momentum_buffer2'] = torch.zeros_like(stacked_grads[..., :1, :])

            # Update
            lr = group['lr'] * (max(1, p.size(-2) / p.size(-1)))**0.5
            beta2 = group['beta2'] if group['beta2'] is not None else 0.0

            # 0-D CPU tensors to avoid re-compilation when values change
            lr = torch.tensor(lr, device='cpu', dtype=torch.float32)
            momentum = torch.tensor(group['momentum'], device='cpu', dtype=torch.float32)
            wd = torch.tensor(group['weight_decay'], device='cpu', dtype=torch.float32)
            beta2 = torch.tensor(beta2, device='cpu', dtype=torch.float32)
            fused_muon_step(
                params=stacked_params,
                grad=stacked_grads,
                momentum_buffer=self.state[p]['momentum_buffer'],
                momentum_buffer2=self.state[p]['momentum_buffer2'],
                lr=lr,
                momentum=momentum,
                wd=wd,
                beta2=beta2,
                steps=group['ns_steps'],
                compute_dtype=self.compute_dtype,
            )

            # copy back params
            torch._foreach_copy_(group["params"], list(stacked_params.unbind(0)))



class DistMuon(torch.optim.Optimizer):
    """ZeRO-2 version of Muon optimizer"""
    def __init__(self, params, lr=0.01, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=0.1, compute_dtype=torch.bfloat16):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, beta2=beta2, weight_decay=weight_decay)
        self.compute_dtype = compute_dtype
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        assert all(p.grad is not None for group in self.param_groups for p in group["params"])
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        temp_buffers = {}

        # This will reduce scatter grads, such that each rank gets full averaged grad for owned param
        for i, group in enumerate(self.param_groups):
            p = group['params'][0]  # shape, dtype, device
            num_params = len(group['params'])
            padded_num_params = ((len(group['params']) + world_size - 1) // world_size) * world_size
            num_params_per_rank = padded_num_params // world_size

            if len(group['params']) % world_size != 0:
                group['zero_buffer'] = torch.zeros_like(group['params'][0].grad)

            padded_grads = [p.grad for p in group['params']]
            if len(group['params']) % world_size != 0:
                padded_grads.extend([group['zero_buffer']] * (padded_num_params-len(group['params'])))
            stacked_all_grads = torch.stack(padded_grads)
            stacked_grads = torch.empty(num_params_per_rank, *p.shape, dtype=p.dtype, device=p.device)

            reduce_scatter_future = torch.distributed.reduce_scatter_tensor(
                output=stacked_grads,
                input=stacked_all_grads,
                op=torch.distributed.ReduceOp.AVG,
                async_op=True
            ).get_future()

            # Temp buffers
            temp_buffers[i] = {
                'reduce_scatter_future': reduce_scatter_future,
                'stacked_grads': stacked_grads,
                'stacked_all_grads': stacked_all_grads
            }

        # Do fused muon step
        for i, group in enumerate(self.param_groups):
            p = group['params'][0]  # shape, dtype, device
            num_params = len(group['params'])
            padded_num_params = ((len(group['params']) + world_size - 1) // world_size) * world_size
            num_params_per_rank = padded_num_params // world_size

            # Wait and get buffers
            temp_buffers[i].pop('reduce_scatter_future').wait()
            stacked_grads = temp_buffers[i].pop('stacked_grads')

            # Sync point 2
            idx_start = num_params_per_rank * rank
            padded_params = [p for p in group['params']]
            if len(group['params']) % world_size != 0:
                padded_params.extend([group['zero_buffer']] * (padded_num_params-len(group['params'])))
            stacked_params = torch.stack(padded_params[idx_start:idx_start+num_params_per_rank])

            # Create buffers
            if 'momentum_buffer' not in self.state[p]:
                self.state[p]['momentum_buffer'] = torch.zeros_like(stacked_grads)
                if p.size(-2) >= p.size(-1):
                    self.state[p]['momentum_buffer2'] = torch.zeros_like(stacked_grads[..., :1])
                else:
                    self.state[p]['momentum_buffer2'] = torch.zeros_like(stacked_grads[..., :1, :])

            num_params_this_rank = min(num_params_per_rank, max(0, num_params - idx_start))
            if num_params_this_rank > 0:

                # Update
                lr = group['lr'] * (max(1, p.size(-2) / p.size(-1)))**0.5
                beta2 = group['beta2'] if group['beta2'] is not None else 0.0

                # 0-D CPU tensors to avoid re-compilation when values change
                lr = torch.tensor(lr, device='cpu', dtype=torch.float32)
                momentum = torch.tensor(group['momentum'], device='cpu', dtype=torch.float32)
                wd = torch.tensor(group['weight_decay'], device='cpu', dtype=torch.float32)
                beta2 = torch.tensor(beta2, device='cpu', dtype=torch.float32)
                fused_muon_step(
                    params=stacked_params[:num_params_this_rank],
                    grad=stacked_grads[:num_params_this_rank],
                    momentum_buffer=self.state[p]['momentum_buffer'][:num_params_this_rank],
                    momentum_buffer2=self.state[p]['momentum_buffer2'][:num_params_this_rank],
                    lr=lr,
                    momentum=momentum,
                    wd=wd,
                    beta2=beta2,
                    steps=group['ns_steps'],
                    compute_dtype=self.compute_dtype,
                )

            # Reuse the stacked_all_grads buffer for params
            stacked_all_grads = temp_buffers[i]['stacked_all_grads']
            all_gather_future = torch.distributed.all_gather_into_tensor(
                stacked_all_grads, stacked_params, async_op=True
            ).get_future()
            temp_buffers[i]['all_gather_future'] = all_gather_future

        # Copy back params
        for i, group in enumerate(self.param_groups):
            num_params = len(group['params'])
            temp_buffers[i].pop('all_gather_future').wait()
            stacked_all_grads = temp_buffers[i].pop('stacked_all_grads')
            torch._foreach_copy_(group["params"], list(stacked_all_grads[:num_params].unbind(0)))
