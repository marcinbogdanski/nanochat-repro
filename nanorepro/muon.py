import torch
from nanorepro.nsight_trace import record_event, collective_range

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
    metrics=False,  # effectively a compile-time flag to enable metric calculation
):

    # Metrics
    # We want ||delta_W||/||W|| aggregated by transformer block. For that we need
    # sum(delta_W**2) and sum(W**2) for the whole block, before we can reduce and divide.
    # This is why we return per-tensor sums here. Then later on, in GPT class,
    # we can aggregate them by block and calculate the final ratio.
    grad_sum_squares, update_sum_squares, params_sum_squares = None, None, None
    if metrics:
        reduce_dims = tuple(range(1, grad.ndim))  # zero-th dim is stack size
        grad_sum_squares = grad.float().square().sum(dim=reduce_dims)  # capture before Nesterov look-ahead

    # Update v: v = B1 * v + (1-B) * g
    v = momentum_buffer
    v.lerp_(grad, 1 - momentum)
    # Nesterov look-ahead: vv = B*v + (1-B)*g
    grad = grad.lerp(v, momentum)
    # BF16 for speed
    X = grad.to(compute_dtype)

    # -----------------------
    # MuonEq row equilibrium
    # https://arxiv.org/abs/2603.28254
    target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2)**0.5)
    row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    X = X * (target / row_norm).to(X.dtype)

    # --------------------------------
    # Polar express orthogonalization
    # https://arxiv.org/pdf/2505.16932
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
    grad = X.to(params.dtype)  # MPS compatibility fix from Nanochat

    # -----------------------
    # Muon+ renormalization
    # https://arxiv.org/abs/2602.21545
    target_norm = min(grad.size(-2), grad.size(-1))**0.5
    grad_norm = grad.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    grad = grad * (target_norm / grad_norm).to(grad.dtype)

    # ---------------------------------------------
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

    # -------------------------------
    # Decoupled Cautious Weight Decay
    lr = lr.to(update.dtype)
    wd = wd.to(update.dtype)
    mask = (update * params) >= 0
    update_full = lr * update + lr * wd * params * mask

    if metrics:
        reduce_dims = tuple(range(1, update_full.ndim))
        # technically update_full should have flipped sign, but square() makes it ok to omit it
        update_sum_squares = update_full.float().square().sum(dim=reduce_dims)
        params_sum_squares = params.float().square().sum(dim=reduce_dims)

    params.sub_(update_full)

    return grad_sum_squares, update_sum_squares, params_sum_squares




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
    def __init__(self, params, lr=0.01, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=0.1, compute_dtype=torch.bfloat16, enable_metrics=False):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, beta2=beta2, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.compute_dtype = compute_dtype
        self.enable_metrics = enable_metrics
        self.debug_stats = {}    # metrics, if enabled

    def get_metrics(self):
        return self.debug_stats

    @torch.no_grad()
    def step(self):
        assert all(p.grad is not None for group in self.param_groups for p in group["params"])
        self.debug_stats = {}  # clear every step

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
            grad_sum_squares, update_sum_squares, params_sum_squares = fused_muon_step(
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
                metrics=self.enable_metrics,
            )

            if self.enable_metrics:
                none_list = [None] * len(group['params'])
                grad_sum_squares = grad_sum_squares.cpu().numpy() if grad_sum_squares is not None else none_list
                update_sum_squares = update_sum_squares.cpu().numpy() if update_sum_squares is not None else none_list
                params_sum_squares = params_sum_squares.cpu().numpy() if params_sum_squares is not None else none_list
                assert len(grad_sum_squares) == len(group['params'])
                assert len(update_sum_squares) == len(group['params'])
                assert len(params_sum_squares) == len(group['params'])
                for jj, param in enumerate(group['params']):
                    self.debug_stats[param] = {
                        'grad_sq_sum': float(grad_sum_squares[jj]),  # np.float32 -> float
                        'update_sq_sum': float(update_sum_squares[jj]),
                        'params_sq_sum': float(params_sum_squares[jj]),
                        'params_num_el': param.numel(),
                    }

            # copy back params
            torch._foreach_copy_(group["params"], list(stacked_params.unbind(0)))



class DistMuon(torch.optim.Optimizer):
    """ZeRO-2 version of Muon optimizer"""
    def __init__(self, params, lr=0.01, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=0.1, compute_dtype=torch.bfloat16, enable_metrics=False):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, beta2=beta2, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.compute_dtype = compute_dtype
        self.enable_metrics = enable_metrics
        self.debug_stats = {}    # metrics, if enabled
        self.group_buffers = []  # static param/grad buffers, parameter .data/.grad point here

        # Initialize Static Buffers
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        # Each param group corresponds to exactly one comms bucket, I index by bucket_idx everywhere for consistency
        for bucket_idx, group in enumerate(self.param_groups):
            # Size and Pointer Accounting
            anchor = group['params'][0]  # shape, dtype, device
            num_params = len(group['params'])  # param objects in this group
            num_params_padded = -(-num_params // world_size) * world_size  # ceil div * world_size; params in group padded to multiple of world_size
            params_buffer_shape = [num_params_padded, *anchor.shape]
            num_params_per_rank = num_params_padded // world_size
            param_start = num_params_per_rank * rank

            # Create Static Buffers
            params_buffer = torch.zeros(params_buffer_shape, dtype=anchor.dtype, device=anchor.device)  # no requires_grad=True because this is just storage
            grads_buffer = torch.zeros_like(params_buffer)
            for i, param in enumerate(group["params"]):
                params_buffer[i].copy_(param.detach())
                param.data = params_buffer[i]  # assign view, now p.data points to our params_buffer[i]
                param.grad = grads_buffer[i]   # same here for grad, no copy needed since grads were not computed yet at init
            grads_shard = grads_buffer[param_start:param_start+num_params_per_rank]  # just a view
            params_shard = params_buffer[param_start:param_start+num_params_per_rank]  # just a view
            num_params_this_rank = min(num_params_per_rank, max(0, num_params-param_start))  # last rank may be padded
            # Group Name for Events
            shape_str = "x".join(map(str, group["params"][0].shape))
            group_name = f"muon_g{bucket_idx}_{shape_str}"

            self.group_buffers.append({
                'params': params_buffer,
                'grads': grads_buffer,
                'grads_shard': grads_shard,
                'params_shard': params_shard,
                'param_start': param_start,
                'num_local': num_params_this_rank,
                'group_name': group_name,
            })

            # Create Momentum Buffers
            self.state[anchor]['momentum_buffer'] = torch.zeros_like(grads_shard)
            if anchor.size(-2) >= anchor.size(-1):
                self.state[anchor]['momentum_buffer2'] = torch.zeros_like(grads_shard[..., :1])
            else:
                self.state[anchor]['momentum_buffer2'] = torch.zeros_like(grads_shard[..., :1, :])

        self.reduce_works = [None] * len(self.param_groups)
        self.gather_works = [None] * len(self.param_groups)

    def get_param_to_bucket_idx(self):
        return {param: group_idx for group_idx, group in enumerate(self.param_groups) for param in group['params']}

    def get_metrics(self):
        return self.debug_stats

    @torch.no_grad()
    def zero_grad(self, set_to_none=True):
        # Ignore set_to_none since we took over grad storage anyway
        # Note calling model.zero_grad(set_to_none=True) will still set .grad = None and break things, hence assert in step()
        for buffer in self.group_buffers:
            buffer['grads'].zero_()
        self.debug_stats = {}  # clear every step
        self.reduce_works = [None] * len(self.param_groups)
        self.gather_works = [None] * len(self.param_groups)

    @torch.no_grad()
    def launch_reduce(self, bucket_idx):
        assert self.reduce_works[bucket_idx] is None

        buffers = self.group_buffers[bucket_idx]
        group = self.param_groups[bucket_idx]
        for param_idx, param in enumerate(group["params"]):
            # Calling model.zero_grad(set_to_none=True) will set .grad=None and break static buffers, so we guard explicitly
            assert param.grad is not None and param.grad.data_ptr() == buffers['grads'][param_idx].data_ptr()

        with collective_range(buffers['group_name'] + "_rs"):    # Name the collective so we can label GPU channels in post processing
            self.reduce_works[bucket_idx] = torch.distributed.reduce_scatter_tensor(
                output=buffers['grads_shard'],
                input=buffers['grads'],
                op=torch.distributed.ReduceOp.AVG,
                async_op=True
            )

    @torch.no_grad()
    def bucket_step(self, bucket_idx):
        assert self.gather_works[bucket_idx] is None
        buffers = self.group_buffers[bucket_idx]
        group = self.param_groups[bucket_idx]
        anchor = group['params'][0]  # shape, dtype, device

        # Wait for reduce-scatter
        self.reduce_works[bucket_idx].wait()

        # Guard empty rank
        num_params_this_rank = buffers['num_local']
        if num_params_this_rank > 0:

            # Update LR/beta2
            lr = group['lr'] * (max(1, anchor.size(-2) / anchor.size(-1)))**0.5
            beta2 = group['beta2'] if group['beta2'] is not None else 0.0

            # 0-D CPU tensors to avoid re-compilation when values change
            lr = torch.tensor(lr, device='cpu', dtype=torch.float32)
            momentum = torch.tensor(group['momentum'], device='cpu', dtype=torch.float32)
            wd = torch.tensor(group['weight_decay'], device='cpu', dtype=torch.float32)
            beta2 = torch.tensor(beta2, device='cpu', dtype=torch.float32)

            # Fused Kernel
            record_event(buffers['group_name'] + "_fused.begin")
            grad_sum_squares, update_sum_squares, params_sum_squares = fused_muon_step(
                params=buffers['params_shard'][:num_params_this_rank],
                grad=buffers['grads_shard'][:num_params_this_rank],
                momentum_buffer=self.state[anchor]['momentum_buffer'][:num_params_this_rank],
                momentum_buffer2=self.state[anchor]['momentum_buffer2'][:num_params_this_rank],
                lr=lr,
                momentum=momentum,
                wd=wd,
                beta2=beta2,
                steps=group['ns_steps'],
                compute_dtype=self.compute_dtype,
                metrics=self.enable_metrics,
            )
            record_event(buffers['group_name'] + "_fused.end")

            # Collect Metrics
            if self.enable_metrics:
                none_list = [None] * num_params_this_rank
                grad_sum_squares = grad_sum_squares.cpu().numpy() if grad_sum_squares is not None else none_list
                update_sum_squares = update_sum_squares.cpu().numpy() if update_sum_squares is not None else none_list
                params_sum_squares = params_sum_squares.cpu().numpy() if params_sum_squares is not None else none_list
                idx_start = buffers['param_start']
                owned_params = group['params'][idx_start:idx_start+num_params_this_rank]
                for jj, param in enumerate(owned_params):
                    self.debug_stats[param] = {
                        'grad_sq_sum': float(grad_sum_squares[jj]),  # np.float32 -> float
                        'update_sq_sum': float(update_sum_squares[jj]),
                        'params_sq_sum': float(params_sum_squares[jj]),
                        'params_num_el': param.numel(),
                    }

        # All metrics are sum-reduced across ranks, so fill with zeros to be explicit
        if self.enable_metrics:
            for p in group['params']:
                if p not in self.debug_stats:
                    self.debug_stats[p] = {
                        'grad_sq_sum': 0.0,
                        'update_sq_sum': 0.0,
                        'params_sq_sum': 0.0,
                        'params_num_el': 0,
                    }

        # Do all-gather directly to static buffer
        with collective_range(buffers['group_name'] + "_ag"):
            self.gather_works[bucket_idx] = torch.distributed.all_gather_into_tensor(
                output_tensor=buffers["params"],
                input_tensor=buffers['params_shard'],
                async_op=True
            )

    @torch.no_grad()
    def wait_gather(self, bucket_idx):
        self.gather_works[bucket_idx].wait()

    @torch.no_grad()
    def launch_pending_reduces(self):
        for bucket_idx in range(len(self.param_groups)):
            if self.reduce_works[bucket_idx] is None:
                self.launch_reduce(bucket_idx)  # Launch only if not launched during backward pass

    @torch.no_grad()
    def step(self):
        assert all(p.grad is not None for group in self.param_groups for p in group["params"])

        # Loop 1: Launch reduce-scatter
        self.launch_pending_reduces()

        # Loop 2: Step and launch all-gather
        for bucket_idx in range(len(self.param_groups)):
            self.bucket_step(bucket_idx)

        # Loop 3: Wait for all-gather
        for bucket_idx in range(len(self.param_groups)):
            self.wait_gather(bucket_idx)
