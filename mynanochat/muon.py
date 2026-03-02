import torch

@torch.compile
def zeropower_via_newtonschulz(grad, steps=5):
    """Newton-schulz orthogonalization
    
    Algorithm:
        X = G / ||G||                        # scale so singular values < 1
        repeat 5 times:
            X = 1.5 * X - 0.5 * X @ X.T @ X
        return X
    """
    assert grad.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        X = X.T

    # Scale down to norm at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if grad.size(0) > grad.size(1):
        X = X.T
    return X

# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile
def zeropower_via_polar_express(grad, steps=5):
    """Polar express orthogonalization

    Details: https://arxiv.org/pdf/2505.16932
    
    Algorithm:
        X = G / ||G||                        # scale so singular values < 1
        repeat 5 times:
            X = 1.5 * X - 0.5 * X @ X.T @ X
        return X
    """
    assert grad.ndim == 2
    X = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        X = X.T

    # Ensure spectral norm is at most 1 (with 2% safety factor)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

    for i in range(steps):
        a, b, c = polar_express_coeffs[i]
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if grad.size(0) > grad.size(1):
        X = X.T
    return X


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
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Lazy Init
                if p not in self.state:
                    self.state[p] = {
                        'momentum_buffer': torch.zeros_like(p),
                    }

                # Update v
                # v = B1 * v + (1-B) * g
                v = self.state[p]['momentum_buffer']
                v.lerp_(p.grad, 1 - group['momentum'])

                # Optional Nesterov look-ahead
                # vv = B*v + (1-B)*g
                vv = p.grad.lerp(v, group['momentum']) if group['nesterov'] else v

                # Update
                update = zeropower_via_polar_express(vv, group['ns_steps'])
                lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5
                if group['weight_decay'] != 0:
                    # Decoupled Cautious Weight Decay
                    mask = (update * p) >= 0
                    p.sub_(lr * update + lr * group['weight_decay'] * p * mask)
                else:
                    p.sub_(lr * update)



class DistMuon(torch.optim.Optimizer):
    """ZeRO-2 version of Muon optimizer"""
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Assert all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"])

        # Sync point 1
        # This will reduce scatter grads, such that each rank gets full averated grad for owned param
        for group in self.param_groups:
            if len(group['params']) % world_size != 0:
                group['zero_buffer'] = torch.zeros_like(group['params'][0].grad)
            for i in range(0, len(group['params']), world_size):
                input_grads = [p.grad for p in group['params'][i:i+world_size]]
                if len(input_grads) < world_size:
                    input_grads.extend([group['zero_buffer']] * (world_size - len(input_grads)))
                out_tensor = group['params'][i+rank].grad if i+rank < len(group['params']) else torch.zeros_like(group['zero_buffer'])
                torch.distributed.reduce_scatter(
                    out_tensor,
                    input_grads,
                    op=torch.distributed.ReduceOp.AVG
                )
            
        for group in self.param_groups:
            for i in range(0, len(group['params']), world_size):
                if i+rank < len(group['params']):
                
                    p = group['params'][i+rank]

                    # Lazy Init
                    if p not in self.state:
                        self.state[p] = {
                            'momentum_buffer': torch.zeros_like(p),
                        }

                    # Update v
                    # v = B1 * v + (1-B) * g
                    v = self.state[p]['momentum_buffer']
                    v.lerp_(p.grad, 1 - group['momentum'])

                    # Optional Nesterov look-ahead
                    # vv = B*v + (1-B)*g
                    vv = p.grad.lerp(v, group['momentum']) if group['nesterov'] else v

                    # Update
                    update = zeropower_via_polar_express(vv, group['ns_steps'])
                    lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5
                    if group['weight_decay'] != 0:
                        # Decoupled Cautious Weight Decay
                        mask = (update * p) >= 0
                        p.sub_(lr * update + lr * group['weight_decay'] * p * mask)
                    else:
                        p.sub_(lr * update)

                    input_tensor = p
                else:
                    input_tensor = torch.zeros_like(group['zero_buffer'])

                # Sync point 2
                output_params = [p for p in group['params'][i:i+world_size]]
                if len(output_params) < world_size:
                    output_params.extend(torch.zeros_like(group['zero_buffer']) for _ in range(world_size - len(output_params)))
                torch.distributed.all_gather(output_params, input_tensor)


