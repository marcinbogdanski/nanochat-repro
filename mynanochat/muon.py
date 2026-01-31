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
    eps=1e-7
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

                # Decoupled Weight Decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update v
                # v = B1 * v + (1-B) * g
                v = self.state[p]['momentum_buffer']
                v.lerp_(p.grad, 1 - group['momentum'])

                # Optional Nesterov look-ahead
                # vv = B*v + (1-B)*g
                vv = p.grad.lerp(v, group['momentum']) if group['nesterov'] else v

                # Update
                update = zeropower_via_newtonschulz(vv, group['ns_steps'])
                lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5
                p.add_(update, alpha=-lr)



class DistMuon(torch.optim.Optimizer):
    """ZeRO-2 version of Muon optimizer"""
    def __init__(self, params, lr=0.01, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1, rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        self.rank = rank
        self.world_size = world_size
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):

        # Assert all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"])

        # Sync point 1
        # This will reduce scatter grads, such that each rank gets full averated grad for owned param
        for group in self.param_groups:
            assert len(group['params']) % self.world_size == 0
            for i in range(0, len(group['params']), self.world_size):
                input_grads = [p.grad for p in group['params'][i:i+self.world_size]]
                torch.distributed.reduce_scatter(
                    group['params'][i+self.rank].grad,
                    input_grads,
                    op=torch.distributed.ReduceOp.AVG
                )
            
        for group in self.param_groups:
            for i in range(0, len(group['params']), self.world_size):
                p = group['params'][i+self.rank]

                # Lazy Init
                if p not in self.state:
                    self.state[p] = {
                        'momentum_buffer': torch.zeros_like(p),
                    }

                # Decoupled Weight Decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update v
                # v = B1 * v + (1-B) * g
                v = self.state[p]['momentum_buffer']
                v.lerp_(p.grad, 1 - group['momentum'])

                # Optional Nesterov look-ahead
                # vv = B*v + (1-B)*g
                vv = p.grad.lerp(v, group['momentum']) if group['nesterov'] else v

                # Update
                update = zeropower_via_newtonschulz(vv, group['ns_steps'])
                lr = group['lr'] * (max(1, p.size(0) / p.size(1)))**0.5
                p.add_(update, alpha=-lr)

                # Sync point 2
                input_params = [p for p in group['params'][i:i+self.world_size]]
                torch.distributed.all_gather(input_params, p)


