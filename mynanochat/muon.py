import torch

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
    X.div_(X.norm().clamp(min=eps))
    for _ in range(steps):
        A = X @ X.T
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)
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
