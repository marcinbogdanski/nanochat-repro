import torch

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
            for p in group['params']:
                if p.grad is None:
                    continue
                # Lazy Init
                if p not in self.state:
                    self.state[p] = {
                        'step': torch.tensor(0, dtype=torch.int64, device=p.device),
                        'exp_avg': torch.zeros_like(p),
                        'exp_avg_sq': torch.zeros_like(p),
                    }
                self.state[p]['step'] += 1

                # Weight Decay
                grad = p.grad
                if group['weight_decay'] != 0.0:
                    # AdamW
                    # p = p - lr * weight_decay * p
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update v
                # v = B1 * v + (1-B1) * g
                v = self.state[p]['exp_avg']
                v.mul_(group['betas'][0]).add_(grad, alpha=1-group['betas'][0])

                # Update s
                # s = B2 * s + (1-B2) * g**2
                s = self.state[p]['exp_avg_sq']
                s.mul_(group['betas'][1])
                s.addcmul_(grad, grad, value=1-group['betas'][1])

                # Correction
                # Somewhat convoluted way to do:
                # v_corrected = v / (1-B1**t)
                # s_corrected = s / (1-B2**t)
                # p = p - lr * v_corrected / (sqrt(s_corrected)+eps)
                t = self.state[p]['step']
                bias1 = 1-group['betas'][0]**t
                bias2 = 1-group['betas'][1]**t
                denom = (s / bias2).sqrt().add_(group['eps'])
                update = v.div(denom).mul_(group['lr'] / bias1)
                p.data.add_(update, alpha=-1.0)

