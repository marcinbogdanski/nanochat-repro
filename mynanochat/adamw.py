import torch

class Adam(torch.optim.Optimizer):
    """Adam/AdamW optimizer
    
    Algorithm:
        v = B1 * v + (1 - B1) * g          # first moment (direction)
        s = B2 * s + (1 - B2) * g^2        # second moment (scaling)

        v_corrected = v / (1 - B1^t)       # bias correction
        s_corrected = s / (1 - B2^t)

        p = p - lr * v_corrected / (sqrt(s_corrected) + eps)
        p = p - lr * wd * p                # AdamW: decoupled weight decay
    """
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, decoupled_weight_decay=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decoupled_weight_decay=decoupled_weight_decay)
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
                        'step': 0,
                        'exp_avg': torch.zeros_like(p),
                        'exp_avg_sq': torch.zeros_like(p),
                    }
                self.state[p]['step'] += 1

                # Weight Decay
                grad = p.grad
                if group['weight_decay'] != 0.0:
                    if not group['decoupled_weight_decay']:
                        # Adam
                        # equivalent to: loss = ... + weight_decay/2 * W**2
                        grad = grad.add(p, alpha=group['weight_decay'])  # not in place!
                    else:
                        # AdamW
                        # p = p - lr * weight_decay * p
                        p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update v
                # v = B1 * v + (1-B1) * g
                v = self.state[p]['exp_avg']
                v.lerp_(grad, 1 - group['betas'][0])

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
                bias2_sqrt = bias2**0.5
                denom = (s.sqrt() / bias2_sqrt).add_(group['eps'])
                p.data.addcdiv_(v, denom, value=-group['lr'] / bias1)

