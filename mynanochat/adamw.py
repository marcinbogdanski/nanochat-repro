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



class DistAdamW(torch.optim.Optimizer):
    """ZeRO-2 version of AdamW optimizer"""
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, fused=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Lazy Init
                slice_width = p.size(0) // world_size
                slice_start = rank * slice_width
                slice_end = slice_start + slice_width

                if p not in self.state:
                    self.state[p] = {
                        'step': torch.tensor(0, dtype=torch.int64, device=p.device),
                        'exp_avg': torch.zeros_like(p[:slice_width]),
                        'exp_avg_sq': torch.zeros_like(p[:slice_width]),
                    }
                self.state[p]['step'] += 1

                # Weight Decay
                if group['weight_decay'] != 0.0:
                    # AdamW
                    # p = p - lr * weight_decay * p
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Sync point 1
                grad_slice = torch.empty_like(p.grad[:slice_width])
                torch.distributed.reduce_scatter_tensor(grad_slice, p.grad, op=torch.distributed.ReduceOp.AVG)

                # Update v
                # v = B1 * v + (1-B1) * g
                v = self.state[p]['exp_avg']
                v.mul_(group['betas'][0]).add_(grad_slice, alpha=1-group['betas'][0])

                # Update s
                # s = B2 * s + (1-B2) * g**2
                s = self.state[p]['exp_avg_sq']
                s.mul_(group['betas'][1])
                s.addcmul_(grad_slice, grad_slice, value=1-group['betas'][1])

                # Correction
                # Somewhat convoluted way to do:
                # v_corrected = v / (1-B1**t)
                # s_corrected = s / (1-B2**t)
                # p = p - lr * v_corrected / (sqrt(s_corrected)+eps)
                t = self.state[p]['step']
                bias1 = 1-group['betas'][0]**t
                bias2 = 1-group['betas'][1]**t
                denom = (s / bias2).sqrt().add_(group['eps'])
                update = v.div(denom).mul_(-group['lr'] / bias1)
                p_slice = p[slice_start:slice_end] + update

                # Sync point 2
                torch.distributed.all_gather_into_tensor(p, p_slice)

