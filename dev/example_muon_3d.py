import os
import time
import torch
import torch.nn as nn
from mynanochat.muon import Muon, DistMuon

# Run like this
# CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python -m dev.example_muon_3d

params_def_d12 = [((768, 768), 48), ((768, 3072), 12), ((3072, 768), 12)]

def print0(s="",**kwargs):
    ddp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if ddp_rank == 0:
        print(s, **kwargs)

def main():
    assert torch.cuda.is_available()
    device = 'cuda'

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    matrix_lr = 0.02
    weight_decay = 0.2
    momentum = 0.95
    ns_steps = 5
    beta2 = 0.95

    # Init params
    muon_groups_2d = []
    muon_groups_3d = []
    for shape, count in params_def_d12:
        data = torch.randn(*shape, device=device)
        # expand data to 3D by just duplicating and stacking
        # data = data.unsqueeze(0).expand(count, -1, -1).contiguous()  # count, *shape
        # data = data.unsqueeze(0)
        # Match real model params: same initial values, but distinct storage per Parameter.
        group_params_2d = [torch.nn.parameter.Parameter(data.clone()) for _ in range(count)]
        muon_groups_2d.append({
            'params': group_params_2d,
            'lr': matrix_lr,
            'momentum': momentum,
            'ns_steps': ns_steps,
            'beta2': beta2,
            'weight_decay': weight_decay,
        })
        group_params_3d = [torch.nn.parameter.Parameter(data.clone().unsqueeze(0)) for _ in range(count)]
        muon_groups_3d.append({
            'params': group_params_3d,
            'lr': matrix_lr,
            'momentum': momentum,
            'ns_steps': ns_steps,
            'beta2': beta2,
            'weight_decay': weight_decay,
        })

    # My version
    muon_optimizer_2d = Muon(muon_groups_2d)
    muon_optimizer_3d = Muon(muon_groups_3d)

    # Set grads
    for group_2d, group_3d in zip(muon_groups_2d, muon_groups_3d):
        for p_2d, p_3d in zip(group_2d['params'], group_3d['params']):
            grad_2d = torch.randn_like(p_2d)
            p_2d.grad = grad_2d.clone()
            p_3d.grad = grad_2d.unsqueeze(0).clone()

    # Optimizer step
    muon_optimizer_2d.step()
    muon_optimizer_3d.step()

    # Print sum of all params to verify that they are changing
    total_sum_2d = 0.0
    total_sum_3d = 0.0
    for group_2d, group_3d in zip(muon_groups_2d, muon_groups_3d):
        for p_2d, p_3d in zip(group_2d['params'], group_3d['params']):
            total_sum_2d += p_2d.sum().item()
            total_sum_3d += p_3d.sum().item()
    print0(f"Total sum of all 2D params: {total_sum_2d}")
    print0(f"Total sum of all 3D params: {total_sum_3d}")

    print0("Done")


    
if __name__ == "__main__":
    main()



