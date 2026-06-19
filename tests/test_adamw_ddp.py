import os
import torch
from nanorepro.adamw import AdamW, DistAdamW




ddp = int(os.environ.get('RANK', -1)) != -1  # is this ddp run?
if ddp:
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    ddp_master = ddp_rank == 0  # is this a master?
    device = f'cuda:{ddp_local_rank}'
    device_type = 'cuda'
    assert torch.cuda.is_available()
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend='nccl', device_id=ddp_local_rank)  # device_id= to suppress barrier warning
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    ddp_master = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = device





# Seed
torch.manual_seed(42)

# Forward + backward
x = torch.randn(16, 32, device=device)
target = torch.randn(16, 64, device=device)

# Create identical weights
W1 = torch.randn(64, 32, requires_grad=True, device=device)
W2 = W1.clone().detach().requires_grad_(True)

# Params
lr = 0.02

# Optimizers
opt_ref = AdamW([W2], lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
opt_dist = DistAdamW([W1], lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

for i in range(20):
    opt_dist.zero_grad()
    opt_ref.zero_grad()
    loss1 = ((x @ W1.T - target) ** 2).mean()
    loss2 = ((x @ W2.T - target) ** 2).mean()
    loss1.backward()
    loss2.backward()

    opt_dist.step()
    opt_ref.step()

    state1 = opt_dist.state[W1]
    state2 = opt_ref.state[W2]
    assert list(state1.keys()) == ['step', 'exp_avg', 'exp_avg_sq']
    assert state1['step'] == state2['step']

    weight_max_diff = (W1 - W2).abs().max().item()
    # assert weight_max_diff == 0.0
    print(f"Diff {weight_max_diff}    {W1.sum().item()}    {W2.sum().item()}")

if weight_max_diff == 0.0:
    print(f"All good after {i+1} iterations!")

if ddp:
    torch.distributed.destroy_process_group()
