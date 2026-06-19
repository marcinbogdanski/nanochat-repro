import os
import torch
import torch.nn.functional as F
from nanorepro.muon import Muon, DistMuon

# NOTE: If you disable fp16 in newtonschulz, the diff is <1e-06
# The current DistMuon matches NanoChat bit-wise on this test (yay!)
# Possibly, if we did ref path in two chunks it would match


ddp = int(os.environ.get('RANK', -1)) != -1  # is this ddp run?
assert ddp

ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
ddp_master = ddp_rank == 0  # is this a master?
device = f'cuda:{ddp_local_rank}'
device_type = 'cuda'
assert torch.cuda.is_available()
torch.cuda.set_device(device)
torch.distributed.init_process_group(backend='nccl', device_id=ddp_local_rank)  # device_id= to suppress barrier warning

assert ddp_world_size == 2

# Seed
torch.manual_seed(42)

# Forward + backward
x1 = torch.randn(16, 32, device=device)
x2 = x1[:8].clone().detach() if ddp_rank == 0 else x1[8:].clone().detach()
target1 = torch.randn(16, 32, device=device)
target2 = target1[:8].clone().detach() if ddp_rank == 0 else target1[8:].clone().detach()

# Create identical weights
W1 = torch.randn(32, 32, requires_grad=True, device=device)
W2 = torch.randn(32, 32, requires_grad=True, device=device)
W3 = W1.clone().detach().requires_grad_(True)
W4 = W2.clone().detach().requires_grad_(True)

# Params
lr = 0.02
momentum = 0.95
nesterov = True
ns_steps = 5

# Optimizers
opt_ref = Muon([W1, W2], lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=0.0)
opt_dist = DistMuon([W3, W4], lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=0.0)

for i in range(20):
    if ddp_master:
        opt_ref.zero_grad()
        out1 = F.relu(x1 @ W1.T) @ W2.T
        loss1 = ((out1 - target1) ** 2).mean()
        loss1.backward()
        opt_ref.step()

    opt_dist.zero_grad()
    out2 = F.relu(x2 @ W3.T) @ W4.T   # half of x, this is per rank
    loss2 = ((out2 - target2) ** 2).mean()
    loss2.backward()
    opt_dist.step()

    if ddp_master:
        state1 = opt_ref.state[W1]
        state2 = opt_ref.state[W2]
        state3 = opt_dist.state[W3]
        state4 = opt_dist.state[W4]
        
        assert list(state1.keys()) == ['momentum_buffer']
        assert list(state2.keys()) == ['momentum_buffer']

        weight_max_diff = (W1 - W3).abs().max().item()
        weight_max_diff += (W2 - W4).abs().max().item()
        
        # Final diff 0.00451226532459259 - NanoChat has exact same value
        print(f"Diff {weight_max_diff}    {W1.sum().item()}    {W2.sum().item()}   {W3.sum().item()}   {W4.sum().item()}")

if ddp_master and weight_max_diff == 0.0:
    print(f"All good after {i+1} iterations!")

if ddp:
    torch.distributed.destroy_process_group()
