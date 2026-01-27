import torch
from mynanochat.muon_karpathy import MuonK
from mynanochat.muon_torch import MuonT

# Forward + backward
x = torch.randn(16, 32)
target = torch.randn(16, 64)

# Create identical weights
W1 = torch.randn(64, 32, requires_grad=True)
W2 = W1.clone().detach().requires_grad_(True)

# Params
lr = 0.02
momentum = 0.95
nesterov = True
ns_steps = 5

# Optimizers
opt_torch = MuonT([W1], lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                             weight_decay=0, adjust_lr_fn="original")
opt_custom = MuonK([W2], lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

# NOTE: this passes, prints 0.0 0.0 for all iterations
for i in range(20):
    opt_torch.zero_grad()
    opt_custom.zero_grad()
    loss1 = ((x @ W1.T - target) ** 2).mean()
    loss2 = ((x @ W2.T - target) ** 2).mean()
    loss1.backward()
    loss2.backward()
    opt_torch.step()
    opt_custom.step()

    weight_max_diff = (W1 - W2).abs().max().item()
    # compare optmizer states
    state1 = opt_torch.state[W1]
    state2 = opt_custom.state[W2]
    momentum_buffer_diff = (state1['momentum_buffer'] - state2['momentum_buffer']).abs().max().item()
    print(i, weight_max_diff, momentum_buffer_diff)
