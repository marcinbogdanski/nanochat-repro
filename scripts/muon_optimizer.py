import os
os.environ['TORCH_LOGS'] = "graph_breaks,recompiles"
import time
import torch
import torch.nn as nn
from mynanochat.muon import Muon

# Run like this
# python -m scripts.muon_optimizer


params_def_d4 = [((256, 256), 16), ((256, 1024), 4), ((1024, 256), 4)]
params_def_d12 = [((768, 768), 48), ((768, 3072), 12), ((3072, 768), 12)]


def main():

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    matrix_lr = 0.02
    weight_decay = 0.2

    muon_groups = []
    for shape, count in params_def_d12:
        group_params = [torch.nn.parameter.Parameter(torch.randn(*shape, device='cuda')) for _ in range(count)]
        muon_groups.append({'params': group_params})

    muon_optimizer = Muon(
        muon_groups,
        lr=matrix_lr,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=weight_decay,
    )

    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"Memory allocated after optimizer init: {mem_alloc:.2f} GB")

    # Set grads
    for group in muon_groups:
        for p in group['params']:
            p.grad = torch.randn_like(p)

    # Warmup
    for i in range(5):
        muon_optimizer.step()

    torch.cuda.reset_peak_memory_stats()
    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"Memory allocated after warmup: {mem_alloc:.2f} GB")

    torch.cuda.synchronize()
    ts = time.time()
    for i in range(100):
        muon_optimizer.step()
    torch.cuda.synchronize()

    te = time.time()
    print(f"Time taken for 100 steps: {te - ts} seconds")

    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Max memory allocated during 100 steps: {max_mem:.2f} GB")

    # Print sum of all params to verify that they are changing
    total_sum = 0.0
    for group in muon_groups:
        for p in group['params']:
            total_sum += p.sum().item()
    print(f"Total sum of all params: {total_sum}")

    print("Done")


    
if __name__ == "__main__":
    main()







