import os
if os.environ.get("RANK", "0") == "0":  # set on rank 0 only
    os.environ["TORCH_LOGS"] = "graph_breaks,recompiles"
else:
    os.environ.pop("TORCH_LOGS", None)
import time
import torch
import torch.nn as nn
from mynanochat.muon import Muon, DistMuon
# from mynanochat.muon_karpathy import Muon as MuonKarpathy
# from mynanochat.muon_karpathy import DistMuon as DistMuonKarpathy

# Run like this
# python -m scripts.muon_optimizer
# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.muon_optimizer

params_def_d4 = [((256, 256), 16), ((256, 1024), 4), ((1024, 256), 4)]
params_def_d12 = [((768, 768), 48), ((768, 3072), 12), ((3072, 768), 12)]

def print0(s="",**kwargs):
    ddp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if ddp_rank == 0:
        print(s, **kwargs)

def schedule_lr_etc(muon_optimizer):
    # Simulate Scheduler
    for group in muon_optimizer.param_groups:
        group['lr'] *= 0.99
        group['momentum'] *= 0.99
        group['weight_decay'] *= 0.99


def main():
    assert torch.cuda.is_available()

    # DDP Init
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this ddp run?
    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
        # device_id= to suppress barrier warning
        torch.distributed.init_process_group(backend='nccl', device_id=ddp_local_rank)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        device = 'cuda'

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    matrix_lr = 0.02
    weight_decay = 0.2

    muon_groups = []
    for shape, count in params_def_d12:
        group_params = [
            torch.nn.parameter.Parameter(torch.randn(*shape, device=device)) for _ in range(count)
        ]
        muon_groups.append({'params': group_params})

    # My version
    muon_factory = DistMuon if ddp else Muon
    muon_optimizer = muon_factory(
        muon_groups,
        lr=matrix_lr,
        momentum=0.95,
        ns_steps=5,
        weight_decay=weight_decay,
    )
    # Karpathy version
    # muon_factory = DistMuonKarpathy if ddp else MuonKarpathy
    # all_params = [p for group in muon_groups for p in group['params']]
    # muon_optimizer = muon_factory(
    #     all_params,
    #     lr=matrix_lr,
    #     momentum=0.95,
    #     ns_steps=5,
    #     weight_decay=weight_decay,
    # )

    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    print0(f"Memory allocated after optimizer init: {mem_alloc:.2f} GB")

    # Set grads
    for group in muon_groups:
        for p in group['params']:
            p.grad = torch.randn_like(p)

    # Warmup
    for i in range(5):
        muon_optimizer.step()
        schedule_lr_etc(muon_optimizer)

    torch.cuda.reset_peak_memory_stats()
    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    print0(f"Memory allocated after warmup: {mem_alloc:.2f} GB")

    if ddp:
        torch.distributed.barrier()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        muon_optimizer.step()
        schedule_lr_etc(muon_optimizer)
        torch.cuda.synchronize()

        muon_optimizer.step()
        schedule_lr_etc(muon_optimizer)
        torch.cuda.synchronize()

    if ddp:
        torch.distributed.barrier()
    torch.cuda.synchronize()
    ts = time.time()

    print0(" -------- HOT ITER START --------")
    for i in range(100):
        muon_optimizer.step()
        schedule_lr_etc(muon_optimizer)
    print0(" -------- HOT ITER END --------")

    if ddp:
        torch.distributed.barrier()
    torch.cuda.synchronize()

    te = time.time()
    print0(f"Time taken for 100 steps: {te - ts} seconds")

    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print0(f"Max memory allocated during 100 steps: {max_mem:.2f} GB")

    print0("---")
    print0(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print0("---")
    prof.export_chrome_trace(f"muon_optimizer_trace_rank{ddp_rank}.json")

    # Print sum of all params to verify that they are changing
    total_sum = 0.0
    for group in muon_groups:
        for p in group['params']:
            total_sum += p.sum().item()
    print0(f"Total sum of all params: {total_sum}")

    print0("Done")

    if ddp:
        torch.distributed.destroy_process_group()

    
if __name__ == "__main__":
    main()







