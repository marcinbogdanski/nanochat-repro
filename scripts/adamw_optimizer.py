import os
if os.environ.get("RANK", "0") == "0":  # set on rank 0 only
    os.environ["TORCH_LOGS"] = "graph_breaks,recompiles"
else:
    os.environ.pop("TORCH_LOGS", None)
import time
import torch
import torch.nn as nn
from mynanochat.adamw import AdamW, DistAdamW
# from mynanochat.muon_karpathy import Muon as MuonKarpathy
# from mynanochat.muon_karpathy import DistMuon as DistMuonKarpathy

# Run like this
# CUDA_VISIBLE_DEVICES=0 python -m scripts.adamw_optimizer
# CUDA_VISIBLE_DEVICES=0,2 OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.adamw_optimizer

def print0(s="",**kwargs):
    ddp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if ddp_rank == 0:
        print(s, **kwargs)

def schedule_lr_etc(adam_optimizer):
    # Simulate Scheduler
    for group in adam_optimizer.param_groups:
        group['lr'] *= 0.99


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

    embedding_lr = 0.3
    unembedding_lr = 0.004
    scalar_lr = 0.5

    params_lm_head = [
        torch.nn.parameter.Parameter(torch.randn(65536, 256, dtype=torch.float32, device=device))
    ]
    params_embedding = [
        torch.nn.parameter.Parameter(torch.randn(65536, 256, dtype=torch.bfloat16, device=device))
    ]
    params_resid = [
        torch.nn.parameter.Parameter(torch.randn(4, dtype=torch.float32, device=device))
    ]
    params_x0 = [
        torch.nn.parameter.Parameter(torch.randn(4, dtype=torch.float32, device=device))
    ]

    adam_groups = [
        {
            'params': params_lm_head,
            'lr': unembedding_lr,
            'is_small': False,
        },
        {
            'params': params_embedding,
            'lr': embedding_lr,
            'is_small': False,
        },
        {
            'params': params_resid,
            'lr': scalar_lr * 0.01,
            'is_small': True,
        },
        {
            'params': params_x0,
            'lr': scalar_lr,
            'is_small': True,
        }
    ]

    # My version
    adamw_factory = DistAdamW if ddp else AdamW
    adamw_optimizer = adamw_factory(
        adam_groups,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
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
    for group in adam_groups:
        for p in group['params']:
            p.grad = torch.randn_like(p)

    # Warmup
    for i in range(5):
        adamw_optimizer.step()
        schedule_lr_etc(adamw_optimizer)

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
        adamw_optimizer.step()
        schedule_lr_etc(adamw_optimizer)
        torch.cuda.synchronize()

        adamw_optimizer.step()
        schedule_lr_etc(adamw_optimizer)
        torch.cuda.synchronize()

    if ddp:
        torch.distributed.barrier()
    torch.cuda.synchronize()
    ts = time.time()

    print0(" -------- HOT ITER START --------")
    for i in range(100):
        adamw_optimizer.step()
        schedule_lr_etc(adamw_optimizer)
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
    prof.export_chrome_trace(f"adamw_optimizer_trace_rank{ddp_rank}.json")

    # Print sum of all params to verify that they are changing
    total_sum = 0.0
    for group in adam_groups:
        for p in group['params']:
            total_sum += p.sum().item()
    print0(f"Total sum of all params: {total_sum}")

    print0("Done")

    if ddp:
        torch.distributed.destroy_process_group()

    
if __name__ == "__main__":
    main()







