import math
import torch

@torch.inference_mode()
def evaluate_bpb(model, token_bytes, eval_loader, eval_steps, device):
    total_nats = torch.tensor(0.0, device=device, dtype=torch.float32)
    total_bytes = torch.tensor(0, device=device, dtype=torch.int64)
    eval_loader.reset()
    for _ in range(eval_steps):
        x, y = eval_loader.get_batch_bos()
        assert (y >= 0).all()  # masking with -1 not supported
        x = x.to(device)
        y = y.to(device)
        _, loss_arr = model(x, y, reduction='none', return_logits=False)
        bytes_arr = token_bytes[y.view(-1)]
        loss_arr = loss_arr * (bytes_arr > 0)   # zero loss for tokens with 0 bytes (<bos> etc.)
        total_nats += loss_arr.sum().item()
        total_bytes += bytes_arr.sum().item()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total_nats, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_bytes, op=torch.distributed.ReduceOp.SUM)
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    bpb = float('inf')
    if total_bytes > 0:
        bpb = total_nats / (total_bytes * math.log(2))

    return bpb, total_nats, total_bytes
