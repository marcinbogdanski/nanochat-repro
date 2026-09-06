import math
import torch

@torch.inference_mode()
def evaluate_bpb(model, token_bytes, eval_loader, eval_steps, device):
    total_nats = torch.tensor(0.0, device=device, dtype=torch.float32)
    total_bytes = torch.tensor(0, device=device, dtype=torch.int64)
    eval_loader.reset()

    # To allow indexing token_bytes[y] where y may contain -1 (ignore_index), we d a trick:
    # we extend token_bytes with a single element 0, so that token_bytes[-1] == 0.
    # Now indexing token_bytes[y] for y==-1 will yield 0, exactly what we want:
    # value 0 will not contribute to total_bytes or total_nats, or BPB,
    # which means indexing with -1 means "ignore this token" as intended.
    token_bytes = torch.cat([token_bytes, torch.tensor([0], device=device, dtype=token_bytes.dtype)])

    was_training = model.training
    model.eval()
    try:
        for _ in range(eval_steps):
            x, y = eval_loader.get_batch_bos()
            _, loss_arr, _ = model(x, y, reduction='none', return_logits=False, use_compiled_if_available=True)  # compiled ok here
            assert (y[y < 0] == -1).all()  # assert if negative value exists, it must be -1
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
    finally:
        model.train(was_training)
