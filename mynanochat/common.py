import os
import json
import datetime
import torch
import wandb

def get_base_path():
    """Returns the base path for storing logs and checkpoints."""
    base_path = os.environ.get('MYNANOCHAT_BASE_PATH', os.path.expanduser("~/.cache/mynanochat"))
    os.makedirs(base_path, exist_ok=True)
    return base_path

def ddp_init():
    """Initializes DDP if applicable, returns device, ddp_master, ddp_world_size."""
    # DDP Init
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this ddp run?
    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        ddp_master = ddp_rank == 0  # is this a master?
        device = f'cuda:{ddp_local_rank}'
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend='nccl', device_id=ddp_local_rank)  # device_id= to suppress barrier warning
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        ddp_master = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if ddp_master:
        print(f"Init: {ddp=} {ddp_rank=}, {ddp_local_rank=}, {ddp_world_size=}, {ddp_master=}, {device=}")
    return device, ddp_master, ddp_world_size

def wandb_init(run_name, user_config, ddp_master):
    """Initializes WandB if applicable, returns the logger."""

    # Dummy WandB Logger to simplify calls in the training loop
    class WandBDummy:
        def __init__(self):
            pass
        def log(self, *args, **kwargs):
            pass
        def finish(self):
            pass

    # WandB Init
    if run_name is not None and ddp_master:
        wandb_logger = wandb.init(project="nanochat", name=run_name, config=user_config, dir=get_base_path())
    else:
        wandb_logger = WandBDummy()
    return wandb_logger


class FileLogger:
    def __init__(self, run_path):
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.log_filepath = os.path.join(run_path, f"train_log_rank{self.rank}.jsonl")
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)

    def log0(self, event, step, data, override=False):
        if self.rank == 0:
            self.log(event, step, data, override=override)

    def log(self, event, step, data, override=False):
        datetime_iso = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        with open(self.log_filepath, 'a' if not override else 'w') as f:
            json.dump({'timestamp': datetime_iso, 'event': event, 'step': step, 'rank': rank, **data}, f)
            f.write('\n')
