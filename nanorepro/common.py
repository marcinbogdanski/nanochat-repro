import os
import json
import datetime
import torch
import wandb
import requests
import tempfile

def get_base_path():
    """Returns the base path for storing logs and checkpoints."""
    base_path = os.environ.get('NANOREPRO_BASE_PATH', os.path.expanduser("~/.cache/nanorepro"))
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

def wandb_init(project, run_name, user_config, ddp_master):
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
        wandb_logger = wandb.init(project=project, name=run_name, config=user_config, dir=get_base_path())
    else:
        wandb_logger = WandBDummy()
    return wandb_logger


def download_file_rank0(filepath, url):
    """Downloads a file from a URL if it does not exist, only on rank 0."""
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0 and not os.path.exists(filepath):
        folder_path = os.path.dirname(filepath)
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)
        r = requests.get(url)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(r.content)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # wait for rank 0

class FileLogger:
    def __init__(self, run_path, resume_from_step=None):
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.log_filepath = os.path.join(run_path, f"train_log_rank{self.rank}.jsonl")
        if resume_from_step is None:
            os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
            open(self.log_filepath, "w").close()  # clear the file if not resuming
        else:
            # Rewrite the log file to remove entries with step >= resume_from_step
            # Otherwise resume just writes more into old log, which may have more steps, imagine:
            # run 0..345 steps crashes, then resume from checkpoint at 250 appends: 0..345,250.. and so on
            with tempfile.NamedTemporaryFile('w', dir=run_path, delete=False) as tmp:
                tmp_path = tmp.name
                try:
                    with open(self.log_filepath, 'r') as src:
                        for line in src:
                            obj = json.loads(line)
                            keep = (
                                obj["event"] != "run_summary"
                                and (obj["step"] is None or obj["step"] < resume_from_step)
                            )
                            if keep:
                                tmp.write(line)  # preserve original formatting
                except Exception:
                    os.remove(tmp_path)
                    raise
            os.replace(tmp_path, self.log_filepath)

    def log0(self, event, step, data):
        if self.rank == 0:
            self.log(event, step, data)

    def log(self, event, step, data):
        datetime_iso = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
        with open(self.log_filepath, 'a') as f:
            json.dump({'timestamp': datetime_iso, 'event': event, 'step': step, 'rank': self.rank, **data}, f)
            f.write('\n')
