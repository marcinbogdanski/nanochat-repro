import os
import json
import torch

def save_checkpoint(checkpoints_path, model, optimizers, dataloader, loop_vars, user_config):
    os.makedirs(checkpoints_path, exist_ok=True)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    step = loop_vars['step']

    model_md5sum = None
    if rank == 0:
        # Metadata
        metadata = {
            'step': step,
            'total_time': loop_vars['total_time'],
            'smooth_tloss': loop_vars['smooth_tloss'],
            'model_config': model.config.to_dict(),
            'user_config': user_config,
        }
        meta_path = os.path.join(checkpoints_path, f"meta_{step:06d}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        
        # Model state
        model_path = os.path.join(checkpoints_path, f"model_{step:06d}.pt")
        torch.save(model.state_dict(), model_path)
        model_md5sum = os.popen(f"md5sum {model_path}").read().split()[0]
    
    # Optimizer state
    adamw_opt, muon_opt = optimizers
    optim_path = os.path.join(checkpoints_path, f"optim_{step:06d}_rank{rank:d}.pt")
    torch.save({'adamw': adamw_opt.state_dict(), 'muon': muon_opt.state_dict()}, optim_path)
    
    # Dataloader state
    dataloader_path = os.path.join(checkpoints_path, f"dataloader_{step:06d}_rank{rank:d}.pt")
    torch.save(dataloader.state_dict(), dataloader_path)

    return model_md5sum

def load_checkpoint(checkpoints_path, model, optimizers, dataloader, device, step=None):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # If step is not specified, load the latest checkpoint
    if step is None:
        fn_list = [fn for fn in os.listdir(checkpoints_path) if fn.startswith("meta_") and fn.endswith(".json")]
        steps_list = [int(fn[len("meta_"):-len(".json")]) for fn in fn_list]
        step = max(steps_list)  # Throws if no checkpoints found

    meta_path = os.path.join(checkpoints_path, f"meta_{step:06d}.json")
    model_path = os.path.join(checkpoints_path, f"model_{step:06d}.pt")
    optim_path = os.path.join(checkpoints_path, f"optim_{step:06d}_rank{rank:d}.pt")
    dataloader_path = os.path.join(checkpoints_path, f"dataloader_{step:06d}_rank{rank:d}.pt")

    # Metadata / loop vars
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    loop_vars = {
        "step": metadata["step"],
        "total_time": metadata["total_time"],
        "smooth_tloss": metadata["smooth_tloss"],
    }

    # Model state
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)

    # Optimizer state
    adamw_opt, muon_opt = optimizers
    optim_state = torch.load(optim_path, map_location=device)
    adamw_opt.load_state_dict(optim_state["adamw"])
    muon_opt.load_state_dict(optim_state["muon"])

    # Dataloader state
    dataloader_state = torch.load(dataloader_path, map_location="cpu")
    dataloader.load_state_dict(dataloader_state)

    return loop_vars
