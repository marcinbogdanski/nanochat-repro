import os
import json
import torch

def save_checkpoint(checkpoints_path, model, step, user_config):
    os.makedirs(checkpoints_path, exist_ok=True)

    # Model state
    model_path = os.path.join(checkpoints_path, f"model_{step:06d}.pt")
    torch.save(model.state_dict(), model_path)
    
    # Metadata
    metadata = {
        'step': step,
        'model_config': model.config.to_dict(),
        'user_config': user_config,
    }
    meta_path = os.path.join(checkpoints_path, f"meta_{step:06d}.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    
    # Calculate MD5 sum of saved file by running os command
    md5sum = os.popen(f"md5sum {model_path}").read().split()[0]
    return md5sum
