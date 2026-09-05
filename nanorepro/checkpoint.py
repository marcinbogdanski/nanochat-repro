import os
import json
import torch
from nanorepro.gpt import GPTModel, GPTConfig

def save_checkpoint(checkpoints_path, model, optimizers, dataloader, loop_vars, user_config, training_hyperparameters):
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
            'training_hyperparameters': training_hyperparameters,
        }
        meta_path = os.path.join(checkpoints_path, f"meta_{step:06d}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        
        # Model state
        model_path = os.path.join(checkpoints_path, f"model_{step:06d}.pt")
        model_state = model.state_dict()
        # Model params may be views into static buffers created by optimizer. Here we save them as clean, independent copies.
        # Preferably I would just save them as clean CPU tensors, but I want MD5 checkpoint compatibility with Nanochat,
        # so instead I preserve original container class and _metadata and keep tensors on GPU. Prob cleanup later.
        # Also, this GPU copy creates unnecessary memory pressure point during save, but at that point fwd/bwd/optim are dormant, so hopefully ok for now.
        model_state_meta = model_state._metadata
        model_state = model_state.__class__((name, tensor.detach().clone()) for name, tensor in model_state.items())
        model_state._metadata = model_state_meta
        torch.save(model_state, model_path)
        model_md5sum = os.popen(f"md5sum {model_path}").read().split()[0]
    
    # Optimizer state
    adamw_opt, muon_opt = optimizers
    optim_path = os.path.join(checkpoints_path, f"optim_{step:06d}_rank{rank:d}.pt")
    torch.save({'adamw': adamw_opt.state_dict(), 'muon': muon_opt.state_dict()}, optim_path)
    
    # Dataloader state
    dataloader_path = os.path.join(checkpoints_path, f"dataloader_{step:06d}_rank{rank:d}.pt")
    torch.save(dataloader.state_dict(), dataloader_path)

    return model_md5sum


def get_latest_checkpoint_step(checkpoints_path):
    fn_list = [fn for fn in os.listdir(checkpoints_path) if fn.startswith("meta_") and fn.endswith(".json")]
    steps_list = [int(fn[len("meta_"):-len(".json")]) for fn in fn_list]
    return max(steps_list)  # Throws if no checkpoints found


def load_checkpoint(checkpoints_path, model, optimizers, dataloader, device, step=None):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # If step is not specified, load the latest checkpoint
    if step is None:
        step = get_latest_checkpoint_step(checkpoints_path)

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

def create_model(model_config, compute_dtype, enable_fa3, fp8_training, backward_overlap, enable_metrics, device):
    """Create a GPT model on device and call init_weights(). Returns ready, non-compiled model."""
    with torch.device("meta"):
        model = GPTModel(
            model_config,
            compute_dtype=compute_dtype,
            enable_fa3=enable_fa3,
            fp8_training=fp8_training,
            backward_overlap=backward_overlap,
            enable_metrics=enable_metrics,
        )
    model.to_empty(device=device)
    model.init_weights()  # RoPE buffers, random weight init
    return model

def load_model(checkpoints_path, compute_dtype, enable_fa3, fp8_training, backward_overlap, enable_metrics, device, step=None):
    """Load a GPT model from checkpoint to device. Returns ready, non-compiled model in eval mode and loaded metadata."""
    checkpoint_step = get_latest_checkpoint_step(checkpoints_path) if step is None else step

    # Load Model Metadata
    metadata_path = os.path.join(checkpoints_path, f"meta_{checkpoint_step:06d}.json")
    with open(metadata_path, "r") as f:
        model_metadata = json.load(f)
    assert model_metadata["step"] == checkpoint_step
    model_config = GPTConfig(**model_metadata["model_config"])
    model = create_model(model_config, compute_dtype, enable_fa3, fp8_training, backward_overlap, enable_metrics, device)

    # Load Model State
    model_path = os.path.join(checkpoints_path, f"model_{checkpoint_step:06d}.pt")
    model_state = torch.load(model_path, map_location=device)
    model_state = {k.removeprefix("_orig_mod."): v for k, v in model_state.items()}  # patch if loading compiled model
    model.load_state_dict(model_state)

    model.eval()
    return model, model_metadata  # model is initialized, ready to use, not compiled
