import os
import gc
import json
import torch
from nanorepro.gpt import GPTModel, GPTConfig

def optim_state_to_nanochat_format(adamw_state_dict, muon_state_dict):
    """Convert our AdamW/Muon state dicts into Nanochat combined optimizer format."""
    num_adamw_params = sum(len(g['params']) for g in adamw_state_dict['param_groups'])

    # Combine our two optimizer param groups into the Nanochat format
    nanochat_format_param_groups = []
    for g in adamw_state_dict['param_groups']:
        assert set(g) == {'lr', 'betas', 'weight_decay', 'is_small', 'eps', 'initial_lr', 'params'}
        nanochat_format_param_groups.append({
            'kind': 'adamw',
            'lr': g['lr'],
            'betas': g['betas'],
            'eps': g['eps'],
            'weight_decay': g['weight_decay'],
            'initial_lr': g['initial_lr'],
            'params': list(g['params'])    # list of int, copy just in case
        })
    for g in muon_state_dict['param_groups']:
        assert set(g) == {'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay', 'initial_lr', 'params'}
        nanochat_format_param_groups.append({
            'kind': 'muon',
            'lr': g['lr'],
            'momentum': g['momentum'],
            'ns_steps': g['ns_steps'],
            'beta2': g['beta2'],
            'weight_decay': g['weight_decay'],
            'initial_lr': g['initial_lr'],
            'params': [p + num_adamw_params for p in g['params']]  # list of int, offset to match Nanochat combined optimizer
        })

    # Dict:
    # - AdamW: param_id -> {'step': int, 'exp_avg': Tensor, 'exp_avg_sq': Tensor}
    # - Muon: param_id -> {'momentum_buffer': Tensor, 'second_momentum_buffer': Tensor}
    nanochat_format_state = {}
    for i, st in adamw_state_dict['state'].items():
        nanochat_format_state[i] = {
            'step': st['step'],
            'exp_avg': st['exp_avg'],
            'exp_avg_sq': st['exp_avg_sq']
        }
    for i, st in muon_state_dict['state'].items():
        # in Nanochat Muon buffers are created with zeros_like() on views and inherit 
        nanochat_format_state[i + num_adamw_params] = {  # offset to match Nanochat combined optimizer
            'momentum_buffer': st['momentum_buffer'].clone(memory_format=torch.contiguous_format),
            'second_momentum_buffer': st['momentum_buffer2'].clone(memory_format=torch.contiguous_format)
        }

    # This is what Nanochat combined optimizer saves
    nanochat_format_state_dict = {
        'state': nanochat_format_state,
        'param_groups': nanochat_format_param_groups,
    }
    return nanochat_format_state_dict

def optim_state_from_nanochat_format(nanochat_format_state_dict, adamw_sd_template, muon_sd_template):
    """Split Nanochat combined optimizer state dict back into separate AdamW and Muon state dicts."""
    nanochat_adamw_groups = [g for g in nanochat_format_state_dict['param_groups'] if g['kind'] == 'adamw']
    nanochat_muon_groups = [g for g in nanochat_format_state_dict['param_groups'] if g['kind'] == 'muon']
    assert len(nanochat_adamw_groups) == len(adamw_sd_template['param_groups'])
    assert len(nanochat_muon_groups) == len(muon_sd_template['param_groups'])
    num_adamw_params = sum(len(g['params']) for g in nanochat_adamw_groups)

    # Reconstruct AdamW state dict
    adamw_state = {}
    for param_id, st in nanochat_format_state_dict['state'].items():
        if param_id < num_adamw_params:
            adamw_state[param_id] = st  # no offset, no renames
    adamw_param_groups = []
    for nanochat_g, template_g in zip(nanochat_adamw_groups, adamw_sd_template['param_groups']):
        adamw_param_groups.append({
            'lr': nanochat_g['lr'],
            'betas': nanochat_g['betas'],
            'weight_decay': nanochat_g['weight_decay'],
            'is_small': template_g['is_small'],
            'eps': nanochat_g['eps'],
            'initial_lr': nanochat_g['initial_lr'],
            'params': [p_idx for p_idx in nanochat_g['params']]
        })
    adamw_state_dict = {
        'state': adamw_state,
        'param_groups': adamw_param_groups
    }

    muon_state = {}
    for i, st in nanochat_format_state_dict['state'].items():
        if i >= num_adamw_params:
            muon_state[i - num_adamw_params] = {  # offset to match original Muon state dict
                'momentum_buffer': st['momentum_buffer'],
                'momentum_buffer2': st['second_momentum_buffer']
            }
    muon_param_groups = []
    for nanochat_g, template_g in zip(nanochat_muon_groups, muon_sd_template['param_groups']):
        muon_param_groups.append({
            'lr': nanochat_g['lr'],
            'momentum': nanochat_g['momentum'],
            'ns_steps': nanochat_g['ns_steps'],
            'beta2': nanochat_g['beta2'],
            'weight_decay': nanochat_g['weight_decay'],
            'initial_lr': nanochat_g['initial_lr'],
            'params': [p_idx - num_adamw_params for p_idx in nanochat_g['params']]
        })
    muon_state_dict = {
        'state': muon_state,
        'param_groups': muon_param_groups
    }
    return adamw_state_dict, muon_state_dict


def load_optimizer_state(optim_path, device, adamw_optimizer, muon_optimizer):
    optim_state = torch.load(optim_path, map_location=device)
    if 'adamw' in optim_state and 'muon' in optim_state:
        return optim_state['adamw'], optim_state['muon']   # legacy
    return optim_state_from_nanochat_format(      # nanochat format
        nanochat_format_state_dict=optim_state,
        adamw_sd_template=adamw_optimizer.state_dict(),
        muon_sd_template=muon_optimizer.state_dict()
    )
    

def save_checkpoint(checkpoints_path, model, optimizers, dataloader, loop_vars, user_config, training_hyperparameters):
    os.makedirs(checkpoints_path, exist_ok=True)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
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

        # Nanochat saves per-param tensors directly from the GPU. This ceremony maintains MD5 checksum compatibility.
        # In this repo, model params may be views into static buffers created by optimizer,
        # so we need to create independent copies of each param tensor. To save GPU VRAM we copy them to CPU,
        # but now they are CPU-tagged, which breaks checksum. So we hack again and override PyTorch's location tagging mechanism.
        # This way we have VRAM-friendly CPU saving, params are saved as independent tensors, and .pt file is GPU-tagged. Happy days.
        model_state_meta = model_state._metadata
        model_state = model_state.__class__((name, tensor.detach().to("cpu", copy=True)) for name, tensor in model_state.items())
        model_state._metadata = model_state_meta
        try:
            orig_location_tag = torch.serialization.location_tag  # function that takes storage object and returns str tag
            device = model.get_device()
            torch.serialization.location_tag = lambda storage: str(device)   # ignore param and just tag with "cuda:0" etc
            torch.save(model_state, model_path)
        finally:
            torch.serialization.location_tag = orig_location_tag  # always restore
        model_md5sum = os.popen(f"md5sum {model_path}").read().split()[0]

        gc.collect()  # This function creates bunch of tensors on CPU, since GC is disabled in base_train.py, we cleanup manually
    
    # Optimizer state
    adamw_opt, muon_opt = optimizers
    nanochat_format_combined_sd = optim_state_to_nanochat_format(adamw_opt.state_dict(), muon_opt.state_dict())
    optim_path = os.path.join(checkpoints_path, f"optim_{step:06d}_rank{rank:d}.pt")
    torch.save(nanochat_format_combined_sd, optim_path)
    optim_md5sum = os.popen(f"md5sum {optim_path}").read().split()[0]
    
    # Dataloader state
    dataloader_path = os.path.join(checkpoints_path, f"dataloader_{step:06d}_rank{rank:d}.pt")
    torch.save(dataloader.state_dict(), dataloader_path)

    # Print MD5 sums
    if rank == 0:
        print(f"Saved model_{step:06d}.pt with MD5 sum: {model_md5sum}")
    for r in range(world_size):
        if r == rank:
            md5sum_optim = os.popen(f"md5sum {optim_path}").read().split()[0]
            print(f"Saved optim optim_{step:06d}_rank{r:d}.pt MD5 sum: {md5sum_optim}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    return model_md5sum, optim_md5sum


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
    adamw_sd, muon_sd = load_optimizer_state(optim_path, device, adamw_opt, muon_opt)
    adamw_opt.load_state_dict(adamw_sd)
    muon_opt.load_state_dict(muon_sd)

    # Dataloader state
    dataloader_state = torch.load(dataloader_path, map_location="cpu")
    dataloader.load_state_dict(dataloader_state)

    return loop_vars

def create_model(model_config, compute_dtype, enable_fa, fp8_training, enable_metrics, device):
    """Create a GPT model on device and call init_weights(). Returns ready, non-compiled model."""
    with torch.device("meta"):
        model = GPTModel(
            model_config,
            compute_dtype=compute_dtype,
            enable_fa=enable_fa,
            fp8_training=fp8_training,
            enable_metrics=enable_metrics,
        )
    model.to_empty(device=device)
    model.init_weights()  # RoPE buffers, random weight init
    return model

def load_model(checkpoints_path, compute_dtype, enable_fa, fp8_training, enable_metrics, device, step=None):
    """Load a GPT model from checkpoint to device. Returns ready, non-compiled model in eval mode and loaded metadata."""
    checkpoint_step = get_latest_checkpoint_step(checkpoints_path) if step is None else step

    # Load Model Metadata
    metadata_path = os.path.join(checkpoints_path, f"meta_{checkpoint_step:06d}.json")
    with open(metadata_path, "r") as f:
        model_metadata = json.load(f)
    assert model_metadata["step"] == checkpoint_step
    model_config = GPTConfig(**model_metadata["model_config"])
    model = create_model(model_config, compute_dtype, enable_fa, fp8_training, enable_metrics, device)

    # Load Model State
    model_path = os.path.join(checkpoints_path, f"model_{checkpoint_step:06d}.pt")
    model_state = torch.load(model_path, map_location=device)
    model_state = {k.removeprefix("_orig_mod."): v for k, v in model_state.items()}  # patch if loading compiled model
    model.load_state_dict(model_state)

    model.eval()
    return model, model_metadata  # model is initialized, ready to use, not compiled
