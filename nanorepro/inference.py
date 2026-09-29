"""Explicit inference-only preparation; checkpoint loading/training stay unchanged."""
import torch


@torch.no_grad()
def prepare_fp16_inference(model):
    """Rescale the dense squared-ReLU MLP before converting linear weights.

    relu(h / 16)**2 * 256 == relu(h)**2 in real arithmetic. Moving the
    reciprocal scales into the two bias-free weights avoids extra decode ops
    and keeps the intermediate square in FP16 range for the tested checkpoints.
    This is not a guarantee against overflow for every checkpoint/input.
    """
    if getattr(model, "_fp16_inference_prepared", False):
        return model
    if model.training or model.compute_dtype != torch.float16:
        raise ValueError("FP16 preparation requires an eval model loaded with compute_dtype=float16")
    if model.config.moe_enable:
        raise ValueError("FP16 scaling is only supported for the dense MLP")
    if model._compiled_layer_regions is not None:
        raise ValueError("Prepare FP16 weights before compiling the model")

    for block in model.transformer.h:
        block.mlp.c_fc.weight.div_(16)
        block.mlp.c_proj.weight.mul_(256)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(dtype=torch.float16)
    model.requires_grad_(False)
    model._fp16_inference_prepared = True
    return model
