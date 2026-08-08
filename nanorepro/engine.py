import torch

class KVCache:
    """Mini data class to store KV cache related tensors."""
    def __init__(self, config, batch_size, compute_dtype, device):
        head_size = config.n_embd // config.n_head
        self.k_cache = torch.zeros((config.n_layer, batch_size, config.block_size, config.n_head, head_size), dtype=compute_dtype, device=device)
        self.v_cache = torch.zeros((config.n_layer, batch_size, config.block_size, config.n_head, head_size), dtype=compute_dtype, device=device)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.previous_embd = None
