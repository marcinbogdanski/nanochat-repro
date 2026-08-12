import torch

from nanorepro.gpt import GPTModel

class KVCache:
    """Mini data class to store KV cache related tensors."""
    def __init__(self, config, batch_size, max_seq_len, compute_dtype, device):
        head_size = config.n_embd // config.n_head
        self.k_cache = torch.zeros((config.n_layer, batch_size, max_seq_len, config.n_head, head_size), dtype=compute_dtype, device=device)
        self.v_cache = torch.zeros((config.n_layer, batch_size, max_seq_len, config.n_head, head_size), dtype=compute_dtype, device=device)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.previous_embd = None

class ResultRow:
    """Internal class to track row status in batch generation."""
    def __init__(self, tokens, stop_tokens=None):
        assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)
        assert stop_tokens is None or (isinstance(stop_tokens, (tuple, list)) and all(isinstance(t, int) for t in stop_tokens))
        self.tokens = tokens
        self.forced_tokens = []  # tool output token
        self.stop_tokens = stop_tokens or tuple()   # if None, then replace with empty tuple

    def append_token(self, token):
        assert isinstance(token, int)
        if not self.is_stopped():
            self.tokens.append(token)

    def add_forced_tokens(self, tokens):
        assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)
        assert len(self.forced_tokens) == 0
        self.forced_tokens = tokens

    def get_forced_token(self):
        if len(self.forced_tokens) > 0:
            return self.forced_tokens.pop(0)
        return None

    def get_tokens(self):
        return self.tokens

    def is_stopped(self):
        return self.tokens[-1] in self.stop_tokens


class Engine:

    def __init__(self, model, stop_tokens=None, tool_handler=None):
        self.model: GPTModel = model
        self.stop_tokens = stop_tokens
        self.tool_handler = tool_handler
        self.tool_trigger_token = None if tool_handler is None else tool_handler.tool_trigger_token

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_new_tokens=None, temperature=1.0, top_k=None, seed=42, return_logits=False):
        assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)
        max_new_tokens = self.model.config.block_size - len(tokens) if max_new_tokens is None else max_new_tokens
        assert 1 <= max_new_tokens <= self.model.config.block_size - len(tokens)  # confirm fits

        device = self.model.get_device()
        compute_dtype = self.model.compute_dtype
        rng = torch.Generator(device=device).manual_seed(seed)
        max_seq_len = len(tokens) + max_new_tokens

        # Prefill and sample
        indices = torch.tensor([tokens], dtype=torch.long, device=device)  # B,T
        kv_cache = KVCache(
            config=self.model.config,
            batch_size=1,
            max_seq_len=max_seq_len,
            compute_dtype=compute_dtype,
            device=device
        )
        logits, _, _ = self.model(indices, kv_cache=kv_cache)       # B,T,C <- B,T
        logits = logits[:, -1, :]             # B,C <- B,T,C  discard all but last
        logits = logits.expand(num_samples, -1)  # B,C <- B=1,C  broadcast to num_samples
        x_col = self.model.sample_one_token(logits, temperature=temperature, top_k=top_k, sample_rng=rng)  # B,1

        # Broadcast to num_samples
        kv_cache.k_cache = kv_cache.k_cache.expand(-1, num_samples, -1, -1, -1).clone()
        kv_cache.v_cache = kv_cache.v_cache.expand(-1, num_samples, -1, -1, -1).clone()
        kv_cache.cache_seqlens = kv_cache.cache_seqlens.expand(num_samples).clone()
        kv_cache.previous_embd = kv_cache.previous_embd.expand(num_samples, -1, -1).clone()  # B,1,E  broadcast to num_samples

        # Will return Python lists
        results = [ResultRow(indices[0].tolist(), self.stop_tokens) for _ in range(num_samples)]
        x_col_list = x_col[:, 0].tolist()
        for i in range(num_samples):
            results[i].append_token(x_col_list[i])
        if return_logits:
            logits_list = [logits]

        # Generate new tokens
        num_generated = 1
        while num_generated < max_new_tokens:
            if all(res.is_stopped() for res in results):
                break
            logits, _, _ = self.model(x_col, kv_cache=kv_cache)       # B,T,C <- B,T
            logits = logits[:, -1, :]            # B,C <- B,T,C  discard all but last
            if return_logits:
                logits_list.append(logits)
            x_col = self.model.sample_one_token(logits, temperature=temperature, top_k=top_k, sample_rng=rng)  # B,1
            x_col_list = x_col[:, 0].tolist()
            for i in range(num_samples):
                if results[i].is_stopped():
                    continue
                forced_token = results[i].get_forced_token()
                if forced_token is not None:
                    x_col[i, 0] = forced_token  # is this safe?
                    results[i].append_token(forced_token)
                else:
                    sampled_token = x_col_list[i]
                    results[i].append_token(sampled_token)
                    if sampled_token == self.tool_trigger_token:
                        tool_output_tokens = self.tool_handler.handle_tool_call(results[i].get_tokens())
                        if tool_output_tokens is not None:
                            results[i].add_forced_tokens(tool_output_tokens)
            num_generated += 1

        result_tokens = [res.get_tokens() for res in results]
        if return_logits:
            logits_list = torch.stack(logits_list, dim=1)  # B,T,C
            return result_tokens, logits_list
        return result_tokens

    @torch.inference_mode()
    def generate_naive(self, tokens, num_samples=1, max_new_tokens=None, temperature=1.0, top_k=None, seed=42, return_logits=False):
        assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)
        max_new_tokens = self.model.config.block_size - len(tokens) if max_new_tokens is None else max_new_tokens
        assert 1 <= max_new_tokens <= self.model.config.block_size - len(tokens)  # confirm fits
        assert self.stop_tokens is None  # we don't support stop tokens here, this function is for base model only and testing

        device = self.model.get_device()
        rng = torch.Generator(device=device).manual_seed(seed)

        # Will return Python lists
        indices = torch.tensor([tokens]*num_samples, dtype=torch.long, device=device)  # B,T
        if return_logits:
            logits_list = []

        # Generate new tokens
        num_generated = 0
        while num_generated < max_new_tokens:
            logits, _, _ = self.model(indices)       # B,T,C <- B,T
            logits = logits[:, -1, :]            # B,C <- B,T,C  discard all but last
            if return_logits:
                logits_list.append(logits)
            x_col = self.model.sample_one_token(logits, temperature=temperature, top_k=top_k, sample_rng=rng)  # B,1
            indices = torch.cat((indices, x_col), dim=1)  # B,T+1  append
            num_generated += 1

        if return_logits:
            logits_list = torch.stack(logits_list, dim=1)  # B,T,C
            return indices.tolist(), logits_list
        return indices.tolist()
