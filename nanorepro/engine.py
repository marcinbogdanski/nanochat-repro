import threading
import torch

from nanorepro.gpt import GPTModel

class KVCache:
    """Mini data class to store KV cache related tensors."""
    def __init__(self, config, batch_size, max_seq_len, compute_dtype, device, static_decode=False):
        head_size = config.n_embd // config.n_head
        # Independent allocations avoid aliasing between layers when compiling updates.
        shape = (batch_size, max_seq_len, config.n_head, head_size)
        self.k_cache = [torch.zeros(shape, dtype=compute_dtype, device=device) for _ in range(config.n_layer)]
        self.v_cache = [torch.zeros(shape, dtype=compute_dtype, device=device) for _ in range(config.n_layer)]
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.previous_embd = None
        self.static_decode = static_decode
        self.positions = torch.arange(max_seq_len, device=device) if static_decode else None

    def expand_batch(self, batch_size):
        """Broadcast a single prefill to multiple samples, without copying for batch 1."""
        if batch_size == self.cache_seqlens.numel():
            return
        assert self.cache_seqlens.numel() == 1
        self.k_cache = [k.expand(batch_size, -1, -1, -1).clone() for k in self.k_cache]
        self.v_cache = [v.expand(batch_size, -1, -1, -1).clone() for v in self.v_cache]
        self.cache_seqlens = self.cache_seqlens.expand(batch_size).clone()
        self.previous_embd = self.previous_embd.expand(batch_size, -1, -1).clone()

class RowState:
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

    def has_forced_tokens(self):
        return len(self.forced_tokens) > 0

    def get_forced_token(self):
        if len(self.forced_tokens) > 0:
            return self.forced_tokens.pop(0)
        else:
            raise ValueError("No forced tokens available to retrieve.")

    def get_tokens(self):
        return self.tokens

    def is_stopped(self):
        return self.tokens[-1] in self.stop_tokens


class Engine:

    def __init__(self, model, stop_tokens=None, tool_handler=None, *, cuda_graphs=False, compile_decode=False):
        self.model: GPTModel = model
        self.stop_tokens = stop_tokens
        self.tool_handler = tool_handler
        self.tool_trigger_token = None if tool_handler is None else tool_handler.tool_trigger_token
        self.cuda_graphs = cuda_graphs
        self.compile_decode = compile_decode
        self._decoder = None
        self._decode_lock = threading.Lock()

    # Bound retained graph memory to one runner, growing in context buckets.
    # Longer requests and batches still use the ordinary eager implementation.
    MAX_STATIC_LENGTH = 4096

    def _get_decoder(self, length, num_samples):
        if (num_samples != 1 or length > self.MAX_STATIC_LENGTH
                or self.model.get_device().type != "cuda" or self.model.training
                or self.model.config.moe_enable or self.model.enable_metrics):
            return None
        if self._decoder is None or self._decoder.capacity < length:
            from nanorepro.decode import DecodeRunner
            capacity = min(max(128, 1 << (length - 1).bit_length()), self.model.max_position_embeddings())
            self._decoder = None  # release the previous graph before allocating a larger one
            self._decoder = DecodeRunner(self.model, capacity, cuda_graphs=self.cuda_graphs,
                                         compile_decode=self.compile_decode)
        return self._decoder

    @torch.inference_mode()
    def generate_stream(self, tokens, max_new_tokens, num_samples=1, temperature=1.0, top_k=None, seed=42, return_logits=False):
        if not tokens or max_new_tokens < 1 or num_samples < 1:
            raise ValueError("Need a nonempty prompt and positive token/sample counts")
        if len(tokens) + max_new_tokens > self.model.max_position_embeddings():
            raise ValueError("Prompt and generation exceed the model's position limit")
        # A suspended generator owns the reusable buffers until it finishes or
        # is closed. Concurrent users of the same Engine take the eager path.
        acquired = (self.cuda_graphs or self.compile_decode) and self._decode_lock.acquire(blocking=False)
        try:
            decoder = None
            if acquired and max_new_tokens > 1:
                decoder = self._get_decoder(len(tokens) + max_new_tokens, num_samples)
            yield from self._generate_stream(tokens, max_new_tokens, num_samples,
                                             temperature, top_k, seed, return_logits, decoder)
        finally:
            if acquired:
                self._decode_lock.release()

    @torch.inference_mode()
    def _generate_stream(self, tokens, max_new_tokens, num_samples=1, temperature=1.0, top_k=None, seed=42, return_logits=False, decoder=None):
        assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)
        assert isinstance(max_new_tokens, int) and max_new_tokens > 0

        device = self.model.get_device()
        compute_dtype = self.model.compute_dtype
        rng = torch.Generator(device=device).manual_seed(seed)
        max_seq_len = len(tokens) + max_new_tokens
        rows = [RowState(tokens.copy(), self.stop_tokens) for _ in range(num_samples)]
        kv_cache = KVCache(config=self.model.config, batch_size=1, max_seq_len=max_seq_len, compute_dtype=compute_dtype, device=device)

        # Generate new tokens
        num_generated = 0
        while True:
            if num_generated == 0:
                # Prefill the model with the initial tokens
                indices = torch.tensor([tokens], dtype=torch.long, device=device)  # B,T
                logits, _, _ = self.model(indices, kv_cache=kv_cache)       # B,T,C <- B,T
                logits = logits[:, -1, :]             # B,C <- B,T,C  discard all but last
                logits = logits.expand(num_samples, -1)  # B,C <- B=1,C  broadcast to num_samples

                # Broadcast to num_samples
                kv_cache.expand_batch(num_samples)
                if decoder is not None:
                    decoder.load_prefill(kv_cache, len(tokens))
                    kv_cache = None  # release the temporary prefill allocation

            elif decoder is not None:
                logits = decoder.step(token_column_t)
            else:
                # Generate next token
                logits, _, _ = self.model(token_column_t, kv_cache=kv_cache)       # B,T,C <- B,T
                logits = logits[:, -1, :]            # B,C <- B,T,C  discard all but last

            # Append new token to the sequence
            token_column_t = self.model.sample_one_token(logits, temperature=temperature, top_k=top_k, sample_rng=rng)  # B,1
            token_column = token_column_t[:, 0].tolist()  # avoid .item() on each loop iteration

            # Update results and handle forced tokens
            for i in range(num_samples):
                if rows[i].is_stopped():
                    # rows[i].append_token(..)          # append: nothing to append, row is done
                    # token_column_t[i, 0] = bos_token  # feed: could pad, but makes no difference what we feed to the model, this row is done
                    token_column[i] = None              # return: what we return
                elif rows[i].has_forced_tokens():
                    forced_token = rows[i].get_forced_token()
                    rows[i].append_token(forced_token)   # append: append row
                    token_column_t[i, 0] = forced_token  # feed: feed to the model
                    token_column[i] = forced_token       # return: what we return
                else:
                    rows[i].append_token(token_column[i])           # append: append row
                    # token_column_t[i, 0] = token_column_t[i, 0]   # feed: current token is valid to feed back to the model
                    # token_column[i] = ...                         # return: already assigned
                    if token_column[i] == self.tool_trigger_token:
                        tool_output_tokens = self.tool_handler.handle_tool_call(rows[i].get_tokens())
                        if tool_output_tokens is not None:
                            rows[i].add_forced_tokens(tool_output_tokens)

            # Are we done?
            num_generated += 1
            finish_reasons = [None] * num_samples
            for i, row in enumerate(rows):
                if row.is_stopped():
                    finish_reasons[i] = "stop"
                elif num_generated >= max_new_tokens:
                    finish_reasons[i] = "length"

            # Yield Result
            if return_logits:
                yield token_column, finish_reasons, logits.clone() if decoder is not None else logits
            else:
                yield token_column, finish_reasons

            # We are done
            if all(fr is not None for fr in finish_reasons):
                break


    def generate_batch(self, tokens, max_new_tokens, num_samples=1, temperature=1.0, top_k=None, seed=42, return_logits=False):
        token_rows = [[] for _ in range(num_samples)]
        logits_list = []
        for result in self.generate_stream(tokens, max_new_tokens, num_samples, temperature, top_k, seed, return_logits):
            if return_logits:
                token_column, finish_reasons, logits_column = result
                logits_list.append(logits_column)
            else:
                token_column, finish_reasons = result
            for i in range(num_samples):
                if token_column[i] is not None:
                    token_rows[i].append(token_column[i])

        # Package and Return
        if return_logits:
            logits_list = torch.stack(logits_list, dim=1)  # B,T,C
            return token_rows, finish_reasons, logits_list
        return token_rows, finish_reasons


    @torch.inference_mode()
    def generate_naive(self, tokens, max_new_tokens, num_samples=1, temperature=1.0, top_k=None, seed=42, return_logits=False):
        assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)
        assert isinstance(max_new_tokens, int) and max_new_tokens > 0
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

        finish_reasons = ["length"] * num_samples  # this is for API compatibility only, since we don't support stop_tokens this is always 'length'
        generated_tokens = indices[:, len(tokens):].tolist()
        if return_logits:
            logits_list = torch.stack(logits_list, dim=1)  # B,T,C
            return generated_tokens, finish_reasons, logits_list
        return generated_tokens, finish_reasons
