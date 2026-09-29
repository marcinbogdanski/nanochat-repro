"""Reusable, single-sequence CUDA decode state. Prefill and sampling stay eager."""
import torch
from nanorepro.engine import KVCache


class DecodeRunner:
    @torch.inference_mode()
    def __init__(self, model, capacity):
        self.model = model
        self.capacity = capacity
        device = model.get_device()
        self.cache = KVCache(model.config, 1, capacity, model.compute_dtype, device, static_decode=True)
        self.cache.previous_embd = torch.zeros((1, 1, model.config.n_embd),
                                               dtype=model.compute_dtype, device=device)
        self.input = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.position = 0
        self.graph = None

        # Warm libraries/allocators on a side stream before capture. These are
        # dummy decode steps, so discard all their cache/state changes afterward.
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._forward()
            torch.cuda.current_stream(device).wait_stream(stream)
            self._reset()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = self._forward()
            self._reset()

    def _forward(self):
        logits, _, _ = self.model(self.input, kv_cache=self.cache)
        return logits[:, -1, :]

    def _reset(self):
        for tensor in self.cache.k_cache + self.cache.v_cache:
            tensor.zero_()
        self.cache.cache_seqlens.zero_()
        self.cache.previous_embd.zero_()
        self.position = 0

    def load_prefill(self, source, length):
        """Copy a fresh prompt into captured storage without replacing any tensor."""
        if not 0 < length < self.capacity:
            raise ValueError("Prefill must leave room for decoding")
        self._reset()
        for dest, src in zip(self.cache.k_cache, source.k_cache):
            dest[:, :length].copy_(src[:, :length])
        for dest, src in zip(self.cache.v_cache, source.v_cache):
            dest[:, :length].copy_(src[:, :length])
        self.cache.cache_seqlens.copy_(source.cache_seqlens)
        self.cache.previous_embd.copy_(source.previous_embd)
        self.position = length

    def step(self, token):
        if self.position >= self.capacity:
            raise ValueError("Decode cache capacity exceeded")
        self.input.copy_(token)
        self.graph.replay()
        self.position += 1
        # A view into graph-owned storage: callers retaining logits must clone.
        return self.output
