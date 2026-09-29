"""Small-model inference checks; no checkpoint required.

    uv run python -m unittest dev.tests.test_inference
"""
import unittest
import torch
from nanorepro.checkpoint import create_model
from nanorepro.engine import Engine, KVCache
from nanorepro.gpt import GPTConfig


def make_model(device="cpu", dtype=torch.float32, enable_fa=False):
    torch.manual_seed(42)
    config = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2,
                       n_head=1, n_embd=128, window_pattern="SL")
    model = create_model(config, dtype, enable_fa, False, False, device).eval()
    # Initial projection weights/smear are zero; use nonzero values to exercise state.
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(0, .02)
            block.mlp.c_proj.weight.normal_(0, .02)
        model.smear_lambda.fill_(.3)
    return model


class CacheTests(unittest.TestCase):
    @torch.inference_mode()
    def test_cached_matches_full_forward_and_keeps_smear_storage(self):
        model = make_model()
        tokens = torch.randint(0, 64, (1, 24))
        cache = KVCache(model.config, 1, 24, torch.float32, "cpu")
        model(tokens[:, :3], kv_cache=cache)
        smear_ptr = cache.previous_embd.data_ptr()
        for end in range(4, 25):
            got = model(tokens[:, end-1:end], kv_cache=cache)[0][:, -1]
            expected = model(tokens[:, :end])[0][:, -1]
            torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-5)
            self.assertEqual(cache.previous_embd.data_ptr(), smear_ptr)
        self.assertEqual(cache.cache_seqlens.item(), 24)
        self.assertNotEqual(cache.k_cache[0].untyped_storage().data_ptr(),
                            cache.k_cache[1].untyped_storage().data_ptr())

    @torch.inference_mode()
    def test_static_decode_matches_eager_across_window_boundary(self):
        model = make_model()
        model.window_sizes = [(3, 0), (-1, 0)]
        tokens = torch.randint(0, 64, (1, 24))
        eager = KVCache(model.config, 1, 32, torch.float32, "cpu")
        static = KVCache(model.config, 1, 32, torch.float32, "cpu", static_decode=True)
        for cache in (eager, static):
            model(tokens[:, :2], kv_cache=cache)
        for end in range(3, 25):
            idx = tokens[:, end-1:end]
            expected = model(idx, kv_cache=eager)[0]
            got = model(idx, kv_cache=static)[0]
            torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-5)

    @torch.inference_mode()
    def test_engine_broadcast_matches_independent_greedy_samples(self):
        engine = Engine(make_model())
        one, reasons = engine.generate_batch([1, 2, 3], 12, temperature=0)
        many, many_reasons = engine.generate_batch([1, 2, 3], 12, num_samples=3, temperature=0)
        self.assertEqual(many, one * 3)
        self.assertEqual(many_reasons, reasons * 3)


if __name__ == "__main__":
    unittest.main()
