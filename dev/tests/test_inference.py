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
    def test_training_forward_backward_still_works(self):
        model = make_model().train()
        tokens = torch.randint(0, 64, (2, 8))
        _, loss, _ = model(tokens, targets=tokens)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())
        self.assertTrue(torch.isfinite(model.transformer.h[0].mlp.c_fc.weight.grad).all())

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


class GraphTests(unittest.TestCase):
    def test_cpu_falls_back(self):
        model = make_model()
        expected = Engine(model).generate_batch([1, 2], 5, temperature=0)
        engine = Engine(model, cuda_graphs=True)
        self.assertEqual(engine.generate_batch([1, 2], 5, temperature=0), expected)
        self.assertIsNone(engine._decoder)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_graph_reuse_growth_cancellation_and_batch_fallback(self):
        model = make_model("cuda")
        engine = Engine(model, cuda_graphs=True)
        eager = Engine(model)
        for prompt in ([1, 2, 3], [4] * 135, [7, 8]):
            expected = eager.generate_batch(prompt, 7, temperature=0, return_logits=True)
            got = engine.generate_batch(prompt, 7, temperature=0, return_logits=True)
            self.assertEqual(got[:2], expected[:2])
            torch.testing.assert_close(got[2], expected[2], atol=2e-5, rtol=2e-4)
        runner = engine._decoder
        suspended = engine.generate_stream([3, 4], 7, temperature=0)
        next(suspended)
        # An overlapping request must not overwrite the suspended graph's state.
        self.assertEqual(engine.generate_batch([8, 9], 4, temperature=0),
                         eager.generate_batch([8, 9], 4, temperature=0))
        suspended.close()
        self.assertFalse(engine._decode_lock.locked())
        self.assertEqual(engine.generate_batch([5], 4, num_samples=2, temperature=0),
                         eager.generate_batch([5], 4, num_samples=2, temperature=0))
        engine.generate_batch([1], 4, temperature=0)
        self.assertIs(engine._decoder, runner)
        self.assertIsNone(engine._get_decoder(engine.MAX_STATIC_LENGTH + 1, 1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_forced_tool_tokens_and_stop_are_outside_graph(self):
        model = make_model("cuda")
        class Tool:
            tool_trigger_token = 60
            def handle_tool_call(self, tokens):
                return [9, 10]
        def run(graphs):
            samples = iter([60, 8, 8, 1, 63])
            model.sample_one_token = lambda *a, **kw: torch.tensor([[next(samples)]], device="cuda")
            engine = Engine(model, stop_tokens=[63], tool_handler=Tool(), cuda_graphs=graphs)
            return engine.generate_batch([1, 2], 10, return_logits=True)
        expected = run(False)
        got = run(True)
        self.assertEqual(got[0], [[60, 9, 10, 1, 63]])
        self.assertEqual(got[1], ["stop"])
        torch.testing.assert_close(got[2], expected[2], atol=2e-5, rtol=2e-4)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
                     "FA2 needs an Ampere or newer CUDA GPU")
class FlashDecodeTests(unittest.TestCase):
    @torch.inference_mode()
    def test_compiler_wrapper_preserves_attention_and_cache_writes(self):
        from nanorepro.flash_attention import fa3_attn_with_kvcache, fa3_decode_with_kvcache
        q, k, v = [torch.randn(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        kc, vc = [torch.randn(1, 16, 1, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        expected_k, expected_v = kc.clone(), vc.clone()
        pos = torch.tensor([7], dtype=torch.int32, device="cuda")
        expected = fa3_attn_with_kvcache(q, expected_k, expected_v, k, v, pos, True, (3, 0))
        compiled = torch.compile(fa3_decode_with_kvcache, fullgraph=True,
                                 options={"triton.cudagraphs": False})
        got = compiled(q, kc, vc, k, v, pos, True, (3, 0))
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
        torch.testing.assert_close(kc, expected_k, atol=0, rtol=0)
        torch.testing.assert_close(vc, expected_v, atol=0, rtol=0)

    def test_flash_graph_matches_eager(self):
        model = make_model("cuda", torch.bfloat16, enable_fa=True)
        expected = Engine(model).generate_batch([1, 2, 3], 8, temperature=0, return_logits=True)
        got = Engine(model, cuda_graphs=True).generate_batch([1, 2, 3], 8, temperature=0, return_logits=True)
        self.assertEqual(got[:2], expected[:2])
        torch.testing.assert_close(got[2], expected[2], atol=0, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CompiledDecodeTests(unittest.TestCase):
    def test_compiled_modes_and_context_reuse(self):
        # Isolate Dynamo's per-code-object cache from other tests' model instances.
        torch._dynamo.reset()
        cases = [(torch.float32, False)]
        if torch.cuda.get_device_capability()[0] >= 8:
            cases.append((torch.bfloat16, True))
        for dtype, fa in cases:
            model = make_model("cuda", dtype, enable_fa=fa)
            if dtype == torch.bfloat16:
                from nanorepro.inference import prepare_inference
                prepare_inference(model)
            for graphs in (False, True):
                engine = Engine(model, compile_decode=True, cuda_graphs=graphs)
                for prompt in ([1, 2, 3], [7, 8], [3] * 130):
                    expected = Engine(model).generate_batch(prompt, 6, temperature=0, return_logits=True)
                    got = engine.generate_batch(prompt, 6, temperature=0, return_logits=True)
                    self.assertEqual(got[:2], expected[:2])
                    tol = 2e-5 if dtype == torch.float32 else .015
                    torch.testing.assert_close(got[2], expected[2], atol=tol, rtol=tol)
                self.assertEqual(engine._decoder.graph is not None, graphs)
            torch._dynamo.reset()


class BF16Tests(unittest.TestCase):
    @torch.inference_mode()
    def test_preconversion_preserves_eager_logits_and_other_parameters(self):
        from nanorepro.inference import prepare_inference
        model = make_model(dtype=torch.bfloat16)
        tokens = torch.tensor([[1, 2, 3, 4]])
        expected = model(tokens)[0]
        linear_weights = {id(p): p.clone() for m in model.modules()
                          if isinstance(m, torch.nn.Linear) for p in m.parameters()}
        other_weights = {name: p.clone() for name, p in model.named_parameters()
                         if id(p) not in linear_weights}
        prepare_inference(model)
        for name, p in model.named_parameters():
            if id(p) in linear_weights:
                self.assertEqual(p.dtype, torch.bfloat16)
                torch.testing.assert_close(p, linear_weights[id(p)].bfloat16(), atol=0, rtol=0)
            else:
                torch.testing.assert_close(p, other_weights[name], atol=0, rtol=0)
            self.assertFalse(p.requires_grad)
        torch.testing.assert_close(model(tokens)[0], expected, atol=0, rtol=0)
        prepare_inference(model)  # repeated preparation must not alter weights
        torch.testing.assert_close(model(tokens)[0], expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "reload"):
            model.train()


class FP16Tests(unittest.TestCase):
    @torch.inference_mode()
    def test_scaling_prevents_squared_relu_overflow_and_is_idempotent(self):
        from nanorepro.inference import prepare_fp16_inference
        model = make_model(dtype=torch.float16)
        mlp = model.transformer.h[0].mlp
        mlp.c_fc.weight.zero_()
        mlp.c_proj.weight.zero_()
        mlp.c_fc.weight[0, 0] = 271
        mlp.c_proj.weight[0, 0] = .01
        x = torch.zeros((1, 1, 128), dtype=torch.float16)
        x[..., 0] = 1
        expected = mlp.c_proj(torch.relu(mlp.c_fc(x).float()).square())
        self.assertFalse(torch.isfinite(mlp(x)).all())
        prepare_fp16_inference(model)
        got = mlp(x)
        self.assertTrue(torch.isfinite(got).all())
        torch.testing.assert_close(got.float(), expected, atol=1, rtol=.002)
        first_weight = mlp.c_fc.weight.clone()
        prepare_fp16_inference(model)
        torch.testing.assert_close(mlp.c_fc.weight, first_weight, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "reload"):
            model.train()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_compilation_and_graphs(self):
        from nanorepro.inference import prepare_inference
        torch._dynamo.reset()
        model = prepare_inference(make_model("cuda", torch.float16))
        expected = Engine(model).generate_batch([1, 2, 3], 6, temperature=0, return_logits=True)
        for compile_decode, graphs in ((False, True), (True, False), (True, True)):
            got = Engine(model, cuda_graphs=graphs, compile_decode=compile_decode).generate_batch(
                [1, 2, 3], 6, temperature=0, return_logits=True)
            self.assertEqual(got[:2], expected[:2])
            torch.testing.assert_close(got[2], expected[2], atol=.002, rtol=.01)
        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
