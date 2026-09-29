"""Compare eager, graph, compiled, and compiled+graph decoding on one GPU.

    CUDA_VISIBLE_DEVICES=0 uv run python -m dev.benchmark_decode --run=d24
    CUDA_VISIBLE_DEVICES=0 uv run python -m dev.benchmark_decode --run=d24 --compute-dtype=fp16 --no-fa

Compilation/capture costs appear in setup_s, not steady decode tokens/s.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import argparse
import gc
import json
import pickle
import statistics
import time
import torch
from nanorepro.checkpoint import load_model
from nanorepro.common import get_base_path
from nanorepro.engine import Engine, KVCache
from nanorepro.inference import prepare_fp16_inference


@torch.inference_mode()
def validate(engine, model, prompt, reference_tokens, reference_logits):
    """Feed identical tokens to both paths; sampling divergence cannot skew errors."""
    decoder = engine._decoder
    if decoder is None:
        return {"optimized": False}
    source = KVCache(model.config, 1, len(prompt) + reference_logits.size(1), model.compute_dtype, model.get_device())
    model(torch.tensor([prompt], device=model.get_device()), kv_cache=source)
    decoder.load_prefill(source, len(prompt))
    max_errors, mean_errors, same_top1 = [], [], []
    for i in range(1, reference_logits.size(1)):
        token = torch.tensor([[reference_tokens[0][i-1]]], device=model.get_device())
        got = decoder.step(token).cpu()
        expected = reference_logits[:, i]
        if not torch.isfinite(got).all():
            raise RuntimeError("Nonfinite decode logits")
        difference = (got - expected).abs()
        max_errors.append(difference.max().item())
        mean_errors.append(difference.mean().item())
        same_top1.append(bool(got.argmax() == expected.argmax()))
    return {"optimized": True, "checked_steps": len(max_errors),
            "max_logit_difference": max(max_errors), "mean_logit_difference": statistics.mean(mean_errors),
            "top1_agreement": statistics.mean(same_top1)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="d24")
    parser.add_argument("--compute-dtype", choices=["bf16", "fp32", "fp16"], default="bf16")
    parser.add_argument("--no-fa", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This benchmark requires a CUDA GPU")
    if args.prompt_tokens < 5 or args.decode_tokens < 1 or args.repeats < 1:
        parser.error("Need prompt-tokens >= 5 and positive decode-tokens/repeats")
    if args.prompt_tokens + args.decode_tokens + 1 > Engine.MAX_STATIC_LENGTH:
        parser.error("Prompt and output must fit the optimized context limit")
    torch.set_num_threads(4)
    base = get_base_path()
    with open(os.path.join(base, "tokenizer/tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    text = tokenizer.encode("Explain why the sky is blue in simple terms. ")
    content = (text * (args.prompt_tokens // len(text) + 1))[:args.prompt_tokens - 4]
    prompt = [tokenizer.encode_single_token("<|bos|>"), tokenizer.encode_single_token("<|user_start|>")]
    prompt += content + [tokenizer.encode_single_token("<|user_end|>"), tokenizer.encode_single_token("<|assistant_start|>")]
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.compute_dtype]
    model, _ = load_model(os.path.join(base, "runs_sft", args.run), dtype, not args.no_fa,
                          False, False, "cuda")
    if dtype == torch.float16:
        prepare_fp16_inference(model)
    if len(prompt) + args.decode_tokens + 1 > model.max_position_embeddings():
        parser.error("Prompt and output exceed this model's position limit")
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                      "dtype": args.compute_dtype, "fa": not args.no_fa,
                      "prompt_tokens": len(prompt), "decode_tokens": args.decode_tokens}), flush=True)
    reference_tokens, _, reference_logits = Engine(model).generate_batch(
        prompt, min(33, args.decode_tokens + 1), temperature=.6, top_k=50, seed=42, return_logits=True)
    reference_logits = reference_logits.cpu()
    if not torch.isfinite(reference_logits).all():
        raise RuntimeError("Nonfinite eager logits")

    modes = [("eager", False, False), ("graphs", True, False),
             ("compile", False, True), ("compile+graphs", True, True)]
    for name, graphs, compile_decode in modes:
        engine = Engine(model, cuda_graphs=graphs, compile_decode=compile_decode)
        torch.cuda.synchronize()
        start = time.perf_counter()
        engine.generate_batch(prompt, args.decode_tokens + 1, temperature=.6, top_k=50, seed=42)
        torch.cuda.synchronize()
        setup_s = time.perf_counter() - start  # includes one full warmup request
        checks = validate(engine, model, prompt, reference_tokens, reference_logits)
        rates, first_tokens = [], []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            stream = engine.generate_stream(prompt, args.decode_tokens + 1, temperature=.6, top_k=50, seed=42)
            next(stream)
            torch.cuda.synchronize()
            first = time.perf_counter()
            for _ in stream:
                pass
            torch.cuda.synchronize()
            rates.append(args.decode_tokens / (time.perf_counter() - first))
            first_tokens.append(1000 * (first - start))
        print(json.dumps({"mode": name, "setup_s": setup_s, "decode_tokens_per_s": statistics.median(rates),
                          "first_token_ms": statistics.median(first_tokens), "trials": rates,
                          **checks}), flush=True)
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
