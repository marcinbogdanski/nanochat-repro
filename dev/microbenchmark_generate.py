import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for older PyTorch
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels progress bars
import json
import pickle
import torch
import torch.nn.functional as F
import time
from nanorepro.gpt import GPTConfig, GPTModel
from nanorepro.checkpoint import get_latest_checkpoint_step
from nanorepro.common import get_base_path
from nanorepro.engine import KVCache, Engine
BASE_DIR = get_base_path()

def check_diff(title, t1, t2):
    print(f"--- {title} ---")
    abs_diff = (t1 - t2).abs()
    rel_diff = abs_diff / t1.abs().clamp_min(1e-6)
    cos_sim = F.cosine_similarity(t1.flatten(), t2.flatten(), dim=0)
    norm_ratio = t2.norm() / t1.norm()
    relative_l2 = abs_diff.norm() / t1.norm().clamp_min(1e-6)
    print( f"{abs_diff.max()=}" )
    print( f"{abs_diff.mean()=}" )
    # print( f"{rel_diff.max()=}")
    print( f"{rel_diff.mean()=}")
    print( f"{cos_sim=}")
    print( f"{norm_ratio=}")
    print( f"{relative_l2=}")

@torch.inference_mode()
def main():
    # Compute setup and helpers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    enable_fa3 = True if torch.cuda.is_available() else False

    # Overrides
    compute_dtype = torch.float32
    enable_fa3 = False
    
    # Tokenizer
    tok_base_path = os.path.join(BASE_DIR, "tokenizer")
    tokenizer_path = os.path.join(tok_base_path, "tokenizer.pkl")
    tokenizer = pickle.load(open(tokenizer_path, "rb"))
    token_bytes_path = os.path.join(tok_base_path, "token_bytes.pt")
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)

    # Precision - don't use TF32 as it reduces precision for comparisons
    #if device.startswith("cuda"):
    #    torch.set_float32_matmul_precision("high")  # uses tf32 instead of fp32 for matmuls

    # Model Setup
    checkpoints_path = os.path.join(BASE_DIR, "runs/scaling3/scaling3_1e18_d12")
    latest_checkpoint_step = get_latest_checkpoint_step(checkpoints_path)
    latest_meta_path = os.path.join(checkpoints_path, f"meta_{latest_checkpoint_step:06d}.json")  # last saved file
    with open(latest_meta_path, "r") as f:
        pretrain_metadata = json.load(f)
    model_config = GPTConfig(**pretrain_metadata["model_config"])
    with torch.device("meta"):
        model = GPTModel(
            model_config,
            compute_dtype=compute_dtype,
            enable_fa3=enable_fa3,
            fp8_training=True,
            enable_metrics=False,
        )
    model.to_empty(device=device)
    model.init_weights()  # RoPE buffers, rest of weights will be loaded from checkpoint
    print("Model configuration:")
    for k, v in model.config.to_dict().items():
        print(f"  {k:>16}: {v}")

    # Load Model State
    model_path = os.path.join(checkpoints_path, f"model_{latest_checkpoint_step:06d}.pt")
    model_state = torch.load(model_path, map_location=device)
    model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    model.load_state_dict(model_state)
    model.eval()

    # Generate Test Samples
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    max_new_tokens = 16
    block_size = model.config.block_size
    bos_token = tokenizer.encode_single_token('<|bos|>')

    print()
    print("Test 1: Compare naive generation vs KV cache generation w/o Engine")
    print()
    for prompt in prompts:
        tokens =  [bos_token] + tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)  # B,T
        max_seq_len = len(tokens) + max_new_tokens
        kv_cache = KVCache(config=model.config, batch_size=1, max_seq_len=max_seq_len, compute_dtype=compute_dtype, device=device)
        for i in range(max_new_tokens):
            # Forward without KV cache
            idx_tail = idx[:, -block_size:]      # B,T  sliding window
            logits, _, _ = model(idx_tail)       # B,T,C <- B,T
            logits = logits[:, -1, :]            # B,C <- B,T,C  discard all but last
            # Forward with KV cache
            if i == 0:
                idx_kv = idx_tail                # B,T   prefill cache with full context
            else:
                idx_kv = idx[:, -1:]             # B,1  last token only
            logits_kv, _, _ = model(idx_kv, kv_cache=kv_cache)       # B,T,C <- B,T
            logits_kv = logits_kv[:, -1, :]            # B,C <- B,T,C  discard all but last
            # print max difference between logits and logits_kv
            assert torch.allclose(logits, logits_kv, atol=1e-4, rtol=1e-4)
            #if i % 4 == 0:
            #    check_diff(f"logits vs logits_kv (step={i})", logits, logits_kv)
            # Sample
            xcol = model.sample_one_token(logits, temperature=0.0, top_k=None, sample_rng=None)  # B,1
            xcol_kv = model.sample_one_token(logits_kv, temperature=0.0, top_k=None, sample_rng=None)  # B,1
            assert torch.equal(xcol, xcol_kv)
            idx = torch.cat((idx, xcol), dim=1)  # B,T+1  append
            

        gen_text = tokenizer.decode(idx[0].tolist())
        print(gen_text)

    print()
    print("Test 2: Compare naive generation vs KV cache generation with Engine")
    print()
    engine = Engine(model)
    for prompt in prompts:
        tokens = [bos_token] + tokenizer.encode(prompt)
        results, logits = engine.generate_naive(
            tokens,
            num_samples=3,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
            seed=42,
            return_logits=True
        )
        results_kv, logits_kv = engine.generate(
            tokens,
            num_samples=3,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
            seed=42,
            return_logits=True
        )
        for res, res_kv in zip(results, results_kv):
            gen_text = tokenizer.decode(res)
            print(gen_text)
            gen_text_kv = tokenizer.decode(res_kv)
            print(gen_text_kv)
            # check_diff(f"logits vs logits_kv (prompt={prompt})", logits, logits_kv)
            assert res == res_kv
        assert torch.allclose(logits, logits_kv, atol=1e-4, rtol=1e-4)

    print()
    print("Test 3: time naive generation vs KV cache generation with Engine")
    print()

    # Model
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    enable_fa3 = True if torch.cuda.is_available() else False
    with torch.device("meta"):
        model = GPTModel(
            model_config,
            compute_dtype=compute_dtype,
            enable_fa3=enable_fa3,
            fp8_training=True,
            enable_metrics=False,
        )
    model.to_empty(device=device)
    model.init_weights()  # RoPE buffers, rest of weights will be loaded from checkpoint
    print("Model configuration:")
    for k, v in model.config.to_dict().items():
        print(f"  {k:>16}: {v}")

    # Load Model State
    model_path = os.path.join(checkpoints_path, f"model_{latest_checkpoint_step:06d}.pt")
    model_state = torch.load(model_path, map_location=device)
    model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    model.load_state_dict(model_state)
    model.eval()

    # Engine
    engine = Engine(model)

    # Long prompt:
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 100
    tokens = [bos_token] + tokenizer.encode(long_prompt)
    num_samples = 8
    max_new_tokens = 128
    print(f"Timing generation for prompt of length {len(tokens)} tokens, num_samples={num_samples}, max_new_tokens={max_new_tokens}")

    # Warmup
    for _ in range(2):
        _ = engine.generate_naive(
            tokens,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
            seed=42,
            return_logits=False
        )
    # Timing naive generation
    torch.cuda.synchronize() if device.startswith("cuda") else None
    start_time = time.time()
    _ = engine.generate_naive(
        tokens,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=50,
        seed=42,
        return_logits=False
    )
    torch.cuda.synchronize() if device.startswith("cuda") else None
    end_time = time.time()
    total_time_naive = end_time - start_time
    print(f"Total time naive generation: {total_time_naive:.4f} seconds")

    # Warmup
    for _ in range(2):
        _ = engine.generate(
            tokens,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
            seed=42,
            return_logits=False
        )
    # Timing KV cache generation
    torch.cuda.synchronize() if device.startswith("cuda") else None
    start_time = time.time()
    _ = engine.generate(
        tokens,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=50,
        seed=42,
        return_logits=False
    )
    torch.cuda.synchronize() if device.startswith("cuda") else None
    end_time = time.time()
    total_time_kv = end_time - start_time
    print(f"Total time KV cache generation: {total_time_kv:.4f} seconds")
    print(f"Speedup: {total_time_naive / total_time_kv:.2f}x")
    
if __name__ == "__main__":
    main()
