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

@torch.inference_mode()
def main():
    # Compute setup and helpers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    enable_fa3 = True if torch.cuda.is_available() else False

    # Overrides
    # compute_dtype = torch.float32
    # enable_fa3 = False
    
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
    checkpoints_path = os.path.join(BASE_DIR, "runs_sft/scaling3_6e18_d16")
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
        "What is the capital of France?",
        "What is the chemical symbol of gold?",
        "If yesterday was Friday, then what will tomorrow be?",
        "What is the opposite of hot?",
        "What are the planets of the solar system?",
        "What is your favorite color?",
        "If 5*x + 3 = 13, then what is x?",
    ]
    max_new_tokens = None
    bos_token = tokenizer.encode_single_token('<|bos|>')
    user_start_token = tokenizer.encode_single_token('<|user_start|>')
    user_end_token = tokenizer.encode_single_token('<|user_end|>')
    assistant_start_token = tokenizer.encode_single_token('<|assistant_start|>')
    assistant_end_token = tokenizer.encode_single_token('<|assistant_end|>')
    stop_tokens = [assistant_end_token, bos_token]  # stop generation if either token is generated

    print("="*80)
    print()
    print("Test 1: Generate assistant turns on general questions")
    print()
    engine = Engine(model, stop_tokens=stop_tokens)
    for prompt in prompts:
        tokens = [bos_token, user_start_token] + tokenizer.encode(prompt) + [user_end_token, assistant_start_token]
        results = engine.generate(
            tokens,
            num_samples=4,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
            seed=42,
        )
        for res in results:
            gen_text = tokenizer.decode(res)
            print("-"*80)
            print(gen_text)

    
if __name__ == "__main__":
    main()
