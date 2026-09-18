import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for older PyTorch
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels progress bars
import json
import pickle
import argparse
import torch
import torch.nn.functional as F
from nanorepro.gpt import GPTConfig, GPTModel
from nanorepro.checkpoint import get_latest_checkpoint_step
from nanorepro.common import get_base_path
from nanorepro.engine import Engine
from nanorepro.calculator import CalculatorAndCounter
BASE_DIR = get_base_path()

@torch.inference_mode()
def main():


    parser = argparse.ArgumentParser(description='Chat with the trained SFT model.')
    parser.add_argument('--run', type=str, default="default", help="Current run name (default: 'default').")
    parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt for the model, returns one response (omit for interactive mode)')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Control randomness, lower values favor top-scoring tokens, higher for more random generation (0.0 is greedy; default: 0.6)')
    parser.add_argument('-k', '--top-k', type=int, default=50, help='Restricts sampling to the top K highest-scoring tokens (1 is greedy; default: 50)')
    args = parser.parse_args()

    # Compute setup and helpers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    enable_fa = True if torch.cuda.is_available() else False

    # Tokenizer
    tok_base_path = os.path.join(BASE_DIR, "tokenizer")
    tokenizer_path = os.path.join(tok_base_path, "tokenizer.pkl")
    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    # Model Setup
    checkpoints_path = os.path.join(BASE_DIR, "runs_sft", args.run)
    latest_checkpoint_step = get_latest_checkpoint_step(checkpoints_path)
    latest_meta_path = os.path.join(checkpoints_path, f"meta_{latest_checkpoint_step:06d}.json")  # last saved file
    with open(latest_meta_path, "r") as f:
        pretrain_metadata = json.load(f)
    model_config = GPTConfig(**pretrain_metadata["model_config"])
    with torch.device("meta"):
        model = GPTModel(
            model_config,
            compute_dtype=compute_dtype,
            enable_fa=enable_fa,
            fp8_training=True,
            enable_metrics=False,
        )
    model.to_empty(device=device)
    model.init_weights()  # RoPE buffers, rest of weights will be loaded from checkpoint
    if not args.prompt:
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
    bos_token = tokenizer.encode_single_token('<|bos|>')
    user_start_token = tokenizer.encode_single_token('<|user_start|>')
    user_end_token = tokenizer.encode_single_token('<|user_end|>')
    assistant_start_token = tokenizer.encode_single_token('<|assistant_start|>')
    assistant_end_token = tokenizer.encode_single_token('<|assistant_end|>')
    stop_tokens = [assistant_end_token, bos_token]  # stop generation if either token is generated
    calculator = CalculatorAndCounter(tokenizer)
    engine = Engine(model, stop_tokens=stop_tokens, tool_handler=calculator)

    if not args.prompt:
        print()
        print("Nanochat-Repro Chat Mode")
        print("-" * 80)
        print("Commands:")
        print("  /quit, /exit - end the conversation and exit")
        print("  /clear       - reset the conversation")
        print("-" * 80)

    conversation_tokens = [bos_token]
    while True:

        if not args.prompt:
            print("\nUSER:")
            try:
                user_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("Exiting.")
                break

            if user_input in ("/quit", "/exit"):
                print("Exiting.")
                break

            if user_input == "/clear":
                conversation_tokens = [bos_token]
                print("Starting new conversation.")
                continue

            if user_input == "":
                continue

            print("\nASSISTANT:")
        else:
            user_input = args.prompt

        conversation_tokens += [user_start_token] + tokenizer.encode(user_input) + [user_end_token, assistant_start_token]
        for token_column in engine.generate_stream(
            conversation_tokens,
            num_samples=1,
            max_new_tokens=256,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=42,
        ):
            generated_token = token_column[0]    # num_samples=1, so index 0 is our generated token
            conversation_tokens.append(generated_token)  # includes <assistant_end>
            if generated_token != assistant_end_token:
                generated_text = tokenizer.decode([generated_token])  # this can generate incomplete or invalid UTF-8 string, should handle it properly
                print(generated_text, end='', flush=True)

        # In case conversation ends due to max_tokens, we need to append assistant end token
        if conversation_tokens[-1] != assistant_end_token:
            conversation_tokens.append(assistant_end_token)
        print()

        if args.prompt:
            break


if __name__ == "__main__":
    main()
