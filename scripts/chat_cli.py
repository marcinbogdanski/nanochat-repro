import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for older PyTorch
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels progress bars
import sys
import pickle
import argparse
import torch
from nanorepro.checkpoint import load_model
from nanorepro.common import get_base_path, UTF8Buffer
from nanorepro.engine import Engine
from nanorepro.calculator import CalculatorAndCounter
BASE_DIR = get_base_path()

COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"
COLOR_CYAN = "\033[36m"

@torch.inference_mode()
def main():

    parser = argparse.ArgumentParser(description='Chat with the trained SFT model.')
    parser.add_argument('--run', type=str, default="default", help="Current run name (default: 'default').")
    parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt for the model, returns one response (omit for interactive mode)')
    parser.add_argument('--temperature', type=float, default=0.6, help='Control randomness, lower values favor top-scoring tokens, higher for more random generation (0.0 is greedy; default: 0.6)')
    parser.add_argument('--top-k', type=int, default=50, help='Restricts sampling to the top K highest-scoring tokens (1 is greedy; default: 50)')
    # Compute
    parser.add_argument('--compute-dtype', type=str, default='bf16', help="Data type for computation, supported: 'bf16', 'fp32').")
    parser.add_argument('--no-fa', action='store_true', help="Disable Flash Attention, for reproducibility.")    
    args = parser.parse_args()

    # Compute setup and helpers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[args.compute_dtype]

    # Tokenizer
    tok_base_path = os.path.join(BASE_DIR, "tokenizer")
    tokenizer_path = os.path.join(tok_base_path, "tokenizer.pkl")
    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    # Model Setup
    checkpoints_path = os.path.join(BASE_DIR, "runs_sft", args.run)
    model, _ = load_model(
        checkpoints_path=checkpoints_path,
        compute_dtype=compute_dtype,
        enable_fa=not args.no_fa,
        fp8_training=False,    # doesn't matter, inference doesn't use FP8
        enable_metrics=False,  # doesn't matter for inference
        device=device,
        step=None)
    if not args.prompt:
        print("Model configuration:")
        for k, v in model.config.to_dict().items():
            print(f"  {k:>16}: {v}")

    # Generate Test Samples
    bos_token = tokenizer.encode_single_token('<|bos|>')
    user_start_token = tokenizer.encode_single_token('<|user_start|>')
    user_end_token = tokenizer.encode_single_token('<|user_end|>')
    assistant_start_token = tokenizer.encode_single_token('<|assistant_start|>')
    assistant_end_token = tokenizer.encode_single_token('<|assistant_end|>')
    python_start_token = tokenizer.encode_single_token('<|python_start|>')
    python_end_token = tokenizer.encode_single_token('<|python_end|>')
    output_start_token = tokenizer.encode_single_token('<|output_start|>')
    output_end_token = tokenizer.encode_single_token('<|output_end|>')
    special_tokens = [bos_token, user_start_token, user_end_token, assistant_start_token, assistant_end_token,
                      python_start_token, python_end_token, output_start_token, output_end_token]
    stop_tokens = [assistant_end_token, bos_token]  # stop generation if either token is generated
    calculator = CalculatorAndCounter(tokenizer)
    engine = Engine(model, stop_tokens=stop_tokens, tool_handler=calculator)
    utf8_buffer = UTF8Buffer()

    use_color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    def colorize(text, color):
        if use_color:
            return f"{color}{text}{COLOR_RESET}"
        return text
    def colorize_special_token(text, token):
        if token in special_tokens:
            return colorize(text, COLOR_CYAN)  # 36 cyan for special tokens
        return text

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
            print(colorize("\nUSER:", COLOR_BOLD))
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

            print(colorize("\nASSISTANT:", COLOR_BOLD))
        else:
            user_input = args.prompt

        conversation_tokens += [user_start_token] + tokenizer.encode(user_input) + [user_end_token, assistant_start_token]
        for token_column, finish_reasons in engine.generate_stream(
            conversation_tokens,
            max_new_tokens=256,
            num_samples=1,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=42,
        ):
            generated_token = token_column[0]    # num_samples=1, so index 0 is our generated token
            conversation_tokens.append(generated_token)  # includes <assistant_end>
            if generated_token not in stop_tokens:
                token_bytes = tokenizer.decode_single_token_bytes(generated_token)
                generated_text = utf8_buffer.decode(token_bytes)
                colorized_text = colorize_special_token(generated_text, generated_token)
                print(colorized_text, end='', flush=True)

        # In case conversation ends due to max_tokens, we need to append assistant end token
        if conversation_tokens[-1] != assistant_end_token:
            if conversation_tokens[-1] == bos_token:
                conversation_tokens[-1] = assistant_end_token
            else:
                conversation_tokens.append(assistant_end_token)

        print(utf8_buffer.decode(b"", final=True), end="", flush=True)  # flush any remaining bytes
        print()

        if args.prompt:
            break

    

if __name__ == "__main__":
    main()
