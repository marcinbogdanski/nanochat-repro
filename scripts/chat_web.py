import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for older PyTorch
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels progress bars
import time
import uuid
import pickle
import argparse
import threading
from contextlib import asynccontextmanager
import torch
import uvicorn
from pydantic import BaseModel, Field
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from nanorepro.common import get_base_path, UTF8Buffer
from nanorepro.checkpoint import load_model
from nanorepro.engine import Engine
from nanorepro.calculator import CalculatorAndCounter
from nanorepro.tokenizer import ConversationRenderer
BASE_DIR = get_base_path()


parser = argparse.ArgumentParser(description="Run the Nanochat web server")
parser.add_argument('--run', type=str, default="default", help="Current run name (default: 'default').")
parser.add_argument('--temperature', type=float, default=0.6, help='Control randomness, lower values favor top-scoring tokens, higher for more random generation (0.0 is greedy; default: 0.6)')
parser.add_argument('--top-k', type=int, default=50, help='Restricts sampling to the top K highest-scoring tokens (1 is greedy; default: 50)')
parser.add_argument('--max-tokens', type=int, default=512, help='Maximum number of tokens to generate (default: 512)')
# Compute
parser.add_argument('--compute-dtype', type=str, default='bf16', help="Data type for computation, supported: 'bf16', 'fp32').")
parser.add_argument('--no-fa', action='store_true', help="Disable Flash Attention, for reproducibility.")
# Server
parser.add_argument('--host', type=str, default="127.0.0.1", help="Host for the web server (default: '127.0.0.1').")
parser.add_argument('--port', type=int, default=8000, help="Port for the web server (default: 8000).")
args = parser.parse_args()



class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False  # for now

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources here

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
    print("Model configuration:")
    for k, v in model.config.to_dict().items():
        print(f"  {k:>16}: {v}")

    # Generate Test Samples
    convo_renderer = ConversationRenderer(tokenizer)
    stop_tokens = [convo_renderer.assistant_end_token, convo_renderer.bos_token]  # stop generation if either token is generated
    calculator = CalculatorAndCounter(tokenizer)
    engine = Engine(model, stop_tokens=stop_tokens, tool_handler=calculator)    

    app.state.lock = threading.Lock()  # need to synchronize across FastAPI request workers
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.convo_renderer = convo_renderer
    app.state.engine = engine

    yield  # App serves requests during that time

    # Cleanup resources here
    # nothing to destroy, model will be cleaned up automatically


app = FastAPI(lifespan=lifespan)

# Test with:
# curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"nanochat","messages":[{"role":"user","content":"Hello!"}]}'
@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest, request: Request):
    """OpenAI-compatible endpoint to create chat completions"""
    if body.model != "nanochat":
        raise HTTPException(status_code=404, detail="Model not supported")
    if body.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet")

    # Fetch the components
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    convo_renderer = request.app.state.convo_renderer
    engine = request.app.state.engine
    utf8_buffer = UTF8Buffer()

    # Prepare the input tokens
    messages = [message.model_dump() for message in body.messages]
    try:
        conversation_tokens, _ = convo_renderer.render_conversation(messages)  # checks for alternating user/assistant roles etc
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    conversation_tokens.append(convo_renderer.assistant_start_token)

    # Check capacity of position embeddings
    if len(conversation_tokens) + args.max_tokens > model.max_position_embeddings():
        raise HTTPException(status_code=400, detail="Prompt length plus max_tokens exceeds maximum position embeddings")

    # Generate the assistant response text
    assistant_response = ""
    stop_reason = "length"
    with request.app.state.lock:  # synchronize across FastAPI request workers
        for token_column in engine.generate_stream(
            conversation_tokens,
            num_samples=1,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=42,
        ):
            generated_token = token_column[0]    # num_samples=1, so index 0 is our generated token
            if generated_token in engine.stop_tokens:
                stop_reason = "stop"
            else:
                token_bytes = tokenizer.decode_single_token_bytes(generated_token)
                assistant_response += utf8_buffer.decode(token_bytes)
        assistant_response += utf8_buffer.decode(b"", final=True)  # flush any remaining bytes in the UTF-8 buffer


    # Package and return
    result = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "nanochat",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_response  # generated_text
            },
            "finish_reason": stop_reason,
        }]
    }
    return result

@app.get("/v1/models")
def list_models():
    """OpenAI-compatible endpoint to list available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "nanochat",
                "object": "model",
                "created": 0,
                "owned_by": "nanochat-repro",
            }
        ]
    }

@app.get("/")
def read_root():
    """Placeholder for future web endpoint"""
    return {"message": "Hello world"}

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
