"""
OpenAI-compatible chat web server for Nanochat

Loads an SFT-trained model from the "runs_sft" directory at the BASE_DIR (default ~/.cache/nanorepro).

Run with:
uv run python -m scripts.chat_web --run=d18
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for older PyTorch
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable gpt.py kernels progress bars
import time
import json
import uuid
import pickle
import logging
import argparse
import threading
from pathlib import Path
from contextlib import asynccontextmanager
import torch
import uvicorn
from pydantic import BaseModel, Field
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from nanorepro.common import get_base_path, UTF8Buffer
from nanorepro.checkpoint import load_model
from nanorepro.engine import Engine
from nanorepro.calculator import CalculatorAndCounter
from nanorepro.tokenizer import ConversationRenderer
BASE_DIR = get_base_path()
MAX_TOTAL_CONVERSATION_LENGTH = 32000   # max prompt length, utf-8 characters before tokenization
MAX_NUMBER_OF_MESSAGES = 500            # max number of messages in a conversation

# Standard logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Command-line argument parsing
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
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)  # None means use args.temperature as default
    top_k: int | None = Field(default=None, ge=0)                    # None means use args.top_k, max is vocab size, checked at runtime
    max_tokens: int | None = Field(default=None, ge=1, le=4096)      # None means use args.max_tokens, max is to limit abuse

class ResponsesTextPart(BaseModel):
    type: Literal["input_text", "output_text"]
    text: str

class ResponsesMessage(BaseModel):
    type: Literal["message"] = "message"     # optional
    role: Literal["user", "assistant"]
    content: str | list[ResponsesTextPart]

class ResponsesToolOutput(BaseModel):
    type: Literal["logs"]
    logs: str

class ResponsesCodeInterpreterCall(BaseModel):
    type: Literal["code_interpreter_call"]       # required
    # id: str                                      # ignored in our API implementation
    # container_id: str                            # ignored
    # status: Literal["completed", "incomplete"]   # ignored - we close incomplete parts, so model sees conversation history containing properly closed messages
    code: str                                                          # 'code' and 'outputs' (both may be empty, if incomplete) are parsed as part of assistant message content:
    outputs: list[ResponsesToolOutput] = Field(default_factory=list)   # {"content": [{"type": "python", "text": "2+2"}, {"type": "python_output", "text": "4"},]}

class ResponsesRequest(BaseModel):
    model_config = {"extra": "forbid"}  # reject extra pydantic fields
    model: str
    input: str | list[ResponsesMessage | ResponsesCodeInterpreterCall] = Field(min_length=1)   # Pydantic auto resolves input type automatically for us
    stream: Literal[False] = False   # for now only non-streaming is supported
    store: Literal[False] = False   # we don't support server-side message history
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)     # None means use args.temperature as default
    top_k: int | None = Field(default=None, ge=0)                       # None means use args.top_k, max is vocab size, checked at runtime
    max_output_tokens: int | None = Field(default=None, ge=1, le=4096)  # None means use args.max_tokens, max is to limit abuse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI app

    Initializes model, tokenizer, and other resources before the app starts serving requests.
    Currently there is no cleanup needed, as model etc will get released automatically.
    """
    # Compute setup and helpers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[args.compute_dtype]

    # Tokenizer
    tok_base_path = os.path.join(BASE_DIR, "tokenizer")
    tokenizer_path = os.path.join(tok_base_path, "tokenizer.pkl")
    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    # Model Setup
    checkpoints_path = os.path.join(BASE_DIR, "runs_sft", args.run)
    logger.info("Loading model run=%s device=%s dtype=%s", args.run, device, args.compute_dtype)
    model, _ = load_model(
        checkpoints_path=checkpoints_path,
        compute_dtype=compute_dtype,
        enable_fa=not args.no_fa,
        fp8_training=False,    # doesn't matter, inference doesn't use FP8
        enable_metrics=False,  # doesn't matter for inference
        device=device,
        step=None)
    logger.debug("Model configuration: %s", model.config.to_dict())

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
    logger.info("Model loaded run=%s device=%s dtype=%s", args.run, device, args.compute_dtype)

    yield  # App serves requests during that time

    # Cleanup resources here
    # nothing to destroy, model will be cleaned up automatically


app = FastAPI(lifespan=lifespan)

class ChatCompletionsStreamingResponse(StreamingResponse):
    """Stream SSE chunks and make sure to close the generator when response ends
    
    If the client disconnects, StreamingResponse may stop consuming the generator w/o closing it.
    If the generator is suspended at yield, it will not release the model lock and block subsequent requests.
    """
    def __init__(self, sse_chunks_generator):
        super().__init__(sse_chunks_generator, media_type="text/event-stream")
        self.sse_chunks_generator = sse_chunks_generator

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.sse_chunks_generator.close()

def stream_sse_chunks(state, conversation_tokens, completion_id, created_at, model_name, temperature, top_k, max_tokens):
    """Generator that yields Server-Sent Events (SSE) chunks for streaming chat completions."""
    def event(delta, finish_reason):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_at,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,  # None | "stop" | "length"
            }]
        }
        return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"  # ensure_ascii=False so raw stream is more readable

    start_time = time.perf_counter()
    num_generated_tokens = 0
    try:
        yield event({"role": "assistant"}, finish_reason=None)

        # Generate the assistant response text
        utf8_buffer = UTF8Buffer()
        with state.lock:  # synchronize across FastAPI request workers
            for token_column, finish_reasons in state.engine.generate_stream(
                conversation_tokens,
                max_new_tokens=max_tokens,
                num_samples=1,
                temperature=temperature,
                top_k=top_k,
                seed=42,
            ):
                num_generated_tokens += 1
                generated_token = token_column[0]    # num_samples=1, so index 0 is our generated token
                stop_reason = finish_reasons[0]
                if generated_token not in state.engine.stop_tokens:  # if stop_reason is 'length', the last token is a valid part of text
                    token_bytes = state.tokenizer.decode_single_token_bytes(generated_token)
                    text_chunk = utf8_buffer.decode(token_bytes)
                    if text_chunk:
                        yield event({"content": text_chunk}, finish_reason=None)
            text_chunk = utf8_buffer.decode(b"", final=True)  # flush any remaining bytes in the UTF-8 buffer
            if text_chunk:
                yield event({"content": text_chunk}, finish_reason=None)
        yield event({}, finish_reason=stop_reason)  # To ensure we sent stop_reason once, send it separately on it's own
        yield "data: [DONE]\n\n"
    except GeneratorExit:
        logger.info(
            "Stream interrupted id=%s prompt_tokens=%d generated_tokens=%d elapsed_s=%.2f",
            completion_id, len(conversation_tokens), num_generated_tokens, time.perf_counter() - start_time
        )
        raise
    logger.info(
        "Completion OK id=%s stream=True prompt_tokens=%d generated_tokens=%d finish_reason=%s elapsed_s=%.2f",
        completion_id, len(conversation_tokens), num_generated_tokens, stop_reason, time.perf_counter() - start_time
    )

# Test stream=False case:
# curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"nanochat","messages":[{"role":"user","content":"Hello!"}]}'
# Test stream=True case:
# curl -N http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"nanochat","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest, request: Request):
    """OpenAI-compatible endpoint to create chat completions"""
    if body.model != "nanochat":
        raise HTTPException(status_code=404, detail="Model not supported")

    # Fetch the state (tokenizer, model, engine, ..), resolve request params
    state = request.app.state
    temperature = body.temperature if body.temperature is not None else args.temperature
    top_k = body.top_k if body.top_k is not None else args.top_k
    max_tokens = body.max_tokens if body.max_tokens is not None else args.max_tokens
    top_k = None if top_k == 0 else top_k  # convert 0 to None to indicate no top_k filtering
    if top_k is not None and top_k > state.tokenizer.n_vocab:
        raise HTTPException(status_code=400, detail=f"top_k cannot be greater than the vocabulary size ({state.tokenizer.n_vocab})")

    # Check messages length
    messages = [message.model_dump() for message in body.messages]
    if len(messages) > MAX_NUMBER_OF_MESSAGES:
        raise HTTPException(status_code=400, detail=f"Number of messages exceeds the maximum allowed")
    if sum(len(msg["content"]) for msg in messages) > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail="Total conversation length exceeds maximum allowed")

    # Prepare the input tokens
    try:
        conversation_tokens, _ = state.convo_renderer.render_conversation(messages)  # checks for alternating user/assistant roles etc
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    conversation_tokens.append(state.convo_renderer.assistant_start_token)

    # Check capacity of position embeddings
    if len(conversation_tokens) + max_tokens > state.model.max_position_embeddings():
        raise HTTPException(status_code=400, detail="Prompt length plus max_tokens exceeds maximum position embeddings")

    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created_at = int(time.time())
    model_name = "nanochat"
    if body.stream:
        # OpenAI-compatible stream mode implementation
        sse_chunks_generator = stream_sse_chunks(state, conversation_tokens, completion_id, created_at, model_name, temperature, top_k, max_tokens)
        return ChatCompletionsStreamingResponse(sse_chunks_generator)

    else:
        # OpenAI-compatible non-stream mode implementation

        # Run the model
        start_time = time.perf_counter()
        with state.lock:  # synchronize across FastAPI request workers
            new_token_rows, finish_reasons = state.engine.generate_batch(
                conversation_tokens,
                max_new_tokens=max_tokens,
                num_samples=1,
                temperature=temperature,
                top_k=top_k,
                seed=42,
            )
        # Decode the generated tokens into a string
        generated_tokens = new_token_rows[0]    # num_samples=1, so index 0 is our generated token sequence
        stop_reason = finish_reasons[0]
        if generated_tokens[-1] in state.engine.stop_tokens:
            generated_tokens = generated_tokens[:-1]  # remove terminal token if present
        assistant_response = state.tokenizer.decode(generated_tokens)
        logger.info(
            "Completion OK id=%s stream=False prompt_tokens=%d generated_tokens=%d finish_reason=%s elapsed_s=%.2f",
            completion_id, len(conversation_tokens), len(new_token_rows[0]), stop_reason, time.perf_counter() - start_time
        )
        # Package and return
        result = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_at,
            "model": model_name,
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

def message_to_responses_output(message_dict):
    """Convert decoded message dictionary into OpenAI-compatible responses output"""
    msg_parts = message_dict['content']
    assert isinstance(msg_parts, list)

    output = []
    for part in msg_parts:
        if part["type"] == "text":
            output.append({
                "id": "msg_" + uuid.uuid4().hex,
                "type": "message",
                "role": "assistant",
                "status": part["status"],
                "content": [{"type": "output_text", "text": part['text'], "annotations": []}]
            })
        elif part["type"] == "python":
            output.append({
                "id": "ci_" + uuid.uuid4().hex,
                "type": "code_interpreter_call",    # closest official OpenAI compatible type
                "container_id": "python_ast_parser",  # fake container ID
                "status": "incomplete",  # completeness of code_interpreter_call is determined by the presence of completed output part
                "code": part["text"],
                "outputs": [],
            })
        elif part["type"] == "python_output":
            # MessageDecoder guarantees that a python_output part always follows a python part
            output[-1]["outputs"].append({
                "type": "logs",
                "logs": part["text"],
            })
            output[-1]["status"] = part["status"]  # mark whole code_interpreter_call as complete/incomplete

    return output



def convert_responses_input_to_messages(responses_input):
    """Convert OpenAI-compatible responses input into internal message format
    
    Target result shape, roughly:
        messages = [
            {
                "role": "user",
                "content": "Hello!"
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hi there!"},
                    {"type": "python", "text": "print('Hello World')"},
                    {"type": "python_output", "text": "Hello World"},
                    {"type": "text", "text": "Goodbye!"},
                ]
            }
        ]
    """
    if isinstance(responses_input, str):
        return [{"role": "user", "content": responses_input}]

    # First pass, extract flat list of:
    #   {"role": ..., "type": ..., "text": ...}
    content_list = []
    for resp_msg_or_code_int_call in responses_input:
        if isinstance(resp_msg_or_code_int_call, ResponsesMessage):
            resp_message = resp_msg_or_code_int_call
            if isinstance(resp_message.content, str):
                content_list.append({"role": resp_message.role, "type": "text", "text": resp_message.content})
            elif isinstance(resp_message.content, list):
                resp_text_part_list = resp_message.content
                if not resp_text_part_list:
                    content_list.append({"role": resp_message.role, "type": "text", "text": ""})  # preserve role boundary if content is empty []
                else:
                    for text_part in resp_text_part_list:
                        content_list.append({"role": resp_message.role, "type": "text", "text": text_part.text})
            else:
                raise ValueError(f"Unsupported content type: {type(resp_message.content)}")
        elif isinstance(resp_msg_or_code_int_call, ResponsesCodeInterpreterCall):
            code_int_call = resp_msg_or_code_int_call
            content_list.append({"role": "assistant", "type": "python", "text": code_int_call.code})
            if code_int_call.outputs:
                output_text = "".join(out.logs for out in code_int_call.outputs)
                content_list.append({"role": "assistant", "type": "python_output", "text": output_text})
        else:
            raise ValueError(f"Unsupported response type: {type(resp_msg_or_code_int_call)}")

    # Second pass, group by role and convert to the target message format
    messages = []
    current_role = None
    for content_item in content_list:
        role = content_item["role"]
        if role != current_role:
            messages.append({"role": role, "content": []})
            current_role = role
        messages[-1]["content"].append({"type": content_item["type"], "text": content_item["text"]})

    # Third pass: convert user content from list-of-dict to just string
    for message in messages:
        if message["role"] == "user":
            message["content"] = "".join(part["text"] for part in message["content"])

    return messages



# Test stream=False case:
# curl http://127.0.0.1:8000/v1/responses -H 'Content-Type: application/json' -d '{"model":"nanochat","input":"Hello!"}'
@app.post("/v1/responses")
def responses(body: ResponsesRequest, request: Request):
    """OpenAI-compatible responses endpoint. This endpoint does return server-side tool calls."""
    if body.model != "nanochat":
        raise HTTPException(status_code=404, detail="Model not supported")

    # Fetch the state (tokenizer, model, engine, ..), resolve request params
    state = request.app.state
    temperature = body.temperature if body.temperature is not None else args.temperature
    top_k = body.top_k if body.top_k is not None else args.top_k
    max_tokens = body.max_output_tokens if body.max_output_tokens is not None else args.max_tokens
    top_k = None if top_k == 0 else top_k  # convert 0 to None to indicate no top_k filtering
    if top_k is not None and top_k > state.tokenizer.n_vocab:
        raise HTTPException(status_code=400, detail=f"top_k cannot be greater than the vocabulary size ({state.tokenizer.n_vocab})")

    # Check messages length
    messages = convert_responses_input_to_messages(body.input)
    if len(messages) > MAX_NUMBER_OF_MESSAGES:
        raise HTTPException(status_code=400, detail=f"Number of messages exceeds the maximum allowed")
    msg_len = lambda m: len(m["content"]) if m["role"] == "user" else sum(len(part["text"]) for part in m["content"])
    if sum(msg_len(msg) for msg in messages) > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail="Total conversation length exceeds maximum allowed")

    # Prepare the input tokens
    try:
        conversation_tokens, _ = state.convo_renderer.render_conversation(messages)  # checks for alternating user/assistant roles etc
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    conversation_tokens.append(state.convo_renderer.assistant_start_token)

    # Check capacity of position embeddings
    if len(conversation_tokens) + max_tokens > state.model.max_position_embeddings():
        raise HTTPException(status_code=400, detail="Prompt length plus max_tokens exceeds maximum position embeddings")

    response_id = "resp_" + uuid.uuid4().hex
    created_at = int(time.time())
    model_name = "nanochat"
    if body.stream:
        # OpenAI-compatible stream mode implementation
        # sse_chunks_generator = stream_sse_chunks(state, conversation_tokens, completion_id, created_at, model_name, temperature, top_k, max_tokens)
        # return ChatCompletionsStreamingResponse(sse_chunks_generator)
        raise HTTPException(status_code=501, detail="Streaming mode is not implemented")

    else:
        # OpenAI-compatible non-stream mode implementation

        # Run the model
        start_time = time.perf_counter()
        with state.lock:  # synchronize across FastAPI request workers
            new_token_rows, finish_reasons = state.engine.generate_batch(
                conversation_tokens,
                max_new_tokens=max_tokens,
                num_samples=1,
                temperature=temperature,
                top_k=top_k,
                seed=42,
            )
        # Decode the generated tokens into a structured message dictionary
        generated_tokens = new_token_rows[0]    # num_samples=1, so index 0 is our generated token sequence
        stop_reason = finish_reasons[0]
        decoded_message_dict = state.convo_renderer.decode_single_message(
            [state.convo_renderer.assistant_start_token] + generated_tokens  # prepend assistant_start to form full message
        )
        output = message_to_responses_output(decoded_message_dict)
        logger.info(
            "Response OK id=%s stream=False prompt_tokens=%d generated_tokens=%d finish_reason=%s elapsed_s=%.2f",
            response_id, len(conversation_tokens), len(new_token_rows[0]), stop_reason, time.perf_counter() - start_time
        )

        # Package and return
        result = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": model_name,
            "status": "incomplete" if stop_reason == "length" else "completed",
            "error": None,
            "incomplete_details": {"reason": "max_output_tokens"} if stop_reason == "length" else None,
            "output": output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [{"type": "code_interpreter", "container": "python_ast_parser"}],
            "store": False,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "usage": {
                "input_tokens": len(conversation_tokens),
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": len(generated_tokens),  # includes tool output tokens
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": len(conversation_tokens) + len(generated_tokens),
            },
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

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    icon_path = Path(__file__).resolve().parent.parent / "assets" / "favicon.ico"
    return FileResponse(icon_path, media_type="image/x-icon")

@app.get("/tiger_logo.svg", include_in_schema=False)
def tiger_logo():
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "tiger_logo.svg"
    return FileResponse(logo_path, media_type="image/svg+xml")

@app.get("/")
def read_root():
    """Serve the UI"""
    ui_path = Path(__file__).resolve().parent.parent / "nanorepro" / "ui.html"
    return FileResponse(ui_path, media_type="text/html")

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
