"""
OpenAI-compatible chat web server for Nanochat

Loads an SFT-trained model from the "runs_sft" directory at the BASE_DIR (default ~/.cache/nanorepro).
Logs API requests to BASE_DIR/webchat_log.jsonl.

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
import random
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager, closing
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
from nanorepro.tokenizer import ConversationRenderer, MessageDecoder
BASE_DIR = get_base_path()
MAX_TOTAL_CONVERSATION_LENGTH = 32000   # max prompt length, utf-8 characters before tokenization
MAX_NUMBER_OF_MESSAGES = 500            # max number of messages in a conversation

# Standard logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Chat logger
def make_chat_logger():
    """Chat logger that logs all API requests to a file"""
    chat_logger = logging.getLogger("webchat")
    chat_logger.setLevel(logging.INFO)
    chat_logger.propagate = False  # don't also print conversations to console
    handler = logging.FileHandler(
        Path(BASE_DIR) / "webchat_log.jsonl",
        mode="a",
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    chat_logger.addHandler(handler)
    return chat_logger
chat_logger = make_chat_logger()

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Run the Nanochat web server")
parser.add_argument('--run', type=str, default="default", help="Current run name (default: 'default').")
parser.add_argument('--temperature', type=float, default=0.6, help='Control randomness, lower values favor top-scoring tokens, higher for more random generation (0.0 is greedy; default: 0.6)')
parser.add_argument('--top-k', type=int, default=50, help='Restricts sampling to the top K highest-scoring tokens (1 is greedy; default: 50)')
parser.add_argument('--max-tokens', type=int, default=512, help='Maximum number of tokens to generate (default: 512)')
parser.add_argument('--seed', type=int, default=None, help='Random seed for generation (default: None, meaning random seed every request)')
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
    seed: int | None = Field(default=None, ge=0, le=2**64-1)         # None means use args.seed (which by default is also None, meaning random seed every request)

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
    stream: bool = False
    store: Literal[False] = False   # we don't support server-side message history
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)     # None means use args.temperature as default
    top_k: int | None = Field(default=None, ge=0)                       # None means use args.top_k, max is vocab size, checked at runtime
    max_output_tokens: int | None = Field(default=None, ge=1, le=4096)  # None means use args.max_tokens, max is to limit abuse
    seed: int | None = Field(default=None, ge=0, le=2**64-1)            # None means use args.seed (which by default is also None, meaning random seed every request)

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

def log_chat_request(request: Request, body: ResponsesRequest, response_id: str):
    """Append the validated request to the conversation log."""
    record = {
        "id": response_id,
        "time": datetime.now(timezone.utc).isoformat(),
        "ip": request.client.host if request.client else None,
        "run": args.run,
        "request": body.model_dump(mode="json"),
    }
    chat_logger.info("%s", json.dumps(record, ensure_ascii=False))

def make_completion(completion_id, created_at, model_name, text, finish_reason):
    """Build OpenAI-compatible chat completion"""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_at,
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
    }

class ChatStreamingResponse(StreamingResponse):
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

def stream_completions_sse_chunks(state, conversation_tokens, completion_id, created_at, model_name, temperature, top_k, max_tokens, seed):
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
        # state.lock - synchronizes across FastAPI request workers
        # closing() ensures the generator is properly and promptly closed when exiting the context
        with state.lock, closing(state.engine.generate_stream(
            conversation_tokens,
            max_new_tokens=max_tokens,
            num_samples=1,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )) as token_generator:
            for token_column, finish_reasons in token_generator:
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
            "Stream interrupted id=%s prompt_tokens=%d generated_tokens=%d elapsed_s=%.2f seed=%s",
            completion_id, len(conversation_tokens), num_generated_tokens, time.perf_counter() - start_time, seed
        )
        raise
    logger.info(
        "Completion OK id=%s stream=True prompt_tokens=%d generated_tokens=%d finish_reason=%s elapsed_s=%.2f seed=%s",
        completion_id, len(conversation_tokens), num_generated_tokens, stop_reason, time.perf_counter() - start_time, seed
    )

# Test stream=False case:
# curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"nanochat","messages":[{"role":"user","content":"Hello!"}]}'
# Test stream=True case:
# curl -N http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"nanochat","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest, request: Request):
    """OpenAI-compatible endpoint to create chat completions"""
    completion_id = "chatcmpl-" + uuid.uuid4().hex
    log_chat_request(request, body, completion_id)

    if body.model != "nanochat":
        raise HTTPException(status_code=404, detail="Model not supported")

    # Fetch the state (tokenizer, model, engine, ..), resolve request params
    state = request.app.state
    temperature = body.temperature if body.temperature is not None else args.temperature
    top_k = body.top_k if body.top_k is not None else args.top_k
    max_tokens = body.max_tokens if body.max_tokens is not None else args.max_tokens
    seed = body.seed if body.seed is not None else args.seed
    top_k = None if top_k == 0 else top_k  # convert 0 to None to indicate no top_k filtering
    if top_k is not None and top_k > state.tokenizer.n_vocab:
        raise HTTPException(status_code=400, detail=f"top_k cannot be greater than the vocabulary size ({state.tokenizer.n_vocab})")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

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

    created_at = int(time.time())
    model_name = "nanochat"
    if body.stream:
        # OpenAI-compatible stream mode implementation
        sse_chunks_generator = stream_completions_sse_chunks(state, conversation_tokens, completion_id, created_at, model_name, temperature, top_k, max_tokens, seed)
        return ChatStreamingResponse(sse_chunks_generator)

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
                seed=seed,
            )
        # Decode the generated tokens into a string
        generated_tokens = new_token_rows[0]    # num_samples=1, so index 0 is our generated token sequence
        stop_reason = finish_reasons[0]
        if generated_tokens[-1] in state.engine.stop_tokens:
            generated_tokens = generated_tokens[:-1]  # remove terminal token if present
        assistant_response = state.tokenizer.decode(generated_tokens)
        logger.info(
            "Completion OK id=%s stream=False prompt_tokens=%d generated_tokens=%d finish_reason=%s elapsed_s=%.2f seed=%s",
            completion_id, len(conversation_tokens), len(new_token_rows[0]), stop_reason, time.perf_counter() - start_time, seed
        )
        # Package and return
        result = make_completion(completion_id, created_at, model_name, assistant_response, stop_reason)
        return result


def make_response(response_id, created_at, model_name, temperature, max_tokens, output, status, num_input_tokens, num_output_tokens):
    """Build the final response object. Used in both batch and streaming modes."""
    result = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model_name,
        "status": status,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [{"type": "code_interpreter", "container": "python_ast_parser"}],
        "store": False,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "usage": {
            "input_tokens": num_input_tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": num_output_tokens,  # includes tool output tokens
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": num_input_tokens + num_output_tokens,
        },
    }
    return result


def stream_responses_sse_chunks(state, conversation_tokens, response_id, created_at, model_name, temperature, top_k, max_tokens, seed):
    """Generator that yields Server-Sent Events (SSE) chunks for streaming responses API

    Example output, according to Responses API (one assistant turn, including text, tool call, tool result and more):
        [
            {
                "id": "msg_abc",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Let me calculate that.", "annotations": []}]
            },
            {
                "id": "ci_def",
                "type": "code_interpreter_call",
                "container_id": "python_ast_parser",
                "status": "completed",
                "code": "2+2",
                "outputs": [{"type": "logs", "logs": "4"}]
            },
            {
                "id": "msg_ghi",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "The answer is 4.", "annotations": []}]
            }
        ]

    Relevant SSE stream operations that we need to send (for the example above):
        response.created                              Response created; includes ID and metadata
        response.in_progress                          Notify that generation is underway

        response.output_item.added                    First assistant message created
        response.content_part.added                   Empty text part added to that message
        response.output_text.delta                    Append generated text
        response.output_text.done                     Full finished text of this part
        response.content_part.done                    Full finished text part
        response.output_item.done                     Full first assistant message, including status

        response.output_item.added                    Code-interpreter item created
        response.code_interpreter_call.in_progress    Notify that the call is in progress
        response.code_interpreter_call_code.delta     Append generated Python code
        response.code_interpreter_call_code.done      Full, finished Python code
        response.code_interpreter_call.interpreting   Notify that the Python-output part has started
        <no events>                                   Python output, i.e. logs, accumulate here; no log-delta events
        response.code_interpreter_call.completed      Python output finished successfully
        response.output_item.done                     Full tool item: code, logs, status

        response.output_item.added                    Second assistant message created
        response.content_part.added                   Empty text part added
        response.output_text.delta                    Append generated text
        response.output_text.done                     Full text of this part
        response.content_part.done                    Full finished text part
        response.output_item.done                     Full second message, including status
        response.completed                            Full final response, including all messages and code-interpreter calls

    The web client in ui.html needs to decode it, reconstruct the objects as they stream and render in real time. Fun!
    """
    sequence_number = 0
    def sse_event(event_type, **fields):
        nonlocal sequence_number
        chunk = {"type": event_type, "sequence_number": sequence_number, **fields}
        sequence_number += 1
        return "event: " + event_type + "\ndata: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"

    output = []            # list of ResponsesMessage and/or ResponsesCodeInterpreterCall
    current_item = None    # current ResponsesMessage or ResponsesCodeInterpreterCall being constructed
    num_generated_tokens = 0
    stop_reason = "length"
    start_time = time.perf_counter()
    try:
        response = make_response(
            response_id=response_id,
            created_at=created_at,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            output=[],               # initially empty, will be populated as the stream progresses
            status="in_progress",    # will update to 'completed' at the end of the stream
            num_input_tokens=len(conversation_tokens),
            num_output_tokens=0,
        )
        yield sse_event("response.created", response=response)
        yield sse_event("response.in_progress", response=response)

        decoder = MessageDecoder(state.tokenizer)
        decoder.feed(state.convo_renderer.assistant_start_token)
        # state.lock - synchronizes across FastAPI request workers
        # closing() ensures the generator is properly and promptly closed when exiting the context
        with state.lock, closing(state.engine.generate_stream(
            conversation_tokens,
            max_new_tokens=max_tokens,
            num_samples=1,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )) as token_generator:
            for token_column, finish_reasons in token_generator:
                num_generated_tokens += 1
                generated_token = token_column[0]    # num_samples=1, so index 0 is our generated token
                stop_reason = finish_reasons[0]

                # MessageStreamEvent(
                #   type="role_start|part_start|delta|part_end|role_end",
                #   value="user|assistant|text|python|python_output",
                #   status="completed|incomplete")
                events = decoder.feed(generated_token)
                if stop_reason is not None:
                    events.extend(decoder.end_message_stream())  # close any truncated part and process in same loop

                # MessageDecoder guarantees valid event stream, so we don't need to validate too much in the loop
                for event in events:
                    if event.type == "role_start":
                        # We already feed in assistant_start token, and not expecting any new role_start events
                        raise ValueError("Unexpected role_start event")

                    elif event.type == "part_start":
                        current_part = event.value
                        if current_part == "text":
                            # Add message
                            current_item = {"id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant", "status": "in_progress", "content": []}
                            output.append(current_item)
                            yield sse_event("response.output_item.added", output_index=len(output)-1, item=current_item)
                            # Add empty content part
                            current_item["content"].append({"type": "output_text", "text": "", "annotations": []})
                            yield sse_event("response.content_part.added", output_index=len(output)-1, item_id=current_item["id"], content_index=0, part=current_item["content"][0])
                        elif current_part == "python":
                            # Add code interpreter call
                            current_item = {"id": "ci_" + uuid.uuid4().hex, "type": "code_interpreter_call", "container_id": "python_ast_parser", "status": "in_progress", "code": "", "outputs": []}
                            output.append(current_item)
                            yield sse_event("response.output_item.added", output_index=len(output)-1, item=current_item)
                            # Mark 'in_progress'
                            yield sse_event("response.code_interpreter_call.in_progress", output_index=len(output)-1, item_id=current_item["id"])
                        elif current_part == "python_output":
                            # Indicate the start of the Python output section
                            yield sse_event("response.code_interpreter_call.interpreting", output_index=len(output)-1, item_id=current_item["id"])
                            # Responses have no standard event for "tool logs started", so we create empty space, will append logs here and send together when part ends
                            current_item["outputs"].append({"type": "logs", "logs": ""})
                        else:
                            raise ValueError(f"Unknown content part type: {current_part}")

                    elif event.type == "delta":
                        if current_part == "text":
                            current_item["content"][0]["text"] += event.value   # we create exactly one text part per message
                            yield sse_event("response.output_text.delta", output_index=len(output)-1, item_id=current_item["id"], content_index=0, delta=event.value, logprobs=[])
                        elif current_part == "python":
                            current_item["code"] += event.value
                            yield sse_event("response.code_interpreter_call_code.delta", output_index=len(output)-1, item_id=current_item["id"], delta=event.value)
                        elif current_part == "python_output":
                            # Append last tool output log; there is no standard event to send, so we accumulate and send together when part ends
                            current_item["outputs"][-1]["logs"] += event.value
                        else:
                            raise ValueError(f"Unknown content part type: {current_part}")

                    elif event.type == "part_end":
                        if current_part == "text":
                            assert event.value == "text"   # avoid footguns
                            part = current_item["content"][0]   # we create exactly one text part per message
                            # Mark the text part done
                            yield sse_event("response.output_text.done", output_index=len(output)-1, item_id=current_item["id"], content_index=0, text=part["text"], logprobs=[])
                            # Mark the content part done (or incomplete if it was cut short)
                            yield sse_event("response.content_part.done", output_index=len(output)-1, item_id=current_item["id"], content_index=0, part=part)
                            current_item["status"] = event.status
                            # Mark the output item done (i.e. message)
                            yield sse_event("response.output_item.done", output_index=len(output)-1, item=current_item)
                            current_item = None
                        elif current_part == "python":
                            assert event.value == "python"
                            # Mark the code part done. It has ended either normally or was cut short
                            yield sse_event("response.code_interpreter_call_code.done", output_index=len(output)-1, item_id=current_item["id"], code=current_item["code"])
                            # current_item = None   # do DO NOT close the code_interpreter_call item, it will be closed when the python_output is sent
                        elif current_part == "python_output":
                            assert event.value == "python_output"
                            current_item["status"] = event.status
                            if event.status == "completed":
                                # Mark the code_interpreter_call as done
                                yield sse_event("response.code_interpreter_call.completed", output_index=len(output)-1, item_id=current_item["id"])
                            # Send whole code_interpreter_call item, including the Python outputs (the "logs")
                            yield sse_event("response.output_item.done", output_index=len(output)-1, item=current_item)
                            current_item = None
                        else:
                            raise ValueError(f"Unexpected current_part: {current_part}")
                        current_part = None

                    elif event.type == "role_end":
                        if current_item is not None:
                            # This code path triggers, because we have an output item that was not closed
                            # Text and python_output parts close their items (current_item=None)
                            # A python part leaves it's tool item open, since it's waiting for the upcoming python_output
                            # (MessageDecoder should guarantee event protocol integrity so we exclude protocol violations)
                            assert current_part == None
                            assert current_item["type"] == "code_interpreter_call"
                            current_item["status"] = "incomplete"
                            yield sse_event("response.output_item.done", output_index=len(output)-1, item=current_item)
                            current_item = None

                    else:
                        raise ValueError(f"Unexpected event type: {event.type}")

        # Prepare the final response and yield it as an the final SSE event
        status = "incomplete" if stop_reason == "length" else "completed"
        response = make_response(
            response_id=response_id,
            created_at=created_at,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            output=output,               # populate with final output items
            status=status,               # update with final status
            num_input_tokens=len(conversation_tokens),
            num_output_tokens=num_generated_tokens,
        )
        yield sse_event("response." + status, response=response)
        logger.info("Response OK id=%s stream=True prompt_tokens=%d generated_tokens=%d finish_reason=%s elapsed_s=%.2f seed=%s",
                    response_id, len(conversation_tokens), num_generated_tokens, stop_reason, time.perf_counter() - start_time, seed)

    except GeneratorExit:
        logger.info("Response stream interrupted id=%s prompt_tokens=%d generated_tokens=%d elapsed_s=%.2f seed=%s",
                    response_id, len(conversation_tokens), num_generated_tokens, time.perf_counter() - start_time, seed)
    except Exception as error:
        logger.exception("Response stream failed id=%s", response_id)    # logger.exception includes error details
        response = make_response(
            response_id=response_id,
            created_at=created_at,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            output=output,
            status="failed",
            num_input_tokens=len(conversation_tokens),
            num_output_tokens=num_generated_tokens,
        )
        response["error"] = {"code": "server_error", "message": "Response generation failed"}
        yield sse_event("response.failed", response=response)


def message_to_responses_output(message_dict):
    """Convert decoded message dictionary into OpenAI-compatible responses output.

    Example output (single assistant response, including text, tool call, tool result and more text):
        [
            {
                "id": "msg_abc",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Let me calculate that.", "annotations": []}]
            },
            {
                "id": "ci_def",
                "type": "code_interpreter_call",
                "container_id": "python_ast_parser",
                "status": "completed",
                "code": "2+2",
                "outputs": [{"type": "logs", "logs": "4"}]
            },
            {
                "id": "msg_ghi",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "The answer is 4.", "annotations": []}]
            }
        ]
    """
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
# Test stream=True case:
# curl -N http://127.0.0.1:8000/v1/responses -H 'Content-Type: application/json' -d '{"model":"nanochat","input":"Hello!","stream":true}'
@app.post("/v1/responses")
def responses(body: ResponsesRequest, request: Request):
    """OpenAI-compatible responses endpoint. This endpoint does return server-side tool calls."""
    response_id = "resp_" + uuid.uuid4().hex
    log_chat_request(request, body, response_id)

    if body.model != "nanochat":
        raise HTTPException(status_code=404, detail="Model not supported")

    # Fetch the state (tokenizer, model, engine, ..), resolve request params
    state = request.app.state
    temperature = body.temperature if body.temperature is not None else args.temperature
    top_k = body.top_k if body.top_k is not None else args.top_k
    max_tokens = body.max_output_tokens if body.max_output_tokens is not None else args.max_tokens
    seed = body.seed if body.seed is not None else args.seed
    top_k = None if top_k == 0 else top_k  # convert 0 to None to indicate no top_k filtering
    if top_k is not None and top_k > state.tokenizer.n_vocab:
        raise HTTPException(status_code=400, detail=f"top_k cannot be greater than the vocabulary size ({state.tokenizer.n_vocab})")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

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

    created_at = int(time.time())
    model_name = "nanochat"
    if body.stream:
        # OpenAI-compatible stream mode implementation
        sse_chunks_generator = stream_responses_sse_chunks(state, conversation_tokens, response_id, created_at, model_name, temperature, top_k, max_tokens, seed)
        return ChatStreamingResponse(sse_chunks_generator)

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
                seed=seed,
            )
        # Decode the generated tokens into a structured message dictionary
        generated_tokens = new_token_rows[0]    # num_samples=1, so index 0 is our generated token sequence
        stop_reason = finish_reasons[0]
        decoded_message_dict = state.convo_renderer.decode_single_message(
            [state.convo_renderer.assistant_start_token] + generated_tokens  # prepend assistant_start to form full message
        )
        output = message_to_responses_output(decoded_message_dict)
        logger.info(
            "Response OK id=%s stream=False prompt_tokens=%d generated_tokens=%d finish_reason=%s elapsed_s=%.2f seed=%s",
            response_id, len(conversation_tokens), len(new_token_rows[0]), stop_reason, time.perf_counter() - start_time, seed
        )

        # Package and return
        result = make_response(
            response_id=response_id,
            created_at=created_at,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            output=output,
            status="incomplete" if stop_reason == "length" else "completed",
            num_input_tokens=len(conversation_tokens),
            num_output_tokens=len(generated_tokens),
        )
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
