import time
import uuid
from pydantic import BaseModel, Field
from typing import Literal
from fastapi import FastAPI, HTTPException
import uvicorn

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False  # for now

app = FastAPI()

# Test with:
# curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"nanochat","messages":[{"role":"user","content":"Hello!"}]}'
@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    """OpenAI-compatible endpoint to create chat completions"""
    if request.model != "nanochat":
        raise HTTPException(status_code=404, detail="Model not supported")
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet")

    result = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "nanochat",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello, world!"
            },
            "finish_reason": "stop",
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
