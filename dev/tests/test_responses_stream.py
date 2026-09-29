"""Test responses stream, including handling of text, python, and python_output parts.

The stream_responses_sse_chunks() from scripts.chat_web is tested here.

Run with (needs trained tokenizer):
uv run python -m dev.tests.test_responses_stream
"""
import os
import json
import pickle
from contextlib import nullcontext
from types import SimpleNamespace

from nanorepro.common import get_base_path
from nanorepro.tokenizer import ConversationRenderer
from scripts.chat_web import message_to_responses_output, stream_responses_sse_chunks
BASE_DIR = get_base_path()

def test_stream_responses_sse_chunks(tokenizer, convo_renderer, generated_text, finish_reason, expected_tool_status):
    tokens = tokenizer.encode(generated_text, allowed_special="all")  # allow_special="all" to automatically tokenizer <|user_start|> and so on.

    # Dummy engine.simulate_stream()
    def generate_stream(*args, **kwargs):
        for index, token in enumerate(tokens):
            yield [token], [finish_reason if index == len(tokens)-1 else None]

    # Dummy FastAPI state
    state = SimpleNamespace(
        tokenizer=tokenizer,
        convo_renderer=convo_renderer,
        lock=nullcontext(),
        engine=SimpleNamespace(generate_stream=generate_stream)
    )

    # Simulate streaming SSE events
    prompt = [convo_renderer.assistant_start_token]  # we are testing the assistant's response stream, prompt is irrelevant here
    chunks = stream_responses_sse_chunks(
        state=state,
        conversation_tokens=prompt,
        response_id="1234",
        created_at=0,
        model_name="nanochat",
        temperature=1.0,
        top_k=50,
        max_tokens=len(tokens),
        seed=42,
    )

    # Decode SSE chunks:
    #   ---
    #   event: response.output_text.delta
    #   data: {"delta": "Hello"}
    #   ---
    events = []   # [..., {"delta": "Hello"}, ...]
    for chunk in chunks:
        event_line, data_line = chunk.strip().split("\n")
        event = json.loads(data_line.removeprefix("data: "))
        assert event_line == "event: " + event["type"]
        events.append(event)

    # Check first/last events
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert events[0]["type"] == "response.created"
    assert events[0]["response"]["output"] == []     # stream just started, non yet populated
    expected_status = "incomplete" if finish_reason == "length" else "completed"
    assert events[-1]["type"] == "response." + expected_status
    response = events[-1]["response"]                # final event with full response
    done_items = [event["item"] for event in events if event["type"] == "response.output_item.done"]
    assert done_items == response["output"]          # confirm all output items were marked as done (i.e. SSE closed, this is not the same as status=completed/incomplete)

    # Compare streamed items with expected batch output
    decoded_msg_dict = convo_renderer.decode_single_message([convo_renderer.assistant_start_token] + tokens)
    responses_expected_output = message_to_responses_output(decoded_msg_dict)
    items_stream = [{k: v for k, v in item.items() if k != "id"} for item in done_items]  # list of dict
    items_expected = [{k: v for k, v in item.items() if k != "id"} for item in responses_expected_output]
    assert items_stream == items_expected

    # Check tool item
    tool_item = [item for item in done_items if item["type"] == "code_interpreter_call"][0]   # all test examples have exactly one tool call
    assert tool_item["status"] == expected_tool_status
    code = "".join(event["delta"] for event in events if event["type"] == "response.code_interpreter_call_code.delta")
    assert code == tool_item["code"]

    # Check if correct tool completion event was emitted
    event_types = [event["type"] for event in events]
    if expected_tool_status == "completed":
        assert "response.code_interpreter_call.completed" in event_types
    else:
        assert "response.code_interpreter_call.completed" not in event_types

def main():
    # Tokenizer
    tok_base_path = os.path.join(BASE_DIR, "tokenizer")
    tokenizer_path = os.path.join(tok_base_path, "tokenizer.pkl")
    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    renderer = ConversationRenderer(tokenizer)
    code = "Checking: <|python_start|>2+2"
    output = code + "<|python_end|><|output_start|>4"
    test_stream_responses_sse_chunks(tokenizer, renderer, output + "<|output_end|>Done.<|assistant_end|>", "stop", "completed")
    test_stream_responses_sse_chunks(tokenizer, renderer, code, "length", "incomplete")
    test_stream_responses_sse_chunks(tokenizer, renderer, code + "<|python_end|>", "length", "incomplete")
    test_stream_responses_sse_chunks(tokenizer, renderer, output, "length", "incomplete")
    print("Tool streaming tests passed")

if __name__ == "__main__":
    main()
