# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Test if our chat API is working with OpenAI SDK

Run with:
uv run dev/tests/test_chat_api.py
"""

# keep this import out of pyproject.toml as it is not needed anywhere else
import openai  # pyright: ignore[reportMissingImports]

client = openai.OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="unused",
)

def expect_status(status, **kwargs):
    try:
        client.chat.completions.create(**kwargs)
    except openai.APIStatusError as error:
        assert error.status_code == status
    else:
        raise AssertionError(f"Expected HTTP {status}")

def test_models_list():
    # List Models
    model_list = client.models.list()
    assert len(model_list.data) == 1

    model = model_list.data[0]
    print(model)
    assert model.id == "nanochat"
    assert model.owned_by == "nanochat-repro"


def test_chat_completion():
    completion = client.chat.completions.create(
        model="nanochat",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    print(completion)
    assert completion.object == "chat.completion"
    assert completion.model == "nanochat"
    assert len(completion.choices) == 1

    choice = completion.choices[0]
    assert choice.message.role == "assistant"
    assert isinstance(choice.message.content, str)
    assert choice.message.content.strip()
    assert choice.finish_reason in {"stop", "length"}
    assert "<|bos|>" not in choice.message.content  # confirm stop tokens don't leak to client
    assert "<|assistant_end|>" not in choice.message.content

def test_chat_completion_stream():
    with client.chat.completions.create(
        model="nanochat",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ) as stream:
        sse_chunks = list(stream)

    assert sse_chunks
    for chunk in sse_chunks:
        assert chunk.object == "chat.completion.chunk"
        assert chunk.model == "nanochat"
        assert chunk.id == sse_chunks[0].id
        assert chunk.created == sse_chunks[0].created
        assert len(chunk.choices) == 1
        assert chunk.choices[0].index == 0

    choices = [chunk.choices[0] for chunk in sse_chunks]  # each 'choice' is SSE item
    assert choices[0].delta.role == "assistant"
    assert all(choice.finish_reason is None for choice in choices[:-1])
    assert choices[-1].finish_reason in {"stop", "length"}

    response_text = "".join(choice.delta.content or "" for choice in choices)
    print(response_text)
    assert response_text.strip()
    assert "<|bos|>" not in response_text
    assert "<|assistant_end|>" not in response_text

def test_stream_matches_non_stream():
    messages=[{"role": "user", "content": "Hello!"}]

    completion = client.chat.completions.create(
        model="nanochat",
        messages=messages,
    )
    non_stream_response = completion.choices[0].message.content

    with client.chat.completions.create(
            model="nanochat",
            messages=[{"role": "user", "content": "Hello!"}],
            stream=True,
        ) as stream:
            choices = [chunk.choices[0] for chunk in stream]
    stream_response = "".join(choice.delta.content or "" for choice in choices)
    assert non_stream_response == stream_response
    assert completion.choices[0].finish_reason == choices[-1].finish_reason

def test_disconnect_releases_lock():

    # First call: drop mid-stream
    my_client = client.with_options(timeout=60.0, max_retries=0)
    with my_client.chat.completions.create(
        model="nanochat",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ) as stream:
        for sse_chunk in stream:
            if sse_chunk.choices[0].delta.content:
                break  # breaking drops the stream before consuming whole response
        else:
            assert False  # stream produced no content?

    # Second call: if server-side worker-lock did not release, this will hang
    completion = client.chat.completions.create(
        model="nanochat",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    assert completion.choices[0].message.content
    assert completion.choices[0].finish_reason in {"stop", "length"}


def test_invalid_requests():
    valid_messages = [{"role": "user", "content": "Hello!"}]

    expect_status(404, model="unknown", messages=valid_messages)
    expect_status(404, model="unknown", messages=valid_messages, stream=True)
    expect_status(422, model="nanochat", messages=[])
    expect_status(422, model="nanochat", messages=[{"role": "user", "content": ""}])
    expect_status(422, model="nanochat", messages=[{"role": "system", "content": "Hello!"}])

if __name__ == "__main__":
    test_models_list()
    test_chat_completion()
    test_chat_completion_stream()
    test_stream_matches_non_stream()
    test_invalid_requests()
    test_disconnect_releases_lock()
    print("All tests passed!")
