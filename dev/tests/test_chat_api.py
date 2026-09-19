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

def test_invalid_requests():
    valid_messages = [{"role": "user", "content": "Hello!"}]

    expect_status(404, model="unknown", messages=valid_messages)
    expect_status(400, model="nanochat", messages=valid_messages, stream=True)
    expect_status(422, model="nanochat", messages=[])
    expect_status(422, model="nanochat", messages=[{"role": "user", "content": ""}])
    expect_status(422, model="nanochat", messages=[{"role": "system", "content": "Hello!"}])

if __name__ == "__main__":
    test_models_list()
    test_chat_completion()
    test_invalid_requests()
    print("All tests passed!")
