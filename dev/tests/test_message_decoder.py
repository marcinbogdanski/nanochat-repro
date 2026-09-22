"""Test message decoder in tokenizer.py

Tests in this file depend on particular tokenizer, which is not ideal.

Run with:
uv run python -m dev.tests.test_message_decoder
"""
import os
import pickle

from nanorepro.common import get_base_path
from nanorepro.tokenizer import MessageDecoder, MessageStreamEvent
BASE_DIR = get_base_path()

def decode_message(tokenizer, message_tokens):
    """Helper to decode tokenized message into events"""
    assert isinstance(message_tokens, list)
    assert all(isinstance(token, int) for token in message_tokens)
    
    decoder = MessageDecoder(tokenizer)
    events = []
    for token in message_tokens:
        events.extend(decoder.feed(token))
    events.extend(decoder.end_message_stream())
    return events

def test_user_message(tokenizer):
    """Test decode user message"""
    # Prepare the tokens
    user_start_token = tokenizer.encode_single_token('<|user_start|>')
    user_end_token = tokenizer.encode_single_token('<|user_end|>')
    message_tokens = [user_start_token] + tokenizer.encode('Hello!') + [user_end_token]

    # Test decode
    events = decode_message(tokenizer, message_tokens)
    assert events == [
        MessageStreamEvent('role_start', 'user'),
        MessageStreamEvent('part_start', 'text'),
        MessageStreamEvent('delta', 'Hello'),
        MessageStreamEvent('delta', '!'),
        MessageStreamEvent('part_end', 'text', status='completed'),
        MessageStreamEvent('role_end', 'user', status='completed'),
    ]
    print("test_user_message passed")

def test_assistant_tool_call(tokenizer):
    """Test decode assistant answer with embedded tool call"""

    # Prepare the tokens
    assistant_start_token = tokenizer.encode_single_token('<|assistant_start|>')
    assistant_end_token = tokenizer.encode_single_token('<|assistant_end|>')
    python_start_token = tokenizer.encode_single_token('<|python_start|>')
    python_end_token = tokenizer.encode_single_token('<|python_end|>')
    output_start_token = tokenizer.encode_single_token('<|output_start|>')
    output_end_token = tokenizer.encode_single_token('<|output_end|>')
    message_tokens = \
        [assistant_start_token] + tokenizer.encode('2+2=') + \
        [python_start_token] + tokenizer.encode('2+2') + [python_end_token] + \
        [output_start_token] + tokenizer.encode('4') + [output_end_token] + \
        tokenizer.encode('4') + [assistant_end_token]

    # Test decode
    events = decode_message(tokenizer, message_tokens)
    assert events == [
        MessageStreamEvent('role_start', 'assistant'),
        MessageStreamEvent('part_start', 'text'),
        MessageStreamEvent('delta', '2'),
        MessageStreamEvent('delta', '+'),
        MessageStreamEvent('delta', '2'),
        MessageStreamEvent('delta', '='),
        MessageStreamEvent('part_end', 'text', status='completed'),
        MessageStreamEvent('part_start', 'python'),
        MessageStreamEvent('delta', '2'),
        MessageStreamEvent('delta', '+'),
        MessageStreamEvent('delta', '2'),
        MessageStreamEvent('part_end', 'python', status='completed'),
        MessageStreamEvent('part_start', 'python_output'),
        MessageStreamEvent('delta', '4'),
        MessageStreamEvent('part_end', 'python_output', status='completed'),
        MessageStreamEvent('part_start', 'text'),
        MessageStreamEvent('delta', '4'),
        MessageStreamEvent('part_end', 'text', status='completed'),
        MessageStreamEvent('role_end', 'assistant', status='completed'),
    ]
    print("test_assistant_tool_call passed")

def test_split_utf8(tokenizer):
    """Test if multi-token UTF-8 characters are correctly decoded"""
    # Prepare the tokens
    assistant_start_token = tokenizer.encode_single_token('<|assistant_start|>')
    assistant_end_token = tokenizer.encode_single_token('<|assistant_end|>')
    message_tokens = \
        [assistant_start_token] + tokenizer.encode('😀') + [assistant_end_token]
    assert message_tokens == [32762, 6427, 152, 128, 32763]  # smiling face is 3 tokens

    # Test decode
    events = decode_message(tokenizer, message_tokens)
    assert events == [
        MessageStreamEvent('role_start', 'assistant'),
        MessageStreamEvent('part_start', 'text'),
        MessageStreamEvent('delta', '😀'),
        MessageStreamEvent('part_end', 'text', status='completed'),
        MessageStreamEvent('role_end', 'assistant', status='completed'),
    ]
    print("test_split_utf8 passed")

def test_early_termination(tokenizer):
    """Test message terminated unexpectedly, especially mid UTF-8 character"""

    # Prepare the tokens
    assistant_start_token = tokenizer.encode_single_token('<|assistant_start|>')
    assistant_end_token = tokenizer.encode_single_token('<|assistant_end|>')
    message_tokens = [assistant_start_token] + tokenizer.encode('Hello, 😀')[:-1]  # terminated mid UTF-8 character

    # Test decode
    events = decode_message(tokenizer, message_tokens)
    assert events == [
        MessageStreamEvent('role_start', 'assistant'),
        MessageStreamEvent('part_start', 'text'),
        MessageStreamEvent('delta', 'Hello'),
        MessageStreamEvent('delta', ','),
        MessageStreamEvent('delta', ' '),
        MessageStreamEvent('delta', '�'),  # replacement character for incomplete UTF-8 sequence
        MessageStreamEvent('part_end', 'text', status='incomplete'),
        MessageStreamEvent('role_end', 'assistant', status='incomplete'),
    ]
    print("test_early_termination passed")

def test_early_termination_mid_python_output(tokenizer):
    """Test complete Python input, but generation stops before output ends unexpectedly"""

    # Prepare the tokens
    assistant_start_token = tokenizer.encode_single_token('<|assistant_start|>')
    python_start_token = tokenizer.encode_single_token('<|python_start|>')
    python_end_token = tokenizer.encode_single_token('<|python_end|>')
    output_start_token = tokenizer.encode_single_token('<|output_start|>')
    message_tokens = (
           [assistant_start_token, python_start_token]
           + tokenizer.encode('2+2')
           + [python_end_token, output_start_token]
           + tokenizer.encode('4')       # simulate early termination before <|output_end|> token is emitted
    )

    # Test decode
    events = decode_message(tokenizer, message_tokens)
    assert events == [
        MessageStreamEvent('role_start', 'assistant'),
        MessageStreamEvent('part_start', 'python'),
        MessageStreamEvent('delta', '2'),
        MessageStreamEvent('delta', '+'),
        MessageStreamEvent('delta', '2'),
        MessageStreamEvent('part_end', 'python', status='completed'),
        MessageStreamEvent('part_start', 'python_output'),
        MessageStreamEvent('delta', '4'),
        MessageStreamEvent('part_end', 'python_output', status='incomplete'),
        MessageStreamEvent('role_end', 'assistant', status='incomplete'),
    ]
    print("test_early_termination_mid_python_output passed")
    

def test_invalid_tool_boundaries(tokenizer):
    """Test msg with two sequential <python_start> is rejected"""

    # Prepare message
    assistant_start_token = tokenizer.encode_single_token('<|assistant_start|>')
    python_start_token = tokenizer.encode_single_token('<|python_start|>')
    python_end_token = tokenizer.encode_single_token('<|python_end|>')
    assistant_end_token = tokenizer.encode_single_token('<|assistant_end|>')
    message_tokens = [assistant_start_token, python_start_token, python_start_token,
                      python_end_token, python_end_token, assistant_end_token]
    # Test decode
    try:
        decode_message(tokenizer, message_tokens)
    except ValueError:
        pass  # this is expected
    else:
        assert False  # if decode_message() should have raised
    print("test_invalid_tool_boundaries passed")

def main():
    # Tokenizer
    tok_base_path = os.path.join(BASE_DIR, "tokenizer")
    tokenizer_path = os.path.join(tok_base_path, "tokenizer.pkl")
    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    test_user_message(tokenizer)
    test_assistant_tool_call(tokenizer)
    test_split_utf8(tokenizer)
    test_early_termination(tokenizer)
    test_early_termination_mid_python_output(tokenizer)
    test_invalid_tool_boundaries(tokenizer)
    print("All tests passed.")

if __name__ == '__main__':
    main()
