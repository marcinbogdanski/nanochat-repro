from typing import Literal
from enum import Enum, auto
from dataclasses import dataclass
from nanorepro.common import UTF8Buffer

class ConversationRenderer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_token = self.tokenizer.encode_single_token('<|bos|>')
        self.user_start_token = self.tokenizer.encode_single_token('<|user_start|>')
        self.user_end_token = self.tokenizer.encode_single_token('<|user_end|>')
        self.assistant_start_token = self.tokenizer.encode_single_token('<|assistant_start|>')
        self.assistant_end_token = self.tokenizer.encode_single_token('<|assistant_end|>')
        self.python_start_token = self.tokenizer.encode_single_token('<|python_start|>')
        self.python_end_token = self.tokenizer.encode_single_token('<|python_end|>')
        self.output_start_token = self.tokenizer.encode_single_token('<|output_start|>')
        self.output_end_token = self.tokenizer.encode_single_token('<|output_end|>')

    def render_conversation(self, messages):
        """Render a structured conversation into token list, inserting special tokens as needed (<|user_start|> etc.)

        Example:
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
            token_list, mask_list = renderer.render_conversation(messages)
        """
        if messages[0]['role'] == 'system' and messages[1]['role'] == 'user':
            merged_content = messages[0]['content'] + "\n\n" + messages[1]['content']
            messages = [{'role': 'user', 'content': merged_content}] + messages[2:]

        for i, msg in enumerate(messages):
            if i % 2 == 0 and msg['role'] != 'user':
                raise ValueError(f"Expected user message at index {i}, got {msg['role']}")
            if i % 2 == 1 and msg['role'] != 'assistant':
                raise ValueError(f"Expected assistant message at index {i}, got {msg['role']}")

        token_list = []
        mask_list = []
        def append_tokens_and_mask(new_tokens, mask_value):
            token_list.extend(new_tokens)
            mask_list.extend([mask_value] * len(new_tokens))

        # mask:
        # 0 - masked, user/tool-output, don't predict; 1 - unmasked, assistant/tool-request, predict
        append_tokens_and_mask([self.bos_token], 0)
        for msg in messages:
            if msg['role'] == 'user':
                new_tokens = self.tokenizer.encode_ordinary(msg['content'])
                append_tokens_and_mask([self.user_start_token], 0)
                append_tokens_and_mask(new_tokens, 0)
                append_tokens_and_mask([self.user_end_token], 0)
            if msg['role'] == 'assistant':
                append_tokens_and_mask([self.assistant_start_token], 0)

                if isinstance(msg['content'], str):
                    new_tokens = self.tokenizer.encode_ordinary(msg['content'])    
                    append_tokens_and_mask(new_tokens, 1)

                elif isinstance(msg['content'], list):
                    for part in msg['content']:
                        new_tokens = self.tokenizer.encode_ordinary(part['text'])
                        if part['type'] == 'text':
                            append_tokens_and_mask(new_tokens, 1)
                        elif part['type'] == 'python':
                            append_tokens_and_mask([self.python_start_token], 1)
                            append_tokens_and_mask(new_tokens, 1)
                            append_tokens_and_mask([self.python_end_token], 1)
                        elif part['type'] == 'python_output':
                            append_tokens_and_mask([self.output_start_token], 0)
                            append_tokens_and_mask(new_tokens, 0)
                            append_tokens_and_mask([self.output_end_token], 0)

                append_tokens_and_mask([self.assistant_end_token], 1)

        return token_list, mask_list


    def decode_single_message(self, token_list):
        """Decode token list representing a _single_ message into structured message parts."""

        # Decode into intermediate event list
        message_decoder = MessageDecoder(self.tokenizer)
        events = []
        for token in token_list:
            events.extend(message_decoder.feed(token))
        events.extend(message_decoder.end_message_stream())

        # Decode intermediate event list into structured messages
        message = None
        for event in events:
            if event.type == 'role_start':
                message = {
                    'role': event.value,
                    'status': 'incomplete',
                    'content': []
                }
            elif event.type == 'role_end':
                message['status'] = event.status
            elif event.type == 'part_start':
                message['content'].append({
                    'type': event.value,
                    'text': '',
                    'status': 'incomplete',
                })
            elif event.type == 'part_end':
                message['content'][-1]['status'] = event.status
            elif event.type == 'delta':
                message['content'][-1]['text'] += event.value

        # Guard empty case
        if message is None:
            raise ValueError("Failed to decode message: no message found")
        return message


class DecoderState(Enum):
    """Message decoder state machine.
    
    Expected transition chains:
    STREAM_INIT -> USER_READY -> USER_TEXT -> STREAM_FINISHED
    STREAM_INIT -> ASSISTANT_READY -> ASSISTANT_TEXT -> STREAM_FINISHED
    STREAM_INIT -> ASSISTANT_READY -> ASSISTANT_TEXT -> ASSISTANT_PYTHON_INPUT -> ASSISTANT_PYTHON_AWAITING_OUTPUT -> ASSISTANT_PYTHON_OUTPUT -> ASSISTANT_READY -> ... -> STREAM_FINISHED
    """
    STREAM_INIT = auto()
    USER_READY = auto()
    USER_TEXT = auto()
    ASSISTANT_READY = auto()
    ASSISTANT_TEXT = auto()
    ASSISTANT_PYTHON_INPUT = auto()
    ASSISTANT_PYTHON_AWAITING_OUTPUT = auto()
    ASSISTANT_PYTHON_OUTPUT = auto()
    STREAM_FINISHED = auto()

@dataclass
class MessageStreamEvent:
    type: Literal["role_start", "role_end", "part_start", "part_end", "delta"]
    value: str
    status: Literal["completed", "incomplete"] | None = None   # status only applicable to selected events    

class MessageDecoder:
    """Decodes a stream of tokens representing _single_ user/assistant message into structured parts.
    
    To enable OpenAI-compatible responses API, we need to break down messages into structured parts: user_text, assistant_text, python_code, python_output.
    In batch mode (when we have the whole token sequence), this is simple: scan through the tokens, identify parts and decode.
    In streaming mode, we need to track the current role (user/assistant) and part (text, python, python_output) and emit transitions accordingly.
    The consumer can then easily consume our structured events and reconstruct user/assistant text/python/python_output.

    This is useful when streaming to e.g. a web client:
    - model emits tokens
    - MessageDecoder decodes tokens into our structured events (this function)
    - web server incrementally renders OpenAI-compatible responses with SSE
    - client decodes SSE and assembles the final message, with tagged user/assistant and text/python/python_output parts

    Incomplete parts are supported, which allows web UI to flag incomplete generation to the user (max_tokens reached)

    Example 1:
    <|user_start|>Hello<|user_end|>

    Example 1 output:
    [
        MessageStreamEvent('role_start', 'user'),
        MessageStreamEvent('part_start', 'text'),
        MessageStreamEvent('delta', 'Hello'),                            # <- actual text
        MessageStreamEvent('part_end', 'text', status='completed'),
        MessageStreamEvent('role_end', 'user', status='completed')
    ]

    Example 2:
    <|assistant_start|>Let's check: 2+2=<|python_start|>2+2<|python_end|><|output_start|>4<|output_end|>4.<|assistant_end|>

    Example 2 output:
    [
        MessageStreamEvent('role_start', 'assistant'),
        MessageStreamEvent('part_start', 'text'),                          -+
        MessageStreamEvent('delta', "Let's"),                               |
        MessageStreamEvent('delta', " check:"),                             |
        MessageStreamEvent('delta', " 2"),                                  |
        MessageStreamEvent('delta', "+"),                                   |-  assistant_text
        MessageStreamEvent('delta', "2"),                                   |
        MessageStreamEvent('delta', "="),                                   |
        MessageStreamEvent('part_end', 'text', status='completed'),        -+
        MessageStreamEvent('part_start', 'python'),                            -+
        MessageStreamEvent('delta', '2'),                                       |
        MessageStreamEvent('delta', '+'),                                       |- python_code
        MessageStreamEvent('delta', '2'),                                       |
        MessageStreamEvent('part_end', 'python', status='completed'),          -+
        MessageStreamEvent('part_start', 'python_output'),                        -+
        MessageStreamEvent('delta', '4'),                                          |- python_output
        MessageStreamEvent('part_end', 'python_output', status='completed'),      -+
        MessageStreamEvent('part_start', 'text'),                            -+
        MessageStreamEvent('delta', "4"),                                     |
        MessageStreamEvent('delta', "."),                                     |- assistant_text again
        MessageStreamEvent('part_end', 'text', status='completed'),          -+
        MessageStreamEvent('role_end', 'assistant', status='completed')
    ]
    
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_token = self.tokenizer.encode_single_token('<|bos|>')
        self.user_start_token = self.tokenizer.encode_single_token('<|user_start|>')
        self.user_end_token = self.tokenizer.encode_single_token('<|user_end|>')
        self.assistant_start_token = self.tokenizer.encode_single_token('<|assistant_start|>')
        self.assistant_end_token = self.tokenizer.encode_single_token('<|assistant_end|>')
        self.python_start_token = self.tokenizer.encode_single_token('<|python_start|>')
        self.python_end_token = self.tokenizer.encode_single_token('<|python_end|>')
        self.python_output_start_token = self.tokenizer.encode_single_token('<|output_start|>')
        self.python_output_end_token = self.tokenizer.encode_single_token('<|output_end|>')

        self.special_tokens = [
            self.bos_token,
            self.user_start_token,
            self.user_end_token,
            self.assistant_start_token,
            self.assistant_end_token,
            self.python_start_token,
            self.python_end_token,
            self.python_output_start_token,
            self.python_output_end_token,
        ]

        self.state = DecoderState.STREAM_INIT
        self.utf8_buffer = UTF8Buffer()

    def end_message_stream(self, status='incomplete'):
        """End the current message stream and return any remaining events.
        
        This can be called at any point to properly close the current message stream.
        Example use case is when model generation runs out of max_tokens, and does not emit final <|assistant_end|> token.
        """
        if self.state == DecoderState.STREAM_INIT:
            self.state = DecoderState.STREAM_FINISHED
            return []  # we didn't start anything, so nothing to close

        elif self.state == DecoderState.USER_READY:
            self.state = DecoderState.STREAM_FINISHED
            return [MessageStreamEvent('role_end', 'user', status)]

        elif self.state == DecoderState.USER_TEXT:
            self.state = DecoderState.STREAM_FINISHED
            text = self.utf8_buffer.flush()
            if text:
                return [MessageStreamEvent('delta', text), MessageStreamEvent('part_end', 'text', status), MessageStreamEvent('role_end', 'user', status)]
            return [MessageStreamEvent('part_end', 'text', status), MessageStreamEvent('role_end', 'user', status)]

        elif self.state == DecoderState.ASSISTANT_READY:
            self.state = DecoderState.STREAM_FINISHED
            return [MessageStreamEvent('role_end', 'assistant', status)]

        elif self.state == DecoderState.ASSISTANT_TEXT:
            self.state = DecoderState.STREAM_FINISHED
            text = self.utf8_buffer.flush()
            if text:
                return [MessageStreamEvent('delta', text), MessageStreamEvent('part_end', 'text', status), MessageStreamEvent('role_end', 'assistant', status)]
            return [MessageStreamEvent('part_end', 'text', status), MessageStreamEvent('role_end', 'assistant', status)]

        elif self.state == DecoderState.ASSISTANT_PYTHON_INPUT:
            self.state = DecoderState.STREAM_FINISHED
            text = self.utf8_buffer.flush()
            if text:
                return [MessageStreamEvent('delta', text), MessageStreamEvent('part_end', 'python', 'incomplete'), MessageStreamEvent('role_end', 'assistant', 'incomplete')]
            return [MessageStreamEvent('part_end', 'python', 'incomplete'), MessageStreamEvent('role_end', 'assistant', 'incomplete')]

        elif self.state == DecoderState.ASSISTANT_PYTHON_AWAITING_OUTPUT:
            self.state = DecoderState.STREAM_FINISHED
            return [MessageStreamEvent('role_end', 'assistant', 'incomplete')]

        elif self.state == DecoderState.ASSISTANT_PYTHON_OUTPUT:
            self.state = DecoderState.STREAM_FINISHED
            text = self.utf8_buffer.flush()
            if text:
                return [MessageStreamEvent('delta', text), MessageStreamEvent('part_end', 'python_output', 'incomplete'), MessageStreamEvent('role_end', 'assistant', 'incomplete')]
            return [MessageStreamEvent('part_end', 'python_output', 'incomplete'), MessageStreamEvent('role_end', 'assistant', 'incomplete')]

        elif self.state == DecoderState.STREAM_FINISHED:
            return []
        raise ValueError(f"Unexpected state: {self.state}")

    def feed(self, token) -> list[MessageStreamEvent]:
        """Feed one message token at a time and return valid event stream.
        
        This method tracks and enforces the state transitions or raises an error.
        """

        # Start of the message should always be either <|user_start|> or <|assistant_start|>
        if self.state == DecoderState.STREAM_INIT:
            if token == self.user_start_token:
                self.state = DecoderState.USER_READY
                return [MessageStreamEvent('role_start', 'user')]
            elif token == self.assistant_start_token:
                self.state = DecoderState.ASSISTANT_READY
                return [MessageStreamEvent('role_start', 'assistant')]
            else:
                raise ValueError(f"Unexpected token in STREAM_INIT state: {self.tokenizer.decode([token])}")

        elif self.state == DecoderState.USER_READY:
            if token in (self.user_end_token, self.bos_token):
                return self.end_message_stream(status='completed')
            elif token in self.special_tokens:
                self.state = DecoderState.STREAM_FINISHED
                raise ValueError(f"Unexpected special token in USER_READY state: {self.tokenizer.decode([token])}")
            else:
                self.state = DecoderState.USER_TEXT
                token_bytes = self.tokenizer.decode_bytes([token])
                text = self.utf8_buffer.decode(token_bytes)
                if text:
                    return [MessageStreamEvent('part_start', 'text'), MessageStreamEvent('delta', text)]
                return [MessageStreamEvent('part_start', 'text')]

        elif self.state == DecoderState.USER_TEXT:
            if token in (self.user_end_token, self.bos_token):
                return self.end_message_stream(status='completed')
            elif token in self.special_tokens:
                self.state = DecoderState.STREAM_FINISHED
                raise ValueError(f"Unexpected special token in USER_TEXT state: {self.tokenizer.decode([token])}")
            else:
                token_bytes = self.tokenizer.decode_bytes([token])
                text = self.utf8_buffer.decode(token_bytes)
                if text:
                    return [MessageStreamEvent('delta', text)]
                return []

        elif self.state == DecoderState.ASSISTANT_READY:
            if token in (self.assistant_end_token, self.bos_token):
                return self.end_message_stream(status='completed')
            elif token == self.python_start_token:
                self.state = DecoderState.ASSISTANT_PYTHON_INPUT
                return [MessageStreamEvent('part_start', 'python')]
            elif token in self.special_tokens:
                self.state = DecoderState.STREAM_FINISHED
                raise ValueError(f"Unexpected special token in ASSISTANT_READY state: {self.tokenizer.decode([token])}")
            else:
                self.state = DecoderState.ASSISTANT_TEXT
                token_bytes = self.tokenizer.decode_bytes([token])
                text = self.utf8_buffer.decode(token_bytes)
                if text:
                    return [MessageStreamEvent('part_start', 'text'), MessageStreamEvent('delta', text)]
                return [MessageStreamEvent('part_start', 'text')]

        elif self.state == DecoderState.ASSISTANT_TEXT:
            if token in (self.assistant_end_token, self.bos_token):
                return self.end_message_stream(status='completed')
            elif token == self.python_start_token:
                self.state = DecoderState.ASSISTANT_PYTHON_INPUT
                text = self.utf8_buffer.flush()
                if text:
                    return [MessageStreamEvent('delta', text), MessageStreamEvent('part_end', 'text', 'completed'), MessageStreamEvent('part_start', 'python')]
                return [MessageStreamEvent('part_end', 'text', 'completed'), MessageStreamEvent('part_start', 'python')]
            elif token in self.special_tokens:
                self.state = DecoderState.STREAM_FINISHED
                raise ValueError(f"Unexpected special token in ASSISTANT_TEXT state: {self.tokenizer.decode([token])}")
            else:
                token_bytes = self.tokenizer.decode_bytes([token])
                text = self.utf8_buffer.decode(token_bytes)
                if text:
                    return [MessageStreamEvent('delta', text)]
                return []

        elif self.state == DecoderState.ASSISTANT_PYTHON_INPUT:
            if token == self.bos_token:
                return self.end_message_stream(status='incomplete')
            elif token == self.python_end_token:
                self.state = DecoderState.ASSISTANT_PYTHON_AWAITING_OUTPUT
                text = self.utf8_buffer.flush()
                if text:
                    return [MessageStreamEvent('delta', text), MessageStreamEvent('part_end', 'python', 'completed')]
                return [MessageStreamEvent('part_end', 'python', 'completed')]
            elif token in self.special_tokens:
                self.state = DecoderState.STREAM_FINISHED
                raise ValueError(f"Unexpected special token in ASSISTANT_PYTHON_INPUT state: {self.tokenizer.decode([token])}")
            else:
                token_bytes = self.tokenizer.decode_bytes([token])
                text = self.utf8_buffer.decode(token_bytes)
                if text:
                    return [MessageStreamEvent('delta', text)]
                return []

        elif self.state == DecoderState.ASSISTANT_PYTHON_AWAITING_OUTPUT:
            if token == self.bos_token:
                return self.end_message_stream(status='incomplete')
            elif token == self.python_output_start_token:
                self.state = DecoderState.ASSISTANT_PYTHON_OUTPUT
                return [MessageStreamEvent('part_start', 'python_output')]
            else:
                self.state = DecoderState.STREAM_FINISHED
                raise ValueError(f"Unexpected special token in ASSISTANT_PYTHON_AWAITING_OUTPUT state: {self.tokenizer.decode([token])}")

        elif self.state == DecoderState.ASSISTANT_PYTHON_OUTPUT:
            if token == self.bos_token:
                return self.end_message_stream(status='incomplete')
            elif token == self.python_output_end_token:
                self.state = DecoderState.ASSISTANT_READY
                text = self.utf8_buffer.flush()
                if text:
                    return [MessageStreamEvent('delta', text), MessageStreamEvent('part_end', 'python_output', 'completed')]
                return [MessageStreamEvent('part_end', 'python_output', 'completed')]
            elif token in self.special_tokens:
                self.state = DecoderState.STREAM_FINISHED
                raise ValueError(f"Unexpected special token in ASSISTANT_PYTHON_OUTPUT state: {self.tokenizer.decode([token])}")
            else:
                token_bytes = self.tokenizer.decode_bytes([token])
                text = self.utf8_buffer.decode(token_bytes)
                if text:
                    return [MessageStreamEvent('delta', text)]
                return []

        elif self.state == DecoderState.STREAM_FINISHED:
            raise ValueError(f"Message stream has already finished, start new message stream. Unexpected token: {self.tokenizer.decode([token])}")

