



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

