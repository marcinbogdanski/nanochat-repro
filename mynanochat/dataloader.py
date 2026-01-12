import torch
import datasets
import pickle

# TODO; support rank/world size
# TODO: validation split (last shard of data)
# TODO; random access somehow? index document lengths?
class DataLoader:
    def __init__(self, batch_size, block_size, hf_path, hf_name, hf_split, tokenizer):
        self.batch_size = batch_size
        self.block_size = block_size
        # Dataset
        self.dataset = datasets.load_dataset(hf_path, name=hf_name, split=hf_split)
        # Match nanochat repackage_data_reference.py seed
        self.dataset = self.dataset.shuffle(seed=42)
        self.dataset_pos = 0

        # Tokenizer
        self.tokenizer = tokenizer
        self.bos_token = self.tokenizer.encode_single_token('<|bos|>')
        self.token_buffer = []

    def get_batch(self):
        need_tokens = self.batch_size * self.block_size + 1
        while len(self.token_buffer) < need_tokens:
            example = self.dataset[self.dataset_pos]
            prompt = example['text']
            self.token_buffer += [self.bos_token] + self.tokenizer.encode_ordinary(prompt)
            self.dataset_pos += 1
            if self.dataset_pos >= len(self.dataset):
                # Loop without shuffling for simplicity
                self.dataset_pos = 0
        # Consume tokens
        tokens = self.token_buffer[:need_tokens]
        self.token_buffer = self.token_buffer[need_tokens:]

        x = torch.tensor([tokens[:-1]], dtype=torch.long)  # B=1,T
        y = torch.tensor([tokens[1:]], dtype=torch.long)   # B=1,T
        x = x.view(self.batch_size, self.block_size)
        y = y.view(self.batch_size, self.block_size)
        return x, y
