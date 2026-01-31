import torch
import datasets
import pickle

# TODO; support rank/world size
# TODO: validation split (last shard of data)
# TODO; random access somehow? index document lengths?
class DataLoader:
    def __init__(self, batch_size, block_size, hf_path, hf_name, hf_split, tokenizer, group_size, rank, world_size):
        self.batch_size = batch_size
        self.block_size = block_size
        # Dataset
        self.dataset = datasets.load_dataset(hf_path, name=hf_name, split=hf_split)
        # Match nanochat repackage_data_reference.py seed
        self.dataset = self.dataset.shuffle(seed=42)

        # Tokenizer
        self.tokenizer = tokenizer
        self.bos_token = self.tokenizer.encode_single_token('<|bos|>')
        self.token_buffer = []

        # Distributed
        self.idx = 0
        self.group_size = group_size
        self.rank = rank
        self.world_size = world_size

    def map_idx_to_pos(self, idx):
        idx_in_group = idx % self.group_size
        shard_num = idx // self.group_size
        dataset_pos = shard_num * (self.group_size * self.world_size) + self.rank * self.group_size + idx_in_group
        return dataset_pos

    def get_batch(self):
        need_tokens = self.batch_size * self.block_size + 1
        while len(self.token_buffer) < need_tokens:
            dataset_pos = self.map_idx_to_pos(self.idx)
            example = self.dataset[dataset_pos]
            prompt = example['text']
            self.token_buffer += [self.bos_token] + self.tokenizer.encode_ordinary(prompt)
            self.idx += 1
            if dataset_pos >= len(self.dataset):
                # Loop without shuffling for simplicity
                self.idx = 0
        # Consume tokens
        tokens = self.token_buffer[:need_tokens]
        self.token_buffer = self.token_buffer[need_tokens:]

        x = torch.tensor([tokens[:-1]], dtype=torch.long)  # B=1,T
        y = torch.tensor([tokens[1:]], dtype=torch.long)   # B=1,T
        x = x.view(self.batch_size, self.block_size)
        y = y.view(self.batch_size, self.block_size)
        return x, y
