import os
import json
import torch

class DataLoader:
    def __init__(self, dataset, first_shard, last_shard, batch_size, block_size, tokenizer, group_size, rank, world_size):
        self.batch_size = batch_size
        self.block_size = block_size

        # Dataset
        self.dataset = dataset
        self.first_shard = first_shard
        self.last_shard = last_shard

        # Tokenizer
        self.tokenizer = tokenizer
        self.bos_token = self.tokenizer.encode_single_token('<|bos|>')
        self.token_buffer = []

        # Distributed
        self.group_size = group_size
        self.rank = rank
        self.world_size = world_size

        # Read Shard Map
        with open(os.path.dirname(__file__) + "/../data/rowgroup_index.json", "r") as f:
            self.shards = json.load(f)
        assert isinstance(self.shards, list)
        assert all(isinstance(s["num_row_groups"], int) and isinstance(s["start_idx"], int) for s in self.shards)

        # Create Cursor
        self.shard_idx = self.first_shard
        self.group_idx = self.rank
        self.idx_in_group = 0

    def reset(self):
        """Called to reset eval dataloader."""
        self.shard_idx = self.first_shard
        self.group_idx = self.rank
        self.idx_in_group = 0
        self.token_buffer = []

    def step_cursor(self):
        self.idx_in_group += 1
        if self.idx_in_group >= self.group_size:
            self.idx_in_group = 0
            self.group_idx += self.world_size
            if self.group_idx >= self.shards[self.shard_idx]["num_row_groups"]:
                self.group_idx = self.rank
                self.shard_idx += 1
                if self.shard_idx > self.last_shard:
                    self.shard_idx = self.first_shard

    def map_cursor_to_pos(self):
        shard_offset = self.shards[self.shard_idx]["start_idx"]
        group_offset = self.group_idx * self.group_size
        return shard_offset + group_offset + self.idx_in_group

    def get_batch(self):
        need_tokens = self.batch_size * self.block_size + 1
        while len(self.token_buffer) < need_tokens:
            dataset_pos = self.map_cursor_to_pos()
            example = self.dataset[dataset_pos]
            prompt = example['text']
            self.token_buffer += [self.bos_token] + self.tokenizer.encode_ordinary(prompt)
            self.step_cursor()
        # Consume tokens
        tokens = self.token_buffer[:need_tokens]
        self.token_buffer = self.token_buffer[need_tokens:]

        x = torch.tensor([tokens[:-1]], dtype=torch.long)  # B=1,T
        y = torch.tensor([tokens[1:]], dtype=torch.long)   # B=1,T
        x = x.view(self.batch_size, self.block_size)
        y = y.view(self.batch_size, self.block_size)
        return x, y
