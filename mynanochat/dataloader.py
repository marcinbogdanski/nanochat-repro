import torch

# TODO; support rank/world size
# TODO: validation split (last shard of data)
# TODO; random access somehow? index document lengths?
class DataLoader:
    def __init__(self, dataset, start_at, end_at, batch_size, block_size, tokenizer, group_size, rank, world_size):
        self.batch_size = batch_size
        self.block_size = block_size
        
        # Dataset
        self.dataset = dataset

        # Tokenizer
        self.tokenizer = tokenizer
        self.bos_token = self.tokenizer.encode_single_token('<|bos|>')
        self.token_buffer = []

        # Data Range
        self.start_at = 0 if start_at is None else start_at
        self.end_at = len(self.dataset) if end_at is None else end_at

        # Distributed
        self.idx = 0
        self.group_size = group_size
        self.rank = rank
        self.world_size = world_size

    def reset(self):
        self.idx = 0
        self.token_buffer = []

    def map_idx_to_pos(self, idx):
        idx_in_group = idx % self.group_size
        shard_num = idx // self.group_size
        dataset_pos = shard_num * (self.group_size * self.world_size) + self.rank * self.group_size + idx_in_group
        dataset_pos += self.start_at
        return dataset_pos

    def get_batch(self):
        need_tokens = self.batch_size * self.block_size + 1
        while len(self.token_buffer) < need_tokens:
            dataset_pos = self.map_idx_to_pos(self.idx)
            example = self.dataset[dataset_pos]
            prompt = example['text']
            self.token_buffer += [self.bos_token] + self.tokenizer.encode_ordinary(prompt)
            self.idx += 1
            if dataset_pos >= self.end_at - 1:
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
