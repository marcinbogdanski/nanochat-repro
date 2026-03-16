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
        self.document_buffer = []

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
        self.document_buffer = []

    def _step_cursor(self):
        self.idx_in_group += 1
        if self.idx_in_group >= self.group_size:
            self.idx_in_group = 0
            self.group_idx += self.world_size
            if self.group_idx >= self.shards[self.shard_idx]["num_row_groups"]:
                self.group_idx = self.rank
                self.shard_idx += 1
                if self.shard_idx > self.last_shard:
                    self.shard_idx = self.first_shard

    def _map_cursor_to_pos(self):
        shard_offset = self.shards[self.shard_idx]["start_idx"]
        group_offset = self.group_idx * self.group_size
        return shard_offset + group_offset + self.idx_in_group

    def _get_next_document(self):
        dataset_pos = self._map_cursor_to_pos()
        example = self.dataset[dataset_pos]
        prompt = example['text']
        self._step_cursor()
        return [self.bos_token] + self.tokenizer.encode_ordinary(prompt)        

    def get_batch(self):
        need_tokens = self.batch_size * self.block_size + 1
        while len(self.token_buffer) < need_tokens:
            self.token_buffer += self._get_next_document()
        # Consume tokens
        tokens = self.token_buffer[:need_tokens]
        self.token_buffer = self.token_buffer[self.batch_size * self.block_size:]

        x = torch.tensor([tokens[:-1]], dtype=torch.long)  # B=1,T
        y = torch.tensor([tokens[1:]], dtype=torch.long)   # B=1,T
        x = x.view(self.batch_size, self.block_size)
        y = y.view(self.batch_size, self.block_size)
        return x, y
    
    def _fill_doc_buffer(self):
        while len(self.document_buffer) < 1000:
            for _ in range(128):  # match Nanochat behavior
                doc_tokens = self._get_next_document()
                self.document_buffer.append(doc_tokens)

    def get_batch_bos(self):
        need_row_tokens = self.block_size + 1

        batch_rows = []
        for bi in range(self.batch_size):
            row_tokens = []
            while len(row_tokens) < need_row_tokens:
                self._fill_doc_buffer()
                # Find longest doc that fits
                longest_doc_idx = None
                longest_doc_len = 0
                num_tokens_to_fill = need_row_tokens - len(row_tokens)
                for di in range(len(self.document_buffer)):
                    candidate_doc = self.document_buffer[di]
                    if len(candidate_doc) <= num_tokens_to_fill and len(candidate_doc) > longest_doc_len:
                        longest_doc_idx = di
                        longest_doc_len = len(candidate_doc)
                # Extend row
                if longest_doc_idx is not None:
                    longest_doc_that_fits = self.document_buffer.pop(longest_doc_idx)
                    row_tokens.extend(longest_doc_that_fits)
                else:
                    shortest_idx = min(range(len(self.document_buffer)), key=lambda i: len(self.document_buffer[i]))
                    doc_to_trim = self.document_buffer.pop(shortest_idx)
                    row_tokens.extend(doc_to_trim[:num_tokens_to_fill])
            batch_rows.append(row_tokens)

        batch_tensor = torch.tensor(batch_rows, dtype=torch.long)
        x = batch_tensor[:, :-1]
        y = batch_tensor[:, 1:]

        return x, y


