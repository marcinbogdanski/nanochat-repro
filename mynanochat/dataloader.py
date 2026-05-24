import os
import torch
import pyarrow.parquet as pq
from mynanochat.common import get_base_path
BASE_DIR = get_base_path()

class DataLoader:
    def __init__(self, dataset_or_folderpath, split, batch_size, block_size, tokenizer):

        # Dataset path logic
        if dataset_or_folderpath == "fineweb":
            folderpath = os.path.join(BASE_DIR, "base_data")
        elif dataset_or_folderpath == "climbmix":
            folderpath = os.path.join(BASE_DIR, "base_data_climbmix")
        else:
            folderpath = dataset_or_folderpath

        # Dataset
        assert os.path.isdir(folderpath)
        self.folderpath = folderpath

        shard_files = sorted(fn for fn in os.listdir(folderpath) if fn.endswith('.parquet'))
        shard_indices = []
        for shard_fn in shard_files:
            fn_root, fn_index = shard_fn.replace('.parquet', '').split('_')
            shard_indices.append(int(fn_index))

        self.split = split
        if split == 'train':        
            self.first_shard = shard_indices[0]
            self.last_shard = shard_indices[-2]  # inclusive
            assert shard_indices[:-1] == list(range(self.first_shard, self.first_shard+self.last_shard+1))
        elif split == 'val':
            self.first_shard = shard_indices[-1]
            self.last_shard = shard_indices[-1]
        else:
            raise ValueError("Param 'split' must be one of: 'train', 'val'")

        # Hyperparameters
        self.batch_size = batch_size
        self.block_size = block_size

        # Tokenizer
        self.tokenizer = tokenizer
        self.bos_token = self.tokenizer.encode_single_token('<|bos|>')
        self.token_buffer = []
        self.document_buffer = []

        # Distributed
        self.group_size = 1024  # same as nanochat
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        # Create Cursor
        self.shard_idx = self.first_shard
        self.group_idx = self.rank
        self.idx_in_group = 0

        self.loaded_shard_idx = None
        self.loaded_shard_row_groups = None   # list of list or str

    def reset(self):
        """Called to reset eval dataloader."""
        self.shard_idx = self.first_shard
        self.group_idx = self.rank
        self.idx_in_group = 0
        self.token_buffer = []
        self.document_buffer = []
    
    def state_dict(self):
        return {
            "shard_idx": self.shard_idx,
            "group_idx": self.group_idx,
            "idx_in_group": self.idx_in_group,
            "token_buffer": self.token_buffer,
            "document_buffer": self.document_buffer,
        }

    def load_state_dict(self, state):
        self.shard_idx = state["shard_idx"]
        self.group_idx = state["group_idx"]
        self.idx_in_group = state["idx_in_group"]
        self.token_buffer = state["token_buffer"]
        self.document_buffer = state["document_buffer"]
        self.loaded_shard_idx = None
        self.loaded_shard_row_groups = None

    def _get_example_text(self):
        # Lead the requested shard
        # Note we load full shard, even though in ddp we skip a lot, potentially can be improved
        if self.shard_idx != self.loaded_shard_idx:
            filepath = os.path.join(self.folderpath, f"shard_{self.shard_idx:05d}.parquet")
            pf = pq.ParquetFile(filepath)
            self.loaded_shard_row_groups = []
            for rg_index in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_index)
                documents = rg.column('text').to_pylist()
                self.loaded_shard_row_groups.append(documents)  # list of lists or str
            self.loaded_shard_idx = self.shard_idx
        return self.loaded_shard_row_groups[self.group_idx][self.idx_in_group]

    def _get_current_shard_num_row_groups(self):
        return len(self.loaded_shard_row_groups)

    def _step_cursor(self):
        self.idx_in_group += 1
        if self.idx_in_group >= self.group_size:
            self.idx_in_group = 0
            self.group_idx += self.world_size
            if self.group_idx >= self._get_current_shard_num_row_groups():
                self.group_idx = self.rank
                self.shard_idx += 1
                if self.shard_idx > self.last_shard:
                    self.shard_idx = self.first_shard

    def _get_next_document(self):
        prompt = self._get_example_text()        
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


