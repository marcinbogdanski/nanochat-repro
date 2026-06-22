import os
import torch
import pyarrow.parquet as pq
from nanorepro.common import get_base_path
BASE_DIR = get_base_path()

class DataLoader:
    def __init__(self, dataset_or_folderpath, split, batch_size, block_size, tokenizer, device):
        assert device == 'cpu' or device.startswith('cuda')

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
        self.document_buffer = []

        # Distributed
        self.group_size = 1024  # same as nanochat
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        # Create Cursor
        self.shard_idx = self.first_shard
        self.group_idx = self.rank
        self.idx_in_group = 0

        # Cached Shards
        self.loaded_shard_idx = None
        self.loaded_shard_num_rg = None
        self.loaded_shard_pf = None        # cached pq.ParquetFile(filepath)
        self.loaded_shard_group_idx = None
        self.loaded_shard_rg_docs = None   # current row group documents, list or str

        # Tensor Buffers
        self.use_cuda = device.startswith("cuda")
        B, T = batch_size, block_size
        # T+1 because targets look "one beyond" the sequence length
        self.row_buffer = torch.empty((B, T+1), dtype=torch.long)  # Construct rows w/o massive python lists
        self.cpu_buffer = torch.empty((2*B*T), dtype=torch.long, pin_memory=self.use_cuda)  # cpu-side contiguous buffer
        self.cpu_x = self.cpu_buffer[:B*T].view(B, T)  # first half of cpu_buffer is inputs
        self.cpu_y = self.cpu_buffer[B*T:].view(B, T)  # second half is targets
        self.gpu_buffer = torch.empty((2*B*T), dtype=torch.long, device=device)
        self.result_x = self.gpu_buffer[:B*T].view(B, T)
        self.result_y = self.gpu_buffer[B*T:].view(B, T)

    def reset(self):
        """Called to reset eval dataloader."""
        self.shard_idx = self.first_shard
        self.group_idx = self.rank
        self.idx_in_group = 0
        self.document_buffer = []
        self.loaded_shard_idx = None
        self.loaded_shard_num_rg = None
        self.loaded_shard_pf = None
        self.loaded_shard_group_idx = None
        self.loaded_shard_rg_docs = None
    
    def state_dict(self):
        # Currently train loop operates as:
        # while True:
        #     save_checkpoint(dataloader)  <- dataloader advanced cursor, but x,y not consumed
        #     model(x, y)
        #     x, y = dataloader.get_batch_bos()
        # We save last x,y so they can be consumed on resume, otherwise they would be skipped
        return {
            "shard_idx": self.shard_idx,
            "group_idx": self.group_idx,
            "idx_in_group": self.idx_in_group,
            "document_buffer": self.document_buffer,
            "last_x": self.cpu_x,
            "last_y": self.cpu_y,
        }

    def load_state_dict(self, state):
        self.shard_idx = state["shard_idx"]
        self.group_idx = state["group_idx"]
        self.idx_in_group = state["idx_in_group"]
        self.document_buffer = state["document_buffer"]
        self.loaded_shard_idx = None
        self.loaded_shard_num_rg = None
        self.loaded_shard_pf = None
        self.loaded_shard_group_idx = None
        self.loaded_shard_rg_docs = None
        # Restore buffer on gpu
        self.cpu_x.copy_(state["last_x"])
        self.cpu_y.copy_(state["last_y"])
        self.gpu_buffer.copy_(self.cpu_buffer, non_blocking=self.use_cuda)

    def _get_example_text_batch(self, num):
        # Init the requested shard, not load yet
        if self.shard_idx != self.loaded_shard_idx:
            filepath = os.path.join(self.folderpath, f"shard_{self.shard_idx:05d}.parquet")
            self.loaded_shard_pf = pq.ParquetFile(filepath)
            self.loaded_shard_idx = self.shard_idx
            self.loaded_shard_num_rg = self.loaded_shard_pf.num_row_groups
            self.loaded_shard_group_idx = None
            self.loaded_shard_rg_docs = None
        # Load the required row group
        if self.group_idx != self.loaded_shard_group_idx:
            rg = self.loaded_shard_pf.read_row_group(self.group_idx)
            self.loaded_shard_rg_docs = rg.column('text').to_pylist()
            self.loaded_shard_group_idx = self.group_idx
        assert self.idx_in_group + num <= len(self.loaded_shard_rg_docs)
        result = self.loaded_shard_rg_docs[self.idx_in_group:self.idx_in_group+num]
        assert len(result) == num
        return result

    def _step_cursor(self, num):
        assert self.group_size % num == 0  # otherwise we need to support iterating multiple row groups
        self.idx_in_group += num
        if self.idx_in_group >= self.group_size:
            self.idx_in_group = 0
            self.group_idx += self.world_size
            if self.group_idx >= self.loaded_shard_num_rg:
                self.group_idx = self.rank
                self.shard_idx += 1
                if self.shard_idx > self.last_shard:
                    self.shard_idx = self.first_shard

    def _get_next_document_batch(self, num):
        doc_list = self._get_example_text_batch(num=num)
        self._step_cursor(num=num)
        doc_tokens_list_of_lists = self.tokenizer.encode_ordinary_batch(doc_list, num_threads=4)
        for doc_tokens in doc_tokens_list_of_lists:
            doc_tokens.insert(0, self.bos_token)
        return doc_tokens_list_of_lists

    def _fill_doc_buffer(self):
        while len(self.document_buffer) < 1000:
            tok_batch_size = 128
            doc_tokens_list_of_lists = self._get_next_document_batch(num=tok_batch_size)
            for doc_tokens in doc_tokens_list_of_lists:
                self.document_buffer.append(doc_tokens)

    def get_last_batch_without_advancing(self):
        """Useful after state load to consume last data batch"""
        return self.result_x, self.result_y

    def get_batch_bos(self):
        need_row_tokens = self.block_size + 1

        for bi in range(self.batch_size):
            row_pos = 0
            while row_pos < need_row_tokens:
                self._fill_doc_buffer()
                # Find longest doc that fits
                longest_doc_idx = None
                longest_doc_len = 0
                num_tokens_to_fill = need_row_tokens - row_pos
                for di in range(len(self.document_buffer)):
                    candidate_doc = self.document_buffer[di]
                    if len(candidate_doc) <= num_tokens_to_fill and len(candidate_doc) > longest_doc_len:
                        longest_doc_idx = di
                        longest_doc_len = len(candidate_doc)
                # Extend row
                if longest_doc_idx is not None:
                    longest_doc_that_fits = self.document_buffer.pop(longest_doc_idx)
                    doc_length = len(longest_doc_that_fits)
                    self.row_buffer[bi, row_pos:row_pos+doc_length] = torch.tensor(longest_doc_that_fits, dtype=torch.long)
                    row_pos += len(longest_doc_that_fits)
                else:
                    shortest_idx = min(range(len(self.document_buffer)), key=lambda i: len(self.document_buffer[i]))
                    doc_to_trim = self.document_buffer.pop(shortest_idx)
                    trimmed_doc = doc_to_trim[:num_tokens_to_fill]
                    self.row_buffer[bi, row_pos:row_pos+num_tokens_to_fill] = torch.tensor(trimmed_doc, dtype=torch.long)
                    row_pos += num_tokens_to_fill

        # Copy to GPU
        self.cpu_x.copy_(self.row_buffer[:,:-1])  # copy to first half of cpu_buffer through a view
        self.cpu_y.copy_(self.row_buffer[:,1:])
        self.gpu_buffer.copy_(self.cpu_buffer, non_blocking=self.use_cuda)
        
        return self.result_x, self.result_y


