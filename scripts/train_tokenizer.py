import os
import rustbpe
import tiktoken
import pickle
import torch
import pyarrow.parquet as pq

# from nanochat tokenizer.py
# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I haven't validated that this is actually a good idea
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# from nanochat tokenizer.py
SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

BASE_DATA_PATH = os.path.expanduser("~/.cache/nanochat/base_data")
BASE_TOKENIZER_PATH = os.path.expanduser("~/.cache/nanochat/tokenizer")

def doc_generator():
    for shard_idx in range(9999):
        filename = f"shard_{shard_idx:05d}.parquet"
        filepath = os.path.join(BASE_DATA_PATH, filename)
        pf = pq.ParquetFile(filepath)
        for rg_index in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_index)
            documents = rg.column('text').to_pylist()
            for doc in documents:
                yield doc

def main():
    # Match params used in nanochat speedrun.sh
    # python -m scripts.tok_train --max_chars=2000000000 --vocab_size=65536
    max_chars = 2000000000
    vocab_size = 65536
    doc_cap = 10000

    # Load training documents
    train_docs = []
    char_count = 0
    for i, text in enumerate(doc_generator()):
        if len(text) > doc_cap:
            text = text[:doc_cap]
        train_docs.append(text)
        char_count += len(text)
        
        if i % 100000 == 0 or char_count >= max_chars:
            pct = (char_count / max_chars) * 100
            print(f"Processed {char_count} / {max_chars} ({pct:.2f}%)")
        
        if char_count >= max_chars:
            break

    # Create tokenizer and train on your data
    vocab_size_no_specials = vocab_size - len(SPECIAL_TOKENS)
    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(
        train_docs,
        vocab_size=vocab_size_no_specials,
        pattern=SPLIT_PATTERN
    )

    # Save tokenizer
    # Should be bitwise identical to Nanochat version
    pattern = tokenizer.get_pattern()
    mergeable_ranks_list = tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
        special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
    )
    os.makedirs(BASE_TOKENIZER_PATH, exist_ok=True)
    tokenizer_path = os.path.join(BASE_TOKENIZER_PATH, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(enc, f)

    # Save token bytes
    # Should be bitwise identical to Nanochat version
    token_strings = [enc.decode([i]) for i in range(enc.n_vocab)]  # list[str] (token id -> token string)
    token_bytes = []
    for i in range(enc.n_vocab):
        tok_str = token_strings[i]
        if tok_str in special_tokens:
            # special tokens are not byte sequences
            token_bytes.append(0)
        else:
            token_bytes.append(len(tok_str.encode("utf-8")))
    token_bytes_pt = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
    token_bytes_path = os.path.join(BASE_TOKENIZER_PATH, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes_pt, f)


if __name__ == '__main__':
    main()


