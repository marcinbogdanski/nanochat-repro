"""Repackage the raw HuggingFace datasets into the parquet format.

This duplicates the repackage_data_reference.py logic in Nanochat. Our method
generates bit-for-bit identical parquet files, but since Andrej already
published them, we download from his HF instead of duplicating the dataset.
"""

import os
import time
import argparse
import tiktoken
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
assert pa.__version__ == '21.0.0'  # bitwise parity with Nanochat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="fineweb", choices=["fineweb", "climbmix"], help="Dataset to process (fineweb or climbmax)")
    parser.add_argument("-n", "--num-shards", type=int, default=-1, help="Number of shards to create (-1 to process entire dataset)")
    args = parser.parse_args()

    # Parameters
    if args.dataset == "fineweb":
        # NOTE: Takes ~750GB disk space!!!
        dataset_hf_path = "HuggingFaceFW/fineweb-edu"
        dataset_hf_name = "sample-100BT"
        dataset_hf_split = "train"
        data_column = "text"
        tokenizer = None
        output_dir_name = "fineweb-edu-100b-shuffle"
        ref_dir_name = "base_data"
    elif args.dataset == "climbmix":
        # NOTE: Takes 2TB+ disk space!!!!!!
        # Also: note tested vs Nanochat because I couldn't download the whole thing at the time
        dataset_hf_path = "nvidia/Nemotron-ClimbMix"
        dataset_hf_name = None
        dataset_hf_split = "train"
        data_column = "tokens"
        tokenizer = tiktoken.encoding_for_model("gpt-2")
        output_dir_name = "climbmix-400b-shuffle"
        ref_dir_name = "base_data_climbmix"

    output_path = os.path.expanduser(f"~/.cache/mynanochat/{output_dir_name}")
    reference_path = os.path.expanduser(f"~/.cache/nanochat/{ref_dir_name}")
    os.makedirs(output_path, exist_ok=True)

    # Load the dataset
    dataset = datasets.load_dataset(
        dataset_hf_path,
        name=dataset_hf_name,
        split=dataset_hf_split,
        # streaming=True,
    )
    dataset = dataset.shuffle(seed=42)  # Match nanochat repackage_data_reference.py seed

    # Hardcoded for now
    chars_per_shard = 250_000_000
    row_group_size = 1024

    # Iterate through the dataset
    shard_idx = 0
    shard_docs = []
    shard_num_chars = 0
    time_start = time.time()
    for i, example in enumerate(dataset):
        data = example[data_column]
        text = tokenizer.decode(data) if tokenizer is not None else data
        shard_docs.append(text)
        shard_num_chars += len(text)
        if shard_num_chars >= chars_per_shard and len(shard_docs) % row_group_size == 0:
            # Write out the current shard to a parquet file
            shard_table = pa.Table.from_pydict({'text': shard_docs})
            shard_filename = f'shard_{shard_idx:05d}.parquet'
            shard_path = os.path.join(output_path, shard_filename)
            pq.write_table(
                table=shard_table,
                where=shard_path,
                row_group_size=row_group_size,
                use_dictionary=False,
                compression="zstd",
                compression_level=3,
                write_statistics=False,
            )

            # Optional: Compare MD5 vs Nanochat reference if exists
            md5sum_str = ""
            shard_path_ref = os.path.join(reference_path, f"shard_{shard_idx:05d}.parquet")
            if os.path.exists(shard_path_ref):
                md5sum_ref = os.popen(f"md5sum {shard_path_ref}").read().split()[0][:8]  # first 8 chars for brevity
                md5sum_mine = os.popen(f"md5sum {shard_path}").read().split()[0][:8]
                md5sum_str = f" md5 mine: {md5sum_mine}, md5 ref: {md5sum_ref}"

            # Print progress
            pct = (i+1) / len(dataset) * 100
            total_time = time.time() - time_start
            eta = total_time / (i+1) * (len(dataset) - (i+1))
            print(f'Written shard {shard_idx} time={total_time:.2f}s pct={pct:.2f}% eta={eta:.2f}s{md5sum_str}')

            # Advance to next shard
            shard_idx += 1
            shard_docs = []
            shard_num_chars = 0
            if shard_idx == args.num_shards:
                break


if __name__ == "__main__":
    main()
