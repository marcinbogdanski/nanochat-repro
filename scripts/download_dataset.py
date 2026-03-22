import os
import shutil
import argparse
from urllib.request import urlopen

# See notebooks/0100_dataset_prefetch for how these are created
# Our method generates bit-for-bit identical parquet files, but since Andrej already
# published them, we download from his HF instead of duplicating the dataset
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main/"
BASE_PATH = os.path.expanduser("~/.cache/nanochat/base_data")
LAST_FILE = 1822  # inclusive

def download_file(filename):
    remote_url = BASE_URL + filename
    filepath_tmp = os.path.join(BASE_PATH, filename + ".tmp")
    filepath_final = os.path.join(BASE_PATH, filename)

    if os.path.exists(filepath_final):
        print(f"Skipping {filepath_final}: already exists")
        return 0

    try:
        with urlopen(remote_url, timeout=30) as r, open(filepath_tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(filepath_tmp, filepath_final)
        print(f"Downloaded {filepath_final}")
        return 1
    except Exception:
        os.remove(filepath_tmp)
        raise

def main():
    parser = argparse.ArgumentParser(description="Script to download data shards.")
    parser.add_argument('-n', '--num-files', type=int, default=240, help='Num of train shards to get. Validation shard is always added on top.')
    args = parser.parse_args()

    num_downloaded = 0
    to_download = list(range(args.num_files)) + [LAST_FILE]  # Always get last for validation
    for i in to_download:
        filename = f"shard_{i:05d}.parquet"
        num_downloaded += download_file(filename)
    print(f"Num downloaded: {num_downloaded}")

if __name__ == '__main__':
    main()
