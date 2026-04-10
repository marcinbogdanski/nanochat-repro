import os
import shutil
import argparse
from urllib.request import urlopen
from mynanochat.common import get_base_path
BASE_DIR = get_base_path()


def download_file(filename, base_url, base_path):
    remote_url = base_url + filename
    filepath_tmp = os.path.join(base_path, filename + ".tmp")
    filepath_final = os.path.join(base_path, filename)
    os.makedirs(base_path, exist_ok=True)

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
        if os.path.exists(filepath_tmp):
            os.remove(filepath_tmp)
        raise

def main():
    parser = argparse.ArgumentParser(description="Script to download data shards.")
    parser.add_argument("-d", "--dataset", type=str, default="climbmix", choices=["fineweb", "climbmix"], help="Dataset to process (fineweb or climbmix)")
    parser.add_argument('-n', '--num-files', type=int, default=None, help='Num of train shards to get. Validation shard is always added on top.')
    args = parser.parse_args()

    if args.dataset == "fineweb":
        base_url = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main/"
        base_path = os.path.join(BASE_DIR, "base_data")
        last_file = 1822  # inclusive
    elif args.dataset == "climbmix":
        base_url = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/"
        base_path = os.path.join(BASE_DIR, "base_data_climbmix")
        last_file = 6542  # inclusive
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    num_downloaded = 0
    max_shard = last_file if args.num_files is None else min(args.num_files - 1, last_file)
    to_download = list(range(max_shard+1)) + [last_file]  # Always get last for validation
    for i in to_download:
        filename = f"shard_{i:05d}.parquet"
        num_downloaded += download_file(filename, base_url, base_path)
    print(f"Num downloaded: {num_downloaded}")

if __name__ == '__main__':
    main()
