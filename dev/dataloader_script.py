import os
import time
import pickle
import datasets
from mynanochat.dataloader import DataLoader
from mynanochat.dataloader_raw import DataLoader as DataLoaderHF

ddp_rank = 1
ddp_world_size = 4

dataset = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train")
dataset = dataset.shuffle(seed=42)

base_path = os.path.dirname(__file__)+"/../data/"
tokenizer_path = base_path + "tokenizer.pkl"
tokenizer = pickle.load(open(tokenizer_path, "rb"))

micro_batch=8
block_size=2048

folderpath = os.path.expanduser("~/.cache/nanochat/base_data")
train_loader = DataLoader(
    folderpath=folderpath,
    first_shard=0,
    last_shard=238,
    batch_size=micro_batch,
    block_size=block_size,
    tokenizer=tokenizer,
    rank=ddp_rank,
    world_size=ddp_world_size,
)


train_loader_hf = DataLoaderHF(
    dataset=dataset,
    first_shard=0,
    last_shard=238,
    batch_size=micro_batch,
    block_size=block_size,
    tokenizer=tokenizer,
    group_size=1024,   # same as nanochat row_group_size
    rank=ddp_rank,
    world_size=ddp_world_size,
)

print("Testing dataloader")

start_time = time.time()
for i in range(4000):
    x, y = train_loader.get_batch()
    x_hf, y_hf = train_loader_hf.get_batch()
    if i % 10 == 0:
        td = (time.time() - start_time) / (i+1)
        print(f"Idx: {i=} {td=:.2f} Shard: {train_loader.shard_idx=} {train_loader.group_idx=} {train_loader.idx_in_group=}")
    if not x.equal(x_hf) or not y.equal(y_hf):
        print("Very not good!")
