import os
import torch
import pickle
from contextlib import nullcontext
from mynanochat.gpt import GPTConfig, GPTModel

def main():
    # DDP Init
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this ddp run?
    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        ddp_master = ddp_rank == 0  # is this a master?
        device = f'cuda:{ddp_local_rank}'
        device_type = 'cuda'
        assert torch.cuda.is_available()
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend='nccl', device_id=ddp_local_rank)  # device_id= to suppress barrier warning
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        ddp_master = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_type = device
    print(f"{ddp=} {ddp_rank=}, {ddp_local_rank=}, {ddp_world_size=}, {ddp_master=}, {device=}")

    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == 'cuda' else nullcontext()


    # Tokenizer
    tokenizer_path = os.path.dirname(__file__)+"/../data/tokenizer.pkl"
    tokenizer = pickle.load(open(tokenizer_path, "rb"))


    # Model Hyperparameters
    vocab_size = tokenizer.n_vocab
    num_layers = 20
    num_embed = 1280
    head_size = 128
    assert num_embed % head_size == 0
    num_heads = num_embed // head_size
    
    # Training Hyperparameters
    total_batch_size = 524288    # 2**19, ~0.5M
    micro_batch = 16             # what fits in GPU
    block_size = 2048
    assert total_batch_size % (block_size*micro_batch*ddp_world_size) == 0
    grad_accum = total_batch_size // (block_size*micro_batch*ddp_world_size)

    # Reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    ################################ EQUIVALENCE ###############################
    # Dissable TORCH.COMPILE for reproducibility non-DDP/DDP
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    ############################################################################


    # Model
    model_config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_embd=num_embed,
    )
    model = GPTModel(model_config)
    model.to(device)
    model.init_weights()


    ################################ QUICK CHECK ###############################
    # Iterate model params and print first few for each
    # for i, p in enumerate(model.parameters()):
    #     with torch.no_grad():
    #         print(f"{i} {tuple(p.size())}, {p.dtype}, {p.device} {p.flatten()[:5].tolist()}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {num_params} parameters")

    prompt = "Hello, I'm a language model, and"  # 8 tokens
    tokens = tokenizer.encode_ordinary(prompt)

    x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)  # B=1,T
    y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)   # B=1,T
    with torch.no_grad():
        with autocast_ctx:
            logits, loss = model(x, y)  # B,T,C
    print("Logits shape:", logits[0].shape)  # should be (1,8,vocab_size)
    print(f"{tuple(logits.size())}, {logits.dtype}, {logits.device} {logits.flatten()[:5].tolist()}")
    print("Loss:", loss.item())   # ~11.0 for random init
    ############################################################################


if __name__ == "__main__":
    main()
