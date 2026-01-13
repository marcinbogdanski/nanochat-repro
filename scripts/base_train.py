import os
import torch
import pickle
from contextlib import nullcontext
from mynanochat.gpt import GPTConfig, GPTModel
from mynanochat.dataloader import DataLoader

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
    micro_batch = 2              # what fits in GPU
    block_size = 2048
    assert total_batch_size % (block_size*micro_batch*ddp_world_size) == 0
    grad_accum = total_batch_size // (block_size*micro_batch*ddp_world_size)

    # Reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Precision
    if device_type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32" # uses tf32 instead of fp32 for matmuls

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

    # Optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizers = [optimizer]


    train_loader = DataLoader(
        batch_size=micro_batch,
        block_size=block_size,
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_name="sample-100BT",
        hf_split="train",
        tokenizer=tokenizer,
    )


    max_steps = 2
    for step in range(max_steps):


        ### v SAVE v ###
        save_dict = {
            'step': step,
            'x': [],
            'y': [],
        }
        save_dict['weights_before'] = []
        for name, p in model.named_parameters():
            save_dict['weights_before'].append((name, p.detach().clone().cpu()))
        ### ^ SAVE ^ ###

        model.train()
        loss_accum = 0.0
        for opt in optimizers:
            opt.zero_grad()
        for ii in range(grad_accum):
            x, y = train_loader.get_batch()
            x = x.to(device)
            y = y.to(device)
            with autocast_ctx:
                logits, loss = model(x, y)
            loss = loss / grad_accum
            loss_accum += loss.detach()

            ## v SAVE v ###
            save_dict['x'].append(x.detach().clone().cpu())
            save_dict['y'].append(y.detach().clone().cpu())
            if 'loss_div_accum' not in save_dict:
                save_dict['loss_div_accum'] = []
            save_dict['loss_div_accum'].append(loss.detach().clone().cpu())
            ### ^ SAVE ^ ###

            # TODO: Sync only if DDP and last backward in grad_accum
            loss.backward()

        # v SAVE v ###
        save_dict['gradients'] = []
        for _, p in enumerate(model.parameters()):
            if p.grad is not None:
                save_dict['gradients'].append(p.grad.detach().clone().cpu())
            else:
                save_dict['gradients'].append(None)
        ### ^ SAVE ^ ###


        ### v SAVE v ###
        save_dict['lrm'] = 0.0
        save_dict['muon_momentum'] = 0.0
        ### ^ SAVE ^ ###


        # Optimizer step
        for opt in optimizers:
            opt.step()


        ### v SAVE v ###
        save_dict['optimizer_states'] = [opt.state_dict() for opt in optimizers]
        save_dict['weights_after'] = []
        for name, p in model.named_parameters():
            save_dict['weights_after'].append((name, p.detach().clone().cpu()))

        # save the save_dict for this step (for debugging)
        filename = os.path.join(f"dump_step_{step:05d}_rank_{ddp_rank}.pt")
        torch.save(save_dict, filename)
        ### ^ SAVE ^ ###



        # Logs
        print(f"Step {step+1}/{max_steps}, loss: {loss_accum.item():.4f}")
                

    return

    ################################ QUICK CHECK ###############################
    # Iterate model params and print first few for each
    print('-'*40, "Model init parameters", '-'*40)
    for i, p in enumerate(model.parameters()):
        with torch.no_grad():
            print(f"{i} {tuple(p.size())}, {p.dtype}, {p.device} {p.flatten()[:5].tolist()}")
    print('-'*100)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {num_params} parameters")

    prompt = "Hello, I'm a language model, and"  # 8 tokens
    tokens = tokenizer.encode_ordinary(prompt)

    x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)  # B=1,T
    y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)   # B=1,T
    with autocast_ctx:
        logits, loss = model(x, y)  # B,T,C
    print("Logits shape:", logits[0].shape)
    print(f"{tuple(logits.size())}, {logits.dtype}, {logits.device} {logits.flatten()[:5].tolist()}")
    print("Loss:", loss.item())   # ~11.0 for random init

    # setup basic optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizers = [optimizer]

    loss.backward()
    print('-'*40, "Model gradients", '-'*40)
    for i, p in enumerate(model.parameters()):
        if p.grad is not None:
            print(f"Param {i} grad {p.grad.flatten()[:5].tolist()}")
        else:
            print(f"Param {i} grad is None")
    print('-'*100)

    for opt in optimizers:
        opt.step()

    print('-'*40, "Model after update", '-'*40)
    for i, p in enumerate(model.parameters()):
        with torch.no_grad():
            print(f"{i} {tuple(p.size())}, {p.dtype}, {p.device} {p.flatten()[:5].tolist()}")
    print('-'*100)

    ############################################################################


if __name__ == "__main__":
    main()
