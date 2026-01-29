import os
import time
import torch
import pickle
from contextlib import nullcontext
from mynanochat.gpt import GPTConfig, GPTModel
from mynanochat.dataloader import DataLoader
from mynanochat.muon_karpathy import MuonK
from mynanochat.muon_torch import MuonT
from mynanochat.adamw import AdamW
from mynanochat.muon import Muon

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
    autocast_ctx = nullcontext()  # MARCIN - disable autocast for debugging

    # Tokenizer
    tokenizer_path = os.path.dirname(__file__)+"/../data/tokenizer.pkl"
    tokenizer = pickle.load(open(tokenizer_path, "rb"))


    # Model Hyperparameters
    vocab_size = tokenizer.n_vocab
    num_layers = 10  # 20                        ### MARCIN - smaller model
    num_embed = 640  # 1280                      ### MARCIN - smaller model
    head_size = 128
    assert num_embed % head_size == 0
    num_heads = num_embed // head_size
    
    # Training Hyperparameters
    total_batch_size = 524288 // 128   # 2**19, ~0.5M   ### MARCIN - smaller batch for debugging
    micro_batch = 1              # what fits in GPU     ### MARCIN - smaller micro batch for debugging
    block_size = 1024  # 2048                           ### MARCIN - smaller model
    assert total_batch_size % (block_size*micro_batch*ddp_world_size) == 0
    grad_accum = total_batch_size // (block_size*micro_batch*ddp_world_size)

    # Reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Precision                                 ### MARCIN - disable tf32 for debugging
    # if device_type == "cuda":
    #     torch.backends.cuda.matmul.fp32_precision = "tf32" # uses tf32 instead of fp32 for matmuls

    ################################ EQUIVALENCE ###############################
    # Dissable TORCH.COMPILE for reproducibility non-DDP/DDP
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
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
    params_matrix = list(model.transformer.h.parameters())
    params_embedding = list(model.transformer.wte.parameters())
    params_lm_head = list(model.lm_head.parameters())
    assert len(list(model.parameters())) == len(params_matrix) + len(params_embedding) + len(params_lm_head)

    reference_batch_size = 2**19
    batch_ratio = total_batch_size / reference_batch_size
    batch_lr = batch_ratio ** 0.5
    unembedding_lr = 0.004 * batch_lr
    embedding_lr = 0.3 * batch_lr
    matrix_lr = 0.02 * batch_lr
    adam_betas = (0.8, 0.95)

    # LR Scheduler params
    max_steps = 10                    ### MARCIN - fewer steps for debugging
    lr_warmup_ratio = 0.4     # 0.0   ### MARCIN - warmup testing
    lr_warmdown_ratio = 0.4
    lr_final_frac = 0.1       # 0.0   ### MARCIN - warmdown testing
    muon_momentum_warup_steps = 5  # 300  ### MARCIN - faster momentum warmup for debugging

    # LR / Muon Scheduler functions
    def get_lr(step: int):
        warmup_steps = round(lr_warmup_ratio * max_steps)
        warmdown_steps = round(lr_warmdown_ratio * max_steps)
        if step < warmup_steps:
            return (step+1) / warmup_steps
        if step <= max_steps - warmdown_steps:
            return 1.0
        else:
            progress = (max_steps - step) / warmdown_steps
            return (progress * 1.0) + (1.0 - progress) * lr_final_frac

    def get_muon_momentum(step: int):
        muon_frac = min(step / muon_momentum_warup_steps, 1.0)
        muon_momentum = (1.0 - muon_frac) * 0.85 + muon_frac * 0.95
        return muon_momentum

    model_dim = model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    adam_groups = [
        {
            'params': params_lm_head,
            'lr': unembedding_lr * dmodel_lr_scale,
        },
        {
            'params': params_embedding,
            'lr': embedding_lr * dmodel_lr_scale,
        }
    ]
    adamw_optimizer = torch.optim.AdamW(
        adam_groups,
        betas=adam_betas,
        eps=1e-10,
        weight_decay=0.0,
        fused=True,
    )
    muon_groups = []
    for size in {p.numel() for p in params_matrix}:
        group_params = [p for p in params_matrix if p.numel() == size]
        muon_groups.append({'params': group_params})
    muon_optimizer = Muon(
        muon_groups,
        lr=matrix_lr,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
    )

    optimizers = [adamw_optimizer, muon_optimizer]
    for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

    train_loader = DataLoader(
        batch_size=micro_batch,
        block_size=block_size,
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_name="sample-100BT",
        hf_split="train",
        tokenizer=tokenizer,
    )

    for step in range(max_steps):

        ### v SAVE v ###
        save_dict = {
            'step': step,
            'x': [],
            'y': [],
            'logits': [],
            'loss_div_accum': [],
        }
        save_dict['weights_before'] = []
        for name, p in model.named_parameters():
            save_dict['weights_before'].append((name, p.detach().clone().cpu()))
        save_dict['buffs_before'] = {}
        for n, p in model.named_buffers():
            save_dict['buffs_before'][n] = p.detach().clone().cpu()
        ### ^ SAVE ^ ###

        # Training
        ts = time.time()
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
            save_dict['logits'].append(logits.detach()[:,::4,::64].clone().cpu())
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

        # LR Scheduler
        lrm = get_lr(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = group['initial_lr'] * lrm
        muon_momentum = get_muon_momentum(step)
        for group in muon_optimizer.param_groups:
            group['momentum'] = muon_momentum
        
        ### v SAVE v ###
        save_dict['lrm'] = lrm
        save_dict['muon_momentum'] = muon_momentum
        save_dict['optimizer_states_before'] = [opt.state_dict() for opt in optimizers]
        ### ^ SAVE ^ ###

        # Optimizer step
        for opt in optimizers:
            opt.step()

        ### v SAVE v ###
        save_dict['optimizer_states_after'] = [opt.state_dict() for opt in optimizers]
        save_dict['weights_after'] = []
        for name, p in model.named_parameters():
            save_dict['weights_after'].append((name, p.detach().clone().cpu()))

        # save the save_dict for this step (for debugging)
        filename = os.path.join(f"dump_step_{step:05d}_rank_{ddp_rank}.pt")
        torch.save(save_dict, filename)
        ### ^ SAVE ^ ###

        # Logs
        dt = (time.time() - ts)
        print(f"Step {step+1}/{max_steps}, loss: {loss_accum.item():.4f}, dt={dt*1e3:.2f}ms")

    return


if __name__ == "__main__":
    main()
