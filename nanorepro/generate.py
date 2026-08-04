import torch
import torch.nn.functional as F

@torch.inference_mode()
def sample_one_token(logits, temperature=1.0, top_k=None, sample_rng=None):
    assert logits.ndim == 2  # B,C
    assert temperature >= 0.0
    assert top_k is None or 0 < top_k <= logits.size(-1)
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)  # greedy
    if top_k is None:
        probs = F.softmax(logits / temperature, dim=-1)  # B,C
        ix = torch.multinomial(probs, num_samples=1, generator=sample_rng)  # B,1
        return ix
    else:
        topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)  # B,k
        probs = F.softmax(topk_logits / temperature, dim=-1)  # B,k
        ix = torch.multinomial(probs, num_samples=1, generator=sample_rng)  # B,1
        return torch.gather(topk_indices, -1, ix)  # B,1

@torch.inference_mode()
def generate(model, idx, max_new_tokens, temperature=0.0, top_k=None, sample_rng=None):
    """Generate max_tokens starting from idx[B,T]"""
    assert isinstance(idx, torch.Tensor)
    assert idx.dtype == torch.long
    assert len(idx.shape) == 2  # B,T
    assert isinstance(max_new_tokens, int)
    
    is_training = model.training
    model.eval()

    block_size = model.config.block_size
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_tail = idx[:, -block_size:]      # B,T  sliding window
            logits, _, _ = model(idx_tail)      # B,T,C <- B,T
            logits = logits[:, -1, :]            # B,C <- B,T,C  discard all but last
            xcol = sample_one_token(logits, temperature=temperature, top_k=top_k, sample_rng=sample_rng)  # B,1
            idx = torch.cat((idx, xcol), dim=1)  # B,T+1  append
    
    model.train(is_training)
    return idx

@torch.inference_mode()
def generate_test_samples(model, tokenizer, device):
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    was_training = model.training
    model.eval()
    try:
        bos = tokenizer.encode_single_token('<|bos|>')
        results = []
        for prompt in prompts:

            tokens =  [bos] + tokenizer.encode(prompt)
            idx = torch.tensor(tokens, dtype=torch.long, device=device)
            idx = idx.unsqueeze(0)  # B,T
            idx = generate(
                model,
                idx,
                max_new_tokens=16,
                temperature=0.0,
                top_k=None,
                sample_rng=sample_rng
            )  # B,T
            gen_text = tokenizer.decode(idx[0].tolist())
            results.append(gen_text)
        return results
    finally:
        model.train(was_training)
