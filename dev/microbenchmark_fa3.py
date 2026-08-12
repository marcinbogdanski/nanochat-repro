import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # disable FA3 kernels progress bars
import random
import torch


from nanorepro.flash_attention import fa3_attn_func, fa3_attn_with_kvcache, sdpa_attn_func, sdpa_attn_with_kvcache

def main():
    B,T,nh,hs = 2, 8, 4, 16
    window_size = (4, 0)  # sliding window of size 4
    assert window_size[0] < T

    for _ in range(10):
        # test fa3_attn_func, sdpa_attn_func are close
        q = torch.randn(B, T, nh, hs, device='cuda', dtype=torch.float16)   #, B,T,nh,hs
        k = torch.randn(B, T, nh, hs, device='cuda', dtype=torch.float16)
        v = torch.randn(B, T, nh, hs, device='cuda', dtype=torch.float16)
        y_fa3 = fa3_attn_func(q, k, v, causal=True, window_size=(T, 0))    # (T,0) is full causal
        y_sdpa = sdpa_attn_func(q, k, v, causal=True, window_size=(T, 0))
        # Appropriate tolerance comparison
        assert torch.allclose(y_fa3, y_sdpa, atol=1e-2, rtol=1e-2)

    for _ in range(10):
        # test fa3_attn_func, sdpa_attn_func are close with window_size
        q = torch.randn(B, T, nh, hs, device='cuda', dtype=torch.float16)   #, B,T,nh,hs
        k = torch.randn(B, T, nh, hs, device='cuda', dtype=torch.float16)
        v = torch.randn(B, T, nh, hs, device='cuda', dtype=torch.float16)
        window_size = (4, 0)  # sliding window of size 4
        y_fa3 = fa3_attn_func(q, k, v, causal=True, window_size=window_size)
        y_sdpa = sdpa_attn_func(q, k, v, causal=True, window_size=window_size)
        # Appropriate tolerance comparison
        assert torch.allclose(y_fa3, y_sdpa, atol=1e-2, rtol=1e-2)

    B,T_new,T_max,nh,hs = 2, 4, 8, 4, 16
    window_size = (4, 0)  # sliding window of size 4
    for _ in range(10):
        seq_len = random.randint(1, T_max-T_new)
        cache_seqlens = torch.tensor([seq_len]*B, device='cuda', dtype=torch.int32)  # cache lengths for each batch
    
        # test fa3_attn_with_kvcache, sdpa_attn_with_kvcache are close with window_size
        q = torch.randn(B, T_new, nh, hs, device='cuda', dtype=torch.float16)   #, B,T_new,nh,hs
        k = torch.randn(B, T_new, nh, hs, device='cuda', dtype=torch.float16)
        v = torch.randn(B, T_new, nh, hs, device='cuda', dtype=torch.float16)
        k_cache = torch.zeros(B, T_max, nh, hs, device='cuda', dtype=torch.float16)
        v_cache = torch.zeros(B, T_max, nh, hs, device='cuda', dtype=torch.float16)
        # Fill the cache with random values up to cache_seqlens
        for b in range(B):
            k_cache[b, :cache_seqlens[b]] = torch.randn(cache_seqlens[b], nh, hs, device='cuda', dtype=torch.float16)
            v_cache[b, :cache_seqlens[b]] = torch.randn(cache_seqlens[b], nh, hs, device='cuda', dtype=torch.float16)
        k_cache_expected = k_cache.clone()
        v_cache_expected = v_cache.clone()
        for b in range(B):
            start = cache_seqlens[b]
            end = start + T_new
            k_cache_expected[b, start:end] = k[b]
            v_cache_expected[b, start:end] = v[b]

        k_cache_clone = k_cache.clone()
        v_cache_clone = v_cache.clone()
        y_fa3 = fa3_attn_with_kvcache(q, k_cache_clone, v_cache_clone, k, v,
                                      cache_seqlens=cache_seqlens,
                                      causal=True,
                                      window_size=window_size)
        assert torch.allclose(k_cache_clone, k_cache_expected, atol=1e-2, rtol=1e-2)
        assert torch.allclose(v_cache_clone, v_cache_expected, atol=1e-2, rtol=1e-2)
    
        k_cache_clone = k_cache.clone()
        v_cache_clone = v_cache.clone()
        y_sdpa = sdpa_attn_with_kvcache(q, k_cache_clone, v_cache_clone, k, v,
                                        cache_seqlens=cache_seqlens,
                                        causal=True,
                                        window_size=window_size)
        assert torch.allclose(k_cache_clone, k_cache_expected, atol=1e-2, rtol=1e-2)
        assert torch.allclose(v_cache_clone, v_cache_expected, atol=1e-2, rtol=1e-2)

        # Appropriate tolerance comparison
        assert torch.allclose(y_fa3, y_sdpa, atol=1e-2, rtol=1e-2)
        


if __name__ == "__main__":
    main()
