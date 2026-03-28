import torch
import torch.nn.functional as F

# Flash Attention 3, source wheel with 3090 support
from kernels import get_kernel
_fa3 = get_kernel('kernels-community/flash-attn3').flash_attn_interface


def fa3_attn_func(q, k, v, causal, window_size):
    # q, k, v are [B,T,nh,hs] dims
    return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)


def _sdpa_attn_func(q, k, v, causal, window_size):
    """SDPA wrapper to implement window_size"""
    assert causal
    assert window_size[1] == 0  # future attention not supported
    # q, k, v are [B,nh,T,hs] dims
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]
    
    # Full size window and square mask
    #   Tq == Tk           <- square attention matrix
    #   (W < 0 or W > Tk)  <- full window
    # We are reduced to standard square attention matrix, no need for a mask
    # [1, 0, 0]
    # [1, 1, 0]
    # [1, 1, 1]
    if Tq == Tk and (window < 0 or window >= Tk):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # Generation special case, can disable 'causal'
    #   Tq == 1   <- one token generation
    #   W >= 0    <- window enabled (not -1)
    #   W < Tk    <- there is something to cut
    #  v--v--------------- cut these two zeros
    # [0, 0, 1, 1, 1]
    if Tq == 1:
        if window >= 0 and window < Tk:  # window enabled (not -1) and in effect
            # [0, 0, 1, 1, 1] -> [1, 1, 1]
            start = Tk - (window+1)
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)
    
    # Example for T1 = 5, Tk = 8, W = 2
    row_idx = (Tk-Tq) + torch.arange(Tq, device=q.device)
    row_idx = row_idx.unsqueeze(1)                            # column [[3], [4], [5], [6], [7]]
    col_idx = torch.arange(Tk, device=q.device).unsqueeze(0)  # row [0, 1, 2, 3, 4, 5, 6, 7]

    # [[1, 1, 1, 1, 0, 0, 0, 0],
    #  [1, 1, 1, 1, 1, 0, 0, 0],
    #  [1, 1, 1, 1, 1, 1, 0, 0],
    #  [1, 1, 1, 1, 1, 1, 1, 0],
    #  [1, 1, 1, 1, 1, 1, 1, 1]]
    mask = col_idx <= row_idx

    # window enabled (not -1) and has effect
    if window >= 0 and window < Tk:
        # [[0, 1, 1, 1, 1, 1, 1, 1],
        #  [0, 0, 1, 1, 1, 1, 1, 1],
        #  [0, 0, 0, 1, 1, 1, 1, 1],
        #  [0, 0, 0, 0, 1, 1, 1, 1],
        #  [0, 0, 0, 0, 0, 1, 1, 1]]
        in_window_mask = (row_idx - col_idx) <= window

        # Note we could technically 'cut' first column, but meh
        # [[0, 1, 1, 1, 0, 0, 0, 0],
        #  [0, 0, 1, 1, 1, 0, 0, 0],
        #  [0, 0, 0, 1, 1, 1, 0, 0],
        #  [0, 0, 0, 0, 1, 1, 1, 0],
        #  [0, 0, 0, 0, 0, 1, 1, 1]]
        mask = mask & in_window_mask

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def sdpa_attn_func(q, k, v, causal, window_size):
    """SDPA wrapper to match FA3 API"""
    #  q, k, v are [B,T,nh,hs] dims, but PyTorch SDPA expects B,nh,T,hs
    q = q.transpose(1, 2)  # B,nh,T,hs
    k = k.transpose(1, 2)  # B,nh,T,hs
    v = v.transpose(1, 2)  # B,nh,T,hs

    # Actual attention call
    y = _sdpa_attn_func(q, k, v, causal=causal, window_size=window_size)

    # Transpose back
    return y.transpose(1, 2)  # B,T,nh,hs
