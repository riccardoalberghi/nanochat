"""
Pure PyTorch attention implementation — no precompiled kernels.

Simple, hackable causal attention using plain matmuls and masks.
All tensors use (B, T, H, D) layout throughout.

Usage:
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import math
import torch
import torch.nn.functional as F


def _attention(q, k, v, causal=False, window_size=(-1, -1), poly_beta=None):
    """
    Pure PyTorch scaled dot-product attention.

    Args:
        q: (B, Tq, Hq, D)
        k: (B, Tk, Hk, D)
        v: (B, Tk, Hk, D)
        causal: apply causal mask
        window_size: (left, right) sliding window. -1 means unlimited.
        poly_beta: optional float for polynomial attention: scores become poly_beta * s^2 + s

    Returns:
        (B, Tq, Hq, D)
    """
    B, Tq, Hq, D = q.shape
    Tk = k.size(1)
    Hk = k.size(2)
    scale = 1.0 / math.sqrt(D)

    # Handle GQA: repeat KV heads to match Q heads
    if Hq != Hk:
        assert Hq % Hk == 0
        repeats = Hq // Hk
        k = k.unsqueeze(3).expand(B, Tk, Hk, repeats, D).reshape(B, Tk, Hq, D)
        v = v.unsqueeze(3).expand(B, Tk, Hk, repeats, D).reshape(B, Tk, Hq, D)

    # Transpose to (B, H, T, D) for batched matmul
    q = q.transpose(1, 2)  # (B, H, Tq, D)
    k = k.transpose(1, 2)  # (B, H, Tk, D)
    v = v.transpose(1, 2)  # (B, H, Tk, D)

    # Compute attention scores
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, Tq, Tk)

    # Polynomial attention: poly_beta * s^2 + s
    if poly_beta is not None:
        attn = poly_beta * attn.square() + attn

    # Build mask
    if causal or (window_size[0] >= 0 and window_size[0] < Tk):
        # Row indices for queries, accounting for possible offset (Tk > Tq in KV cache case)
        offset = Tk - Tq
        row = offset + torch.arange(Tq, device=q.device).unsqueeze(1)  # (Tq, 1)
        col = torch.arange(Tk, device=q.device).unsqueeze(0)            # (1, Tk)

        mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device)
        if causal:
            mask = mask & (col <= row)
        if window_size[0] >= 0 and window_size[0] < Tk:
            mask = mask & ((row - col) <= window_size[0])

        attn = attn.masked_fill(~mask, float('-inf'))

    attn = F.softmax(attn, dim=-1)

    # Attend to values
    y = torch.matmul(attn, v)  # (B, H, Tq, D)
    return y.transpose(1, 2)   # (B, Tq, H, D)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), poly_beta=None):
    """
    Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.
        poly_beta: optional float for polynomial attention

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    return _attention(q, k, v, causal=causal, window_size=window_size, poly_beta=poly_beta)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1), poly_beta=None):
    """
    Attention with KV cache for inference.

    Updates k_cache/v_cache in-place, then runs attention against full cache.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.
        poly_beta: optional float for polynomial attention

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    T_new = q.size(1)
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache in-place
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Slice full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    return _attention(q, k_full, v_full, causal=causal, window_size=window_size, poly_beta=poly_beta)


# Export: same namespace interface as before
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
