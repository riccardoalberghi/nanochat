"""
Test pure PyTorch attention implementation.

Run: python -m pytest tests/test_attention_fallback.py -v -s
"""
import torch
import pytest
from nanochat.flash_attention import flash_attn
from nanochat.engine import KVCache


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


class TestAttention:
    """Test the pure PyTorch attention implementation."""

    def test_basic_causal(self):
        """Basic causal attention produces valid output."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"

    def test_full_context(self):
        """Full context (window_size=-1)."""
        B, T, H, D = 2, 128, 4, 32
        q = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any()

    def test_sliding_window(self):
        """Sliding window restricts attention to nearby tokens."""
        B, T, H, D = 2, 128, 4, 32
        window = 32
        q = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any()

    def test_gqa(self):
        """Group Query Attention (fewer KV heads than Q heads)."""
        B, T, D = 2, 64, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, T, n_heads, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, T, n_kv_heads, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, T, n_kv_heads, D, device=DEVICE, dtype=DTYPE)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        assert y.shape == (B, T, n_heads, D)
        assert not torch.isnan(y).any()

    def test_backward(self):
        """Test gradients flow through attention."""
        B, T, H, D = 2, 32, 4, 16
        q = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"

    def test_causal_mask_correctness(self):
        """Verify causal mask: token i should not attend to token j > i."""
        B, T, H, D = 1, 8, 1, 4
        # Use one-hot values so output reveals which positions were attended
        q = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.float32)
        # Make v distinct per position
        v = torch.eye(T, D, device=DEVICE, dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # (1, T, 1, D)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        # First token can only attend to itself, so output should be v[0]
        # (weighted by softmax of a single element = 1.0)
        assert torch.allclose(y[0, 0, 0], v[0, 0, 0], atol=1e-5)


class TestKVCache:
    """Test attention with KV cache for inference."""

    def test_prefill(self):
        """Test prefill (inserting multiple tokens into empty cache)."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16
        n_layers = 1

        cache = KVCache(
            batch_size=B, num_heads=H, seq_len=T_max, head_dim=D,
            num_layers=n_layers, device=DEVICE, dtype=DTYPE
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        q = torch.randn(B, T_prefill, H, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=DEVICE, dtype=DTYPE)

        y = flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(T_prefill)

        assert y.shape == (B, T_prefill, H, D)
        assert cache.get_pos() == T_prefill
        assert not torch.isnan(y).any()

    def test_single_token_decode(self):
        """Test single token generation after prefill."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16
        n_layers = 1

        cache = KVCache(
            batch_size=B, num_heads=H, seq_len=T_max, head_dim=D,
            num_layers=n_layers, device=DEVICE, dtype=DTYPE
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        # Prefill
        q = torch.randn(B, T_prefill, H, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=DEVICE, dtype=DTYPE)
        flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(T_prefill)

        # Decode single token
        q_single = torch.randn(B, 1, H, D, device=DEVICE, dtype=DTYPE)
        k_single = torch.randn(B, 1, H, D, device=DEVICE, dtype=DTYPE)
        v_single = torch.randn(B, 1, H, D, device=DEVICE, dtype=DTYPE)

        y = flash_attn.flash_attn_with_kvcache(
            q_single, k_cache, v_cache, k=k_single, v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(1)

        assert y.shape == (B, 1, H, D)
        assert cache.get_pos() == T_prefill + 1
        assert not torch.isnan(y).any()

    def test_sliding_window_decode(self):
        """Test single token decode with sliding window."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 32
        window = 8
        n_layers = 1

        cache = KVCache(
            batch_size=B, num_heads=H, seq_len=T_max, head_dim=D,
            num_layers=n_layers, device=DEVICE, dtype=DTYPE
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        # Prefill
        q = torch.randn(B, T_prefill, H, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=DEVICE, dtype=DTYPE)
        flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(window, 0)
        )
        cache.advance(T_prefill)

        # Decode with window
        q_single = torch.randn(B, 1, H, D, device=DEVICE, dtype=DTYPE)
        k_single = torch.randn(B, 1, H, D, device=DEVICE, dtype=DTYPE)
        v_single = torch.randn(B, 1, H, D, device=DEVICE, dtype=DTYPE)

        y = flash_attn.flash_attn_with_kvcache(
            q_single, k_cache, v_cache, k=k_single, v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(window, 0)
        )

        assert y.shape == (B, 1, H, D)
        assert not torch.isnan(y).any()


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print()
    pytest.main([__file__, "-v", "-s"])
