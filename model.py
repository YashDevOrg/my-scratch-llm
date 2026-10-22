import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex Rotary Positional Embedding (RoPE) frequencies.

    Args:
        dim: Dimension per attention head (head_dim).
        end: Maximum sequence length.
        theta: Frequency base (default: 10000.0).

    Returns:
        torch.Tensor of complex numbers with shape (end, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (end, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis tensor to match the shape of x for broadcasting."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"{freqs_cis.shape} vs {(x.shape[1], x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Positional Embeddings to Query and Key tensors.

    Args:
        xq: Query tensor with shape (bsz, seqlen, n_heads, head_dim)
        xk: Key tensor with shape (bsz, seqlen, n_kv_heads, head_dim)
        freqs_cis: Complex rotary frequencies of shape (seqlen, head_dim // 2)

    Returns:
        Rotated (xq, xk) tensors matching their original input shapes and dtypes.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat Key/Value heads along head dimension when using Grouped-Query Attention (GQA)."""
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class CausalSelfAttention(nn.Module):
    """Grouped-Query Attention (GQA) with RoPE and PyTorch SDPA."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # Linear projections
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # Reshape to (bsz, seqlen, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply RoPE to query and key
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repeat KV heads for Grouped-Query Attention
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Transpose to (bsz, n_heads, seqlen, head_dim) for SDPA
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        # Transpose back and project to model dimension
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.resid_dropout(self.wo(out))


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network: w2(silu(w1(x)) * w3(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Standard LLaMA hidden dimension calculation rounded up to multiple_of
        hidden_dim = int(2 * (4 * cfg.dim) / 3)
        hidden_dim = cfg.multiple_of * ((hidden_dim + cfg.multiple_of - 1) // cfg.multiple_of)

        self.w1 = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer decoder block with Pre-Norm RMSNorm residual connections."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attention = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.feed_forward = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class MiniLlama(nn.Module):
    """MiniLlama ~50M parameter Decoder-Only Transformer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Tied token embeddings
        self.lm_head.weight = self.tok_embeddings.weight

        # Precompute RoPE complex frequencies and store as non-persistent buffer
        head_dim = cfg.dim // cfg.n_heads
        freqs_cis = precompute_freqs_cis(head_dim, cfg.max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Scaled initialization for residual projections (wo and w2)
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("w2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

        # Re-tie weights after initialization
        self.lm_head.weight = self.tok_embeddings.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights: normal(0, 0.02) for linear and embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seqlen = idx.shape
        assert seqlen <= self.cfg.max_seq_len, (
            f"Sequence length {seqlen} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        x = self.tok_embeddings(idx)
        freqs_cis = self.freqs_cis[:seqlen]

        for layer in self.layers:
            x = layer(x, freqs_cis)

        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        banned_token_ids: Optional[list] = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens.

        Args:
            banned_token_ids: Hard-ban list — token IDs forced to -inf at every
                step, making them physically impossible to generate.
        """
        # Pre-build banned token tensor once for efficiency
        banned_tensor: Optional[torch.Tensor] = None
        if banned_token_ids:
            banned_tensor = torch.tensor(
                [t for t in banned_token_ids if 0 <= t < self.cfg.vocab_size],
                dtype=torch.long,
                device=idx.device,
            )

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # Hard ban: force banned token logits to -inf before any other processing
            if banned_tensor is not None and len(banned_tensor) > 0:
                logits[:, banned_tensor] = -float("inf")

            # Apply repetition penalty to already generated/seen tokens
            if repetition_penalty != 1.0:
                for b in range(idx.shape[0]):
                    tokens_seen = torch.unique(idx[b])
                    token_logits = logits[b, tokens_seen]
                    token_logits = torch.where(
                        token_logits > 0,
                        token_logits / repetition_penalty,
                        token_logits * repetition_penalty,
                    )
                    logits[b, tokens_seen] = token_logits

            if not do_sample or temperature <= 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(1e-5, temperature)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    print("--- Running MiniLlama-50M Smoke Test ---")
    cfg = ModelConfig()
    model = MiniLlama(cfg)
    model.eval()

    # Trainable parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M ({total_params:,})")

    # Verify tied weights
    assert id(model.lm_head.weight) == id(model.tok_embeddings.weight), "Weights are not tied!"
    print("Weight tying verified: lm_head.weight <-> tok_embeddings.weight")

    # Dummy forward & backward pass
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    model.train()
    logits, loss = model(x, targets=y)
    expected_loss = math.log(cfg.vocab_size)
    print(f"Initial loss: {loss.item():.4f} (theoretical random loss: ln({cfg.vocab_size}) ~= {expected_loss:.4f})")

    loss.backward()
    assert model.tok_embeddings.weight.grad is not None, "tok_embeddings.weight gradient is None!"
    print(f"Gradient check passed: tok_embeddings grad norm = {model.tok_embeddings.weight.grad.norm().item():.4f}")

    # Autoregressive generation test
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=50)
    print(f"Generated sequence shape: {generated.shape} (input: 8 tokens, generated: 10 new tokens)")

    print("--- Smoke Test Passed Successfully! ---")
