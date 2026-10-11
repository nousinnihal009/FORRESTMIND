"""
Temporal Transformer for ForestMind.

Processes sequences of per-timestep embeddings to capture temporal dynamics
of forest degradation. Key innovation: uses continuous temporal positional
encoding (sine/cosine with actual day offsets) rather than integer positions,
because satellite revisit intervals are irregular.

Input:  (B, T, D) — sequence of T timestep embeddings
Output: (B, T, D) — contextualized temporal embeddings
        (B, D)    — summary embedding (last timestep or pooled)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ContinuousTemporalPE(nn.Module):
    """Continuous temporal positional encoding.
    
    Unlike standard integer positional encoding, this module encodes
    actual timestamps (in days) using learnable frequency bands.
    This correctly handles:
    - Irregular revisit intervals
    - Missing observations
    - Variable-length sequences
    
    Encoding: PE(t) = concat([sin(ω_i * t), cos(ω_i * t)] for i in 1..D/2)
    where ω_i are learnable frequency parameters initialized log-uniformly.
    
    Args:
        embed_dim: Dimension of the embedding (must be even).
        max_period: Maximum period in days for the lowest frequency.
    """

    def __init__(self, embed_dim: int = 256, max_period: float = 3650.0):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos encoding"

        half_dim = embed_dim // 2
        # Initialize frequencies log-uniformly from daily to max_period
        # This gives a spectrum from high-frequency (daily) to low-frequency (multi-year)
        log_freqs = torch.linspace(
            math.log(1.0),  # daily frequency
            math.log(1.0 / max_period),  # multi-year frequency
            half_dim,
        )
        self.frequencies = nn.Parameter(log_freqs, requires_grad=True)
        self.phase = nn.Parameter(torch.zeros(half_dim), requires_grad=True)

    def forward(self, timestamps: Tensor) -> Tensor:
        """Encode timestamps into positional embeddings.
        
        Args:
            timestamps: (B, T) tensor of day offsets.
            
        Returns:
            (B, T, D) positional embeddings.
        """
        # timestamps: (B, T) → (B, T, 1)
        t = timestamps.unsqueeze(-1).float()

        # Compute angular frequencies
        omega = torch.exp(self.frequencies)  # (D/2,)

        # Compute phase: ω * t + φ
        angles = t * omega.unsqueeze(0).unsqueeze(0) + self.phase.unsqueeze(0).unsqueeze(0)

        # Sine and cosine encoding
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return pe


class TemporalTransformerBlock(nn.Module):
    """Transformer encoder block for temporal modeling.
    
    Uses pre-norm with causal attention masking option.
    Causal masking ensures predictions only use past context.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, x: Tensor, observation_mask: Tensor | None = None) -> Tensor:
        # Self-attention with optional causal mask
        h = self.norm1(x)
        attn_mask = None
        if self.causal:
            attn_mask = self._get_causal_mask(x.shape[1], x.device)
        padding_mask = ~observation_mask if observation_mask is not None else None
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=padding_mask,
                                need_weights=False)
        x = x + attn_out

        # Feedforward
        x = x + self.ff(self.norm2(x))
        return x


class TemporalTransformer(nn.Module):
    """Full temporal transformer for processing embedding sequences.
    
    Architecture:
      1. Add continuous temporal positional encoding
      2. N transformer blocks (optionally causal)
      3. Output: full sequence + summary embedding
      
    The summary embedding is the last timestep's contextualized representation,
    which captures the current state informed by the full history.
    
    Args:
        embed_dim: Input/output embedding dimension.
        depth: Number of transformer layers.
        num_heads: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length (for buffer allocation).
        use_continuous_pe: Use continuous (vs integer) positional encoding.
        causal: Use causal attention masking.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 48,
        use_continuous_pe: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_continuous_pe = use_continuous_pe

        # Positional encoding
        if use_continuous_pe:
            self.pe = ContinuousTemporalPE(embed_dim)
        else:
            # Standard learnable positional encoding
            self.pe = nn.Embedding(max_seq_len, embed_dim)

        # Input projection + dropout
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(embed_dim, num_heads, dim_feedforward, dropout, causal)
            for _ in range(depth)
        ])

        # Final norm
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        embeddings: Tensor,
        timestamps: Tensor | None = None,
        observation_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Process temporal sequence of embeddings.
        
        Args:
            embeddings: (B, T, D) sequence of per-timestep embeddings.
            timestamps: (B, T) day offsets for continuous PE.
                       If None and using continuous PE, uses integer positions.
                       
        Returns:
            sequence: (B, T, D) contextualized sequence.
            summary: (B, D) summary embedding from last timestep.
        """
        B, T, D = embeddings.shape
        if T == 0:
            raise ValueError("Temporal sequence cannot be empty")
        if observation_mask is None:
            observation_mask = torch.ones(B, T, device=embeddings.device, dtype=torch.bool)
        if observation_mask.shape != (B, T) or not observation_mask.any(dim=1).all():
            raise ValueError("observation_mask must be (B, T) with at least one observation per sample")
        observation_mask = observation_mask.bool()
        if timestamps is not None:
            if timestamps.shape != (B, T):
                raise ValueError("timestamps must have shape (B, T)")
            if not torch.isfinite(timestamps[observation_mask]).all():
                raise ValueError("Observed timestamps must be finite")
            for row, mask in zip(timestamps, observation_mask):
                observed = row[mask]
                if (observed[1:] < observed[:-1]).any():
                    raise ValueError("Observed timestamps must be chronological")
            timestamps = timestamps.masked_fill(~observation_mask, 0).float()
        embeddings = embeddings.masked_fill(~observation_mask[..., None], 0)

        # Add positional encoding
        if self.use_continuous_pe:
            if timestamps is None:
                # Fallback: use integer positions as "days"
                timestamps = torch.arange(T, device=embeddings.device).unsqueeze(0).expand(B, -1).float()
            pe = self.pe(timestamps)
        else:
            positions = torch.arange(T, device=embeddings.device)
            pe = self.pe(positions).unsqueeze(0).expand(B, -1, -1)

        x = self.input_norm(embeddings) + pe
        x = self.input_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, observation_mask)
            x = x.masked_fill(~observation_mask[..., None], 0)

        x = self.output_norm(x)

        # Summary: last timestep embedding
        last_observed = torch.arange(T, device=x.device).expand(B, -1).masked_fill(
            ~observation_mask, -1).max(dim=1).values
        summary = x[torch.arange(B, device=x.device), last_observed]

        return x, summary
