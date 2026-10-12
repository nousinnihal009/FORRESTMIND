"""
Multimodal fusion module for ForestMind.

Fuses foundation model embeddings with auxiliary context streams
(climate stress indicators, human activity signals) using cross-attention
or gated mechanisms.

The fusion module operates on per-timestep embeddings and enriches them
with domain-specific context before temporal modeling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossAttentionFusion(nn.Module):
    """Cross-attention based multimodal fusion.
    
    Takes the primary embedding (from the foundation model) and attends
    to auxiliary context streams (climate, human activity) to produce
    an enriched embedding.
    
    Architecture:
      primary_embed → Q
      context_embeds → K, V
      cross_attention(Q, K, V) → enriched_embed
      
    Args:
        embed_dim: Primary embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of cross-attention layers.
        dropout: Dropout rate.
        num_context_streams: Number of auxiliary context streams.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_context_streams: int = 2,  # climate + human activity
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_context_streams = num_context_streams

        # Context projection layers (one per stream)
        self.context_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
            for _ in range(num_context_streams)
        ])

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_q": nn.LayerNorm(embed_dim),
                "norm_kv": nn.LayerNorm(embed_dim),
                "cross_attn": nn.MultiheadAttention(
                    embed_dim, num_heads, dropout=dropout, batch_first=True,
                ),
                "norm_ff": nn.LayerNorm(embed_dim),
                "feedforward": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(num_layers)
        ])

    def forward(self, primary: Tensor, contexts: list[Tensor] | None = None) -> Tensor:
        """Fuse primary embedding with context streams.
        
        Args:
            primary: (B, D) primary embedding from foundation model.
            contexts: List of (B, D) context embeddings. If None or empty,
                      returns primary unchanged.
                      
        Returns:
            (B, D) fused embedding.
        """
        if contexts is None or len(contexts) == 0:
            return primary

        # Project context streams
        projected_contexts = []
        for i, ctx in enumerate(contexts):
            if i < len(self.context_projections):
                projected_contexts.append(self.context_projections[i](ctx))

        if not projected_contexts:
            return primary

        # Stack contexts as key/value sequence: (B, num_contexts, D)
        context_seq = torch.stack(projected_contexts, dim=1)

        # Primary as query: (B, 1, D)
        x = primary.unsqueeze(1)

        # Cross-attention layers
        for layer in self.cross_attn_layers:
            # Cross-attend to context
            q = layer["norm_q"](x)
            kv = layer["norm_kv"](context_seq)
            attn_out, _ = layer["cross_attn"](q, kv, kv)
            x = x + attn_out

            # Feedforward
            x = x + layer["feedforward"](layer["norm_ff"](x))

        return x.squeeze(1)  # (B, D)


class GatedFusion(nn.Module):
    """Gated fusion — simpler alternative to cross-attention.
    
    Each context stream is gated and added to the primary embedding.
    Gate values are learned via sigmoid-activated projections.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_context_streams: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            for _ in range(num_context_streams)
        ])
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_context_streams)
        ])
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, primary: Tensor, contexts: list[Tensor] | None = None) -> Tensor:
        if contexts is None or len(contexts) == 0:
            return primary

        x = primary
        for i, ctx in enumerate(contexts):
            if i < len(self.gates):
                gate = self.gates[i](torch.cat([x, ctx], dim=-1))
                projected = self.projections[i](ctx)
                x = x + gate * projected

        return self.output_norm(x)


class ConcatFusion(nn.Module):
    """Simple concatenation fusion with linear projection.
    
    Concatenates all embeddings and projects back to embed_dim.
    Simplest baseline fusion method.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_context_streams: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        total_dim = embed_dim * (1 + num_context_streams)
        self.projection = nn.Sequential(
            nn.Linear(total_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, primary: Tensor, contexts: list[Tensor] | None = None) -> Tensor:
        if contexts is None or len(contexts) == 0:
            return primary

        # Pad if fewer contexts than expected
        all_embeds = [primary]
        for ctx in contexts:
            all_embeds.append(ctx)
        while len(all_embeds) < 3:  # primary + 2 contexts
            all_embeds.append(torch.zeros_like(primary))

        concatenated = torch.cat(all_embeds, dim=-1)
        return self.projection(concatenated)


class MultimodalFusion(nn.Module):
    """Factory wrapper that selects the fusion method.
    
    Also includes lightweight encoders for deriving climate stress
    and human activity context from the primary embedding.
    These could later be replaced with actual external data feeds.
    
    Args:
        embed_dim: Embedding dimension.
        method: Fusion method ("cross_attention", "gated", "concat").
        num_heads: Attention heads (for cross_attention).
        num_layers: Number of layers (for cross_attention).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        method: str = "cross_attention",
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Context stream encoders
        # These derive climate and human-activity signals from the primary embedding.
        # In a full system, these would ingest separate data sources.
        self.climate_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.activity_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Fusion module
        if method == "cross_attention":
            self.fusion = CrossAttentionFusion(
                embed_dim, num_heads, num_layers, dropout, num_context_streams=2,
            )
        elif method == "gated":
            self.fusion = GatedFusion(embed_dim, num_context_streams=2, dropout=dropout)
        elif method == "concat":
            self.fusion = ConcatFusion(embed_dim, num_context_streams=2, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def forward(
        self, embedding: Tensor, climate_context: Tensor | None = None,
        activity_context: Tensor | None = None,
    ) -> Tensor:
        """Produce fused embedding from primary foundation model output.
        
        Args:
            embedding: (B, D) primary embedding.
            
        Returns:
            (B, D) fused embedding enriched with context.
        """
        if climate_context is None and activity_context is None:
            return embedding
        # Context must be independently supplied; transformed image features
        # are not observations of climate or human activity.
        contexts = []
        for context, encoder in ((climate_context, self.climate_encoder),
                                 (activity_context, self.activity_encoder)):
            if context is not None and context.shape != embedding.shape:
                raise ValueError("Context streams must match the primary (B, D) embedding")
            contexts.append(encoder(context) if context is not None else torch.zeros_like(embedding))
        return self.fusion(embedding, contexts)
