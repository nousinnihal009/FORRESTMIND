"""
Custom Vision Transformer encoder for Earth Observation data.

A lightweight ViT (~5M params) designed for CPU-friendly development.
Handles multi-spectral (12-band optical + 2-channel SAR = 14 channels)
input with patch embedding, positional encoding, and [CLS] token pooling.

This serves as the default foundation model when GPU-heavy pretrained
models (Prithvi, TerraMind) are not available.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from forestmind.models.foundation.base import FoundationModelBase


class PatchEmbedding(nn.Module):
    """Convert multi-band image patches into token sequences.
    
    Splits a (B, C, H, W) image into non-overlapping patches of size P×P,
    then projects each patch to a D-dimensional embedding via a linear layer.
    
    Output: (B, N, D) where N = (H/P) × (W/P)
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) → (B, D, H/P, W/P) → (B, N, D)
        x = self.proj(x)
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop_rate, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class CustomViTEncoder(FoundationModelBase):
    """Lightweight Vision Transformer for multi-spectral EO data.
    
    Architecture:
      1. Patch embedding: (B, 14, 128, 128) → (B, 64, D) for P=16
      2. Prepend [CLS] token → (B, 65, D)
      3. Add learnable positional encoding
      4. N transformer blocks
      5. [CLS] token output → (B, D)
      
    Args:
        in_channels: Number of input channels (optical + SAR).
        embed_dim: Embedding dimension (D).
        patch_size: Size of image patches (P).
        depth: Number of transformer layers.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim ratio.
        drop_rate: Dropout rate.
        img_size: Expected input image size (for positional encoding).
    """

    def __init__(
        self,
        in_channels: int = 14,  # 12 optical + 2 SAR
        embed_dim: int = 256,
        patch_size: int = 16,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        img_size: int = 128,
    ):
        super().__init__()
        self._embedding_dim = embed_dim
        self._in_channels = in_channels

        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)

        # Number of patches
        num_patches = (img_size // patch_size) ** 2

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, x: Tensor) -> Tensor:
        """Encode multi-band patch to embedding.
        
        Args:
            x: (B, C, H, W) multi-band satellite patch.
            
        Returns:
            (B, D) embedding vector.
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add positional encoding
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Extract [CLS] token
        x = self.norm(x[:, 0])  # (B, D)
        return x

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def supports_modalities(self) -> list[str]:
        return ["optical", "sar"]
