"""
TerraMind adapter for ForestMind.

Wraps the TerraMind multimodal EO foundation model as a ForestMind-compatible
encoder. TerraMind natively supports optical, SAR, and DEM modalities,
making it ideal for our multimodal pipeline.

Requires:
  - GPU with ≥16GB VRAM
  - Model checkpoint download
  
Reference:
  TerraMind: Large-Scale Generative Multimodality for Earth Observation (arXiv 2024)

Note: This is a placeholder adapter. The actual TerraMind model integration
will depend on the specific release format and API. The structure below
follows the expected interface pattern.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from forestmind.models.foundation.base import FoundationModelBase

logger = logging.getLogger(__name__)


class TerraMindAdapter(FoundationModelBase):
    """Adapter wrapping TerraMind as a ForestMind encoder.
    
    TerraMind's multimodal architecture accepts optical + SAR simultaneously,
    which aligns with our pipeline. This adapter handles:
    1. Splitting our concatenated input into optical and SAR streams
    2. Forwarding through TerraMind's multimodal encoder
    3. Projecting the output embedding to ForestMind's target dimension
    
    Args:
        model_name: Model identifier or checkpoint path.
        target_dim: Target embedding dimension.
        freeze_backbone: Whether to freeze pretrained weights.
        optical_channels: Number of optical input channels.
        sar_channels: Number of SAR input channels.
    """

    def __init__(
        self,
        model_name: str = "terramind-base",
        target_dim: int = 256,
        freeze_backbone: bool = True,
        optical_channels: int = 12,
        sar_channels: int = 2,
    ):
        super().__init__()
        self._target_dim = target_dim
        self._optical_channels = optical_channels
        self._sar_channels = sar_channels

        logger.warning(
            "TerraMind adapter is in placeholder mode. "
            "The actual model integration requires the TerraMind checkpoint. "
            "Using a proxy encoder for architecture validation."
        )

        # Proxy encoder mimicking TerraMind's multimodal architecture
        # This will be replaced with the actual TerraMind model when available
        self._native_dim = 512

        # Separate encoders for each modality (proxy)
        self.optical_encoder = nn.Sequential(
            nn.Conv2d(optical_channels, 64, 7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, 32, 7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Cross-modal fusion (proxy for TerraMind's internal fusion)
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, self._native_dim),
            nn.LayerNorm(self._native_dim),
            nn.GELU(),
        )

        # Projection to target dim
        self.projection = nn.Sequential(
            nn.Linear(self._native_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.GELU(),
        )

        if freeze_backbone:
            logger.info("Note: freeze_backbone has no effect in placeholder mode")

    def encode(self, x: Tensor) -> Tensor:
        """Encode multimodal input.
        
        Args:
            x: (B, C, H, W) where C = optical_channels + sar_channels.
            
        Returns:
            (B, D) embedding vector.
        """
        # Split into modalities
        optical = x[:, :self._optical_channels]
        sar = x[:, self._optical_channels:]

        # Encode each modality
        opt_emb = self.optical_encoder(optical)  # (B, 256)
        sar_emb = self.sar_encoder(sar)  # (B, 128)

        # Fuse
        fused = self.fusion(torch.cat([opt_emb, sar_emb], dim=-1))

        # Project
        return self.projection(fused)

    @property
    def embedding_dim(self) -> int:
        return self._target_dim

    @property
    def supports_modalities(self) -> list[str]:
        return ["optical", "sar", "dem"]
