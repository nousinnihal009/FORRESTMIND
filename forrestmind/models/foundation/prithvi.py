"""
Prithvi-EO-2.0 adapter for ForestMind.

Wraps the NASA/IBM Prithvi-EO-2.0 foundation model (available on HuggingFace)
as a ForestMind-compatible encoder. Prithvi was trained on 4.2M global
time-series samples from Harmonized Landsat/Sentinel-2 data.

Requires:
  - GPU with ≥16GB VRAM
  - pip install transformers
  - Model download from HuggingFace

Reference:
  Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for
  Earth Observation Applications (IEEE TGRS 2025)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor, nn

from forestmind.models.foundation.base import FoundationModelBase

logger = logging.getLogger(__name__)


class PrithviAdapter(FoundationModelBase):
    """Adapter wrapping Prithvi-EO-2.0 as a ForestMind encoder.
    
    Uses the pretrained ViT backbone from Prithvi, extracting the [CLS]
    token embedding. Includes a projection head to map Prithvi's native
    embedding dimension to ForestMind's target dimension.
    
    Args:
        model_name: HuggingFace model identifier.
        target_dim: Target embedding dimension for ForestMind.
        freeze_backbone: Whether to freeze the pretrained weights.
        in_channels: Number of input channels to adapt.
    """

    def __init__(
        self,
        model_name: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        target_dim: int = 256,
        freeze_backbone: bool = True,
        in_channels: int = 14,
    ):
        super().__init__()
        self._target_dim = target_dim
        self._model_name = model_name

        try:
            from transformers import AutoModel
            logger.info(f"Loading Prithvi model: {model_name}")
            self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self._native_dim = self.backbone.config.hidden_size
        except ImportError:
            raise ImportError(
                "Prithvi adapter requires the 'transformers' package. "
                "Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Prithvi model '{model_name}'. "
                f"Ensure you have internet access and the model exists. Error: {e}"
            )

        # Channel adapter: map our 14-channel input to Prithvi's expected channels
        # Prithvi expects 6 HLS bands; we project from 14 to 6
        self.channel_adapter = nn.Conv2d(in_channels, 6, kernel_size=1, bias=False)

        # Projection head: map Prithvi embedding dim → ForestMind target dim
        self.projection = nn.Sequential(
            nn.Linear(self._native_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.GELU(),
        )

        if freeze_backbone:
            self.freeze_backbone_only()

    def freeze_backbone_only(self) -> None:
        """Freeze only the pretrained backbone, keep adapters trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Keep channel adapter and projection trainable
        for param in self.channel_adapter.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True

    def encode(self, x: Tensor) -> Tensor:
        """Encode using Prithvi backbone.
        
        Args:
            x: (B, C, H, W) multi-band satellite patch.
            
        Returns:
            (B, D) embedding vector.
        """
        # Adapt channels
        x = self.channel_adapter(x)  # (B, 6, H, W)

        # Forward through Prithvi
        outputs = self.backbone(x)

        # Extract [CLS] token or global pool
        if hasattr(outputs, "last_hidden_state"):
            embedding = outputs.last_hidden_state[:, 0]  # [CLS] token
        elif hasattr(outputs, "pooler_output"):
            embedding = outputs.pooler_output
        else:
            # Fallback: mean pool
            embedding = outputs.mean(dim=1) if outputs.dim() == 3 else outputs

        # Project to target dimension
        return self.projection(embedding)

    @property
    def embedding_dim(self) -> int:
        return self._target_dim

    @property
    def supports_modalities(self) -> list[str]:
        return ["optical"]  # Prithvi is primarily optical (HLS)
