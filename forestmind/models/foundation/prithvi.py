"""
Prithvi-EO-2.0 adapter for ForestMind.

Wraps the NASA/IBM Prithvi-EO-2.0 foundation model (available on HuggingFace)
as a ForestMind-compatible encoder. Prithvi was trained on 4.2M global
time-series samples from Harmonized Landsat/Sentinel-2 data.

Prithvi-EO-2.0 key properties:
  - Trained on 6-band HLS data (Blue, Green, Red, NIR, SWIR1, SWIR2)
  - Supports multi-temporal input (multiple time steps natively)
  - ViT-based architecture with masked autoencoder pretraining
  - Available in 100M, 300M, and 600M variants on HuggingFace

Integration modes:
  1. FULL MODE: Load actual pretrained weights from HuggingFace (GPU required)
  2. PROXY MODE: Lightweight proxy encoder matching Prithvi's interface
     (CPU-friendly, used when transformers/GPU not available)

Requires (full mode):
  - GPU with ≥16GB VRAM
  - pip install transformers safetensors huggingface-hub
  - Internet access (first run only, then cached)

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

# Prithvi-EO-2.0 expected input: 6 HLS bands
# Blue(B02), Green(B03), Red(B04), NIR(B05/B08), SWIR1(B06/B11), SWIR2(B07/B12)
PRITHVI_HLS_BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
PRITHVI_NUM_BANDS = 6

# Mapping from our 14-channel input (12 optical + 2 SAR) to Prithvi's 6 bands
# Our band order: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12,VV,VH
# Prithvi wants: Blue(B02=idx1), Green(B03=idx2), Red(B04=idx3),
#                NIR(B8A=idx8), SWIR1(B11=idx10), SWIR2(B12=idx11)
OUR_TO_PRITHVI_INDICES = [1, 2, 3, 8, 10, 11]


def _check_transformers_available() -> bool:
    """Check if HuggingFace transformers is available."""
    try:
        import transformers
        return True
    except ImportError:
        return False


class PrithviAdapter(FoundationModelBase):
    """Adapter wrapping Prithvi-EO-2.0 as a ForestMind encoder.

    Operates in two modes:
      - FULL: Loads actual pretrained weights from HuggingFace
      - PROXY: Lightweight CNN encoder matching Prithvi's interface

    The adapter handles:
      1. Channel selection: 14-ch ForestMind input → 6-ch HLS bands
      2. Backbone encoding: 6-ch HLS → native embedding
      3. Projection: native embedding → ForestMind target_dim

    Args:
        model_name: HuggingFace model identifier.
        target_dim: Target embedding dimension for ForestMind.
        freeze_backbone: Whether to freeze the pretrained weights.
        in_channels: Number of input channels to adapt.
        use_proxy: Force proxy mode even if transformers is available.
    """

    def __init__(
        self,
        model_name: str = "prithvi_eo_v2_300",
        target_dim: int = 256,
        freeze_backbone: bool = True,
        in_channels: int = 14,
        use_proxy: bool = False,
    ):
        super().__init__()
        self._target_dim = target_dim
        self._model_name = model_name
        self._in_channels = in_channels
        # A randomly initialized CNN is a development baseline, never a
        # substitute for a requested pretrained model.
        self._is_proxy = use_proxy

        if self._is_proxy:
            self._init_proxy(target_dim)
        else:
            self._init_full(model_name, target_dim, freeze_backbone)

    def _init_full(self, model_name: str, target_dim: int, freeze_backbone: bool) -> None:
        """Initialize with actual Prithvi pretrained weights."""
        try:
            from terratorch.registry import BACKBONE_REGISTRY
            self.backbone = BACKBONE_REGISTRY.build(
                model_name, pretrained=True, num_frames=1,
                bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            )
            self._native_dim = self.backbone.out_channels[-1]

            self._is_proxy = False
            logger.info(
                f"Prithvi loaded: {model_name} "
                f"(native_dim={self._native_dim}, target_dim={target_dim})"
            )

        except Exception as e:
            raise RuntimeError(
                f"Could not load pretrained Prithvi '{model_name}'. No proxy was substituted. "
                "Install terratorch with a compatible checkpoint, or explicitly set use_proxy=true "
                "for an untrained development baseline."
            ) from e

        # Channel selector: pick the 6 HLS bands from our 14-channel input
        self.channel_indices = OUR_TO_PRITHVI_INDICES

        # Projection head: map Prithvi embedding dim → ForestMind target dim
        self.projection = nn.Sequential(
            nn.Linear(self._native_dim, target_dim * 2),
            nn.LayerNorm(target_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(target_dim * 2, target_dim),
            nn.LayerNorm(target_dim),
        )

        if freeze_backbone:
            self.freeze_backbone_only()

    def _init_proxy(self, target_dim: int) -> None:
        """Initialize lightweight proxy encoder (no pretrained weights)."""
        self._is_proxy = True
        self._native_dim = 384

        logger.info(
            "Prithvi adapter: PROXY MODE "
            "(untrained CNN baseline; not Prithvi pretrained weights)"
        )

        # Proxy encoder that mimics Prithvi's encoding pipeline
        self.channel_indices = OUR_TO_PRITHVI_INDICES

        self.proxy_encoder = nn.Sequential(
            # Stage 1: patch embedding equivalent
            nn.Conv2d(PRITHVI_NUM_BANDS, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Stage 2: deeper features
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Stage 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # Stage 4
            nn.Conv2d(256, self._native_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self._native_dim),
            nn.GELU(),
            # Global pool
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.projection = nn.Sequential(
            nn.Linear(self._native_dim, target_dim * 2),
            nn.LayerNorm(target_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(target_dim * 2, target_dim),
            nn.LayerNorm(target_dim),
        )

    def freeze_backbone_only(self) -> None:
        """Freeze only the pretrained backbone, keep adapters trainable."""
        if hasattr(self, "backbone"):
            for param in self.backbone.parameters():
                param.requires_grad = False
        # Keep projection trainable
        for param in self.projection.parameters():
            param.requires_grad = True

    def encode(self, x: Tensor) -> Tensor:
        """Encode using Prithvi backbone (or proxy).

        Args:
            x: (B, C, H, W) multi-band satellite patch.
               C = 14 (12 optical + 2 SAR) or C = 6 (pre-selected HLS bands).

        Returns:
            (B, D) embedding vector.
        """
        # Select HLS bands from our 14-channel input
        if x.ndim != 4 or x.shape[1] not in (6, 12, 14):
            raise ValueError("Prithvi expects 6 HLS bands or 12 optical (+ 2 SAR) channels")
        if x.shape[1] > PRITHVI_NUM_BANDS:
            x = x[:, self.channel_indices]  # (B, 6, H, W)

        if self._is_proxy:
            embedding = self.proxy_encoder(x)  # (B, native_dim)
        else:
            # TerraTorch expects (B, C, T, H, W), and returns layer token
            # sequences including CLS. Inputs use checkpoint normalization.
            outputs = self.backbone(x.unsqueeze(2))
            if not isinstance(outputs, (list, tuple)) or outputs[-1].ndim != 3:
                raise RuntimeError("Unexpected TerraTorch Prithvi output: expected layer token tensors")
            embedding = outputs[-1][:, 0]

        # Project to target dimension
        return self.projection(embedding)

    @property
    def embedding_dim(self) -> int:
        return self._target_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "backbone") and not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    @property
    def supports_modalities(self) -> list[str]:
        return ["optical"]  # Prithvi is primarily optical (HLS)

    @property
    def is_proxy_mode(self) -> bool:
        """Whether the adapter is using the lightweight proxy encoder."""
        return self._is_proxy

    def __repr__(self) -> str:
        mode = "PROXY" if self._is_proxy else "FULL"
        return (
            f"PrithviAdapter(mode={mode}, model='{self._model_name}', "
            f"native_dim={self._native_dim}, target_dim={self._target_dim})"
        )
