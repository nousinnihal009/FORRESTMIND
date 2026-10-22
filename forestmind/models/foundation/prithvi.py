"""Prithvi encoder integration using the official TerraTorch backbone registry.

Pretrained mode requires terratorch, cached/downloadable weights, and inputs
normalized for that checkpoint. Loading failures are explicit. Set use_proxy=True
only for an untrained CNN development baseline; it is not the pretrained model.
Projection and task heads still require supervised training even with a pretrained
encoder. Model memory requirements depend on the selected registry architecture.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor, nn

from forestmind.models.foundation.base import FoundationModelBase
from forestmind.models.foundation.native import validate_native_architecture

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
        model_name: TerraTorch backbone registry identifier.
        target_dim: Target embedding dimension for ForestMind.
        freeze_backbone: Whether to freeze the pretrained weights.
        in_channels: Number of input channels to adapt.
        use_proxy: Explicitly request an untrained CNN development baseline.
    """

    def __init__(
        self,
        model_name: str = "prithvi_eo_v2_300",
        target_dim: int = 256,
        freeze_backbone: bool = True,
        in_channels: int = 14,
        use_proxy: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        validate_native_architecture("prithvi", model_name, pretrained=pretrained, use_proxy=use_proxy)
        self._target_dim = target_dim
        self._model_name = model_name
        self._in_channels = in_channels
        # A randomly initialized CNN is a development baseline, never a
        # substitute for a requested pretrained model.
        self._is_proxy = use_proxy

        if self._is_proxy:
            self._init_proxy(target_dim)
        else:
            self._init_full(model_name, target_dim, freeze_backbone, pretrained)

    def _init_full(self, model_name: str, target_dim: int, freeze_backbone: bool,
                   pretrained: bool) -> None:
        """Build the native encoder; offline loading constructs its architecture only."""
        try:
            from terratorch.registry import BACKBONE_REGISTRY
            self.backbone = BACKBONE_REGISTRY.build(
                model_name, pretrained=pretrained, num_frames=1,
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
                f"Could not construct native Prithvi '{model_name}' (pretrained={pretrained}). No proxy was substituted. "
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
