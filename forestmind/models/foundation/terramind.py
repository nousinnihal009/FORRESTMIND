"""
TerraMind adapter for ForestMind.

Wraps the TerraMind multimodal EO foundation model as a ForestMind-compatible
encoder. TerraMind natively supports optical, SAR, and DEM modalities,
making it ideal for our multimodal pipeline.

TerraMind key properties:
  - Multimodal architecture: processes optical + SAR streams simultaneously
  - Cross-modal attention for joint representation learning
  - Generative pretraining on diverse EO data
  - Available as open-source checkpoint

Integration modes:
  1. FULL MODE: Load actual TerraMind checkpoint (GPU required)
  2. PROXY MODE: Lightweight multimodal proxy encoder (CPU-friendly)

Requires (full mode):
  - GPU with ≥16GB VRAM
  - pip install transformers safetensors
  - Model checkpoint download

Reference:
  TerraMind: Large-Scale Generative Multimodality for Earth Observation (arXiv 2024)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor, nn

from forestmind.models.foundation.base import FoundationModelBase

logger = logging.getLogger(__name__)


def _check_transformers_available() -> bool:
    """Check if HuggingFace transformers is available."""
    try:
        import transformers
        return True
    except ImportError:
        return False


class TerraMindAdapter(FoundationModelBase):
    """Adapter wrapping TerraMind as a ForestMind encoder.

    TerraMind's multimodal architecture accepts optical + SAR simultaneously,
    which aligns with our pipeline. This adapter handles:
      1. Splitting our concatenated input into optical and SAR streams
      2. Encoding each modality (separate encoders or unified backbone)
      3. Cross-modal fusion
      4. Projecting the output embedding to ForestMind's target dimension

    Operates in FULL mode (real weights) or PROXY mode (lightweight fallback).

    Args:
        model_name: Model identifier or HuggingFace checkpoint path.
        target_dim: Target embedding dimension.
        freeze_backbone: Whether to freeze pretrained weights.
        optical_channels: Number of optical input channels.
        sar_channels: Number of SAR input channels.
        use_proxy: Force proxy mode even if checkpoint is available.
    """

    def __init__(
        self,
        model_name: str = "terramind_v1_base",
        target_dim: int = 256,
        freeze_backbone: bool = True,
        optical_channels: int = 12,
        sar_channels: int = 2,
        use_proxy: bool = False,
    ):
        super().__init__()
        self._target_dim = target_dim
        self._model_name = model_name
        self._optical_channels = optical_channels
        self._sar_channels = sar_channels

        # Try to load actual checkpoint
        if not use_proxy:
            loaded = self._try_load_checkpoint(model_name, target_dim, freeze_backbone)
        else:
            loaded = False

        if use_proxy:
            self._init_proxy(target_dim, optical_channels, sar_channels)
        elif not loaded:
            raise RuntimeError(
                f"Could not load pretrained TerraMind '{model_name}'. No proxy was substituted. "
                "Install terratorch and use a supported registry model, or explicitly set "
                "use_proxy=true for an untrained development baseline."
            )

    def _try_load_checkpoint(
        self, model_name: str, target_dim: int, freeze_backbone: bool
    ) -> bool:
        """Attempt to load actual TerraMind checkpoint.

        Returns True if successful, False if fallback to proxy needed.
        """
        try:
            from terratorch.registry import BACKBONE_REGISTRY

            logger.info(f"Attempting to load TerraMind: {model_name}")

            self.backbone = BACKBONE_REGISTRY.build(
                model_name, pretrained=True, modalities=["S2L2A", "S1GRD"],
                merge_method="mean",
            )

            self._native_dim = self.backbone.out_channels[-1]
            self._is_proxy = False

            # Projection head
            self.projection = nn.Sequential(
                nn.Linear(self._native_dim, target_dim * 2),
                nn.LayerNorm(target_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(target_dim * 2, target_dim),
                nn.LayerNorm(target_dim),
            )

            if freeze_backbone:
                self._freeze_backbone_only()

            logger.info(
                f"TerraMind loaded: {model_name} "
                f"(native_dim={self._native_dim}, target_dim={target_dim})"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to load TerraMind '{model_name}': {e}")
            return False

    def _init_proxy(
        self, target_dim: int, optical_channels: int, sar_channels: int
    ) -> None:
        """Initialize multimodal proxy encoder."""
        self._is_proxy = True
        self._native_dim = 512

        logger.info(
            "TerraMind adapter: PROXY MODE "
            "(download the TerraMind checkpoint for full mode)"
        )

        # Modality-specific encoders
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
            nn.AdaptiveAvgPool2d(4),  # keep spatial info for cross-attn
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
            nn.AdaptiveAvgPool2d(4),
        )

        # Cross-modal attention fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.sar_proj = nn.Linear(128, 256)
        self.cross_norm = nn.LayerNorm(256)

        # Final fusion and pooling
        self.fusion = nn.Sequential(
            nn.Linear(256, self._native_dim),
            nn.LayerNorm(self._native_dim),
            nn.GELU(),
        )

        # Projection to target dim
        self.projection = nn.Sequential(
            nn.Linear(self._native_dim, target_dim * 2),
            nn.LayerNorm(target_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(target_dim * 2, target_dim),
            nn.LayerNorm(target_dim),
        )

    def _freeze_backbone_only(self) -> None:
        """Freeze backbone, keep adapters and projection trainable."""
        if hasattr(self, "backbone"):
            for param in self.backbone.parameters():
                param.requires_grad = False
        for module in [self.projection]:
            for param in module.parameters():
                param.requires_grad = True
        if hasattr(self, "optical_adapter"):
            for param in self.optical_adapter.parameters():
                param.requires_grad = True
            for param in self.sar_adapter.parameters():
                param.requires_grad = True
            for param in self.input_merger.parameters():
                param.requires_grad = True

    def encode(self, x: Tensor) -> Tensor:
        """Encode multimodal input.

        Args:
            x: (B, C, H, W) where C = optical_channels + sar_channels.

        Returns:
            (B, D) embedding vector.
        """
        if x.ndim != 4 or x.shape[1] != self._optical_channels + self._sar_channels:
            raise ValueError("TerraMind expects optical and SAR channels in a 4D tensor")
        # Split into modalities
        optical = x[:, :self._optical_channels]
        sar = x[:, self._optical_channels:self._optical_channels + self._sar_channels]

        if self._is_proxy:
            return self._encode_proxy(optical, sar)
        else:
            return self._encode_full(optical, sar)

    def _encode_proxy(self, optical: Tensor, sar: Tensor) -> Tensor:
        """Proxy mode encoding with cross-modal attention."""
        B = optical.shape[0]

        # Encode each modality
        opt_feat = self.optical_encoder(optical)  # (B, 256, 4, 4)
        sar_feat = self.sar_encoder(sar)           # (B, 128, 4, 4)

        # Reshape to sequence for cross-attention
        opt_seq = opt_feat.flatten(2).transpose(1, 2)  # (B, 16, 256)
        sar_seq = sar_feat.flatten(2).transpose(1, 2)  # (B, 16, 128)

        # Project SAR to same dim as optical
        sar_seq = self.sar_proj(sar_seq)  # (B, 16, 256)

        # Cross-modal attention: optical attends to SAR
        fused, _ = self.cross_attn(opt_seq, sar_seq, sar_seq)
        fused = self.cross_norm(opt_seq + fused)  # residual

        # Global pool over spatial tokens
        embedding = fused.mean(dim=1)  # (B, 256)

        # Fusion to native dim
        embedding = self.fusion(embedding)  # (B, 512)

        # Project to target
        return self.projection(embedding)

    def _encode_full(self, optical: Tensor, sar: Tensor) -> Tensor:
        """Full mode encoding with actual TerraMind backbone."""
        # Inputs must already use the checkpoint's per-modality normalization.
        outputs = self.backbone({"S2L2A": optical, "S1GRD": sar})
        if not isinstance(outputs, (list, tuple)) or outputs[-1].ndim != 3:
            raise RuntimeError("Unexpected TerraTorch TerraMind output: expected layer token tensors")
        # TerraMind has no CLS token; pool the final layer's spatial tokens.
        return self.projection(outputs[-1].mean(dim=1))

    def encode_modalities(self, optical: Tensor, sar: Tensor,
                          optical_mask: Tensor, sar_mask: Tensor) -> Tensor:
        """Forward only observed modality keys to a native pretrained backbone."""
        if self._is_proxy:
            return self._encode_proxy(optical, sar)
        result = optical.new_zeros(optical.shape[0], self._target_dim)
        for has_optical, has_sar in ((True, False), (False, True), (True, True)):
            selected = (optical_mask == has_optical) & (sar_mask == has_sar)
            if not selected.any():
                continue
            modalities = {}
            if has_optical:
                modalities["S2L2A"] = optical[selected]
            if has_sar:
                modalities["S1GRD"] = sar[selected]
            outputs = self.backbone(modalities)
            if not isinstance(outputs, (list, tuple)) or outputs[-1].ndim != 3:
                raise RuntimeError("Unexpected TerraTorch TerraMind output: expected layer token tensors")
            result[selected] = self.projection(outputs[-1].mean(dim=1))
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "backbone") and not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    @property
    def embedding_dim(self) -> int:
        return self._target_dim

    @property
    def supports_modalities(self) -> list[str]:
        return ["optical", "sar"]

    @property
    def is_proxy_mode(self) -> bool:
        """Whether the adapter is using the lightweight proxy encoder."""
        return self._is_proxy

    def __repr__(self) -> str:
        mode = "PROXY" if self._is_proxy else "FULL"
        return (
            f"TerraMindAdapter(mode={mode}, model='{self._model_name}', "
            f"native_dim={self._native_dim}, target_dim={self._target_dim})"
        )
