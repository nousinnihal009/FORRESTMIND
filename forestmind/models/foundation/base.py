"""
Abstract base class for Earth Observation foundation models.

All foundation models in ForestMind implement this interface, enabling
seamless swapping between:
  - CustomViTEncoder (lightweight, CPU-friendly, ~5M params)
  - PrithviAdapter (NASA/IBM Prithvi-EO-2.0, requires GPU)
  - TerraMindAdapter (multimodal EO foundation model, requires GPU)

The key contract: encode() takes a single-timestep multi-band image patch
and returns a fixed-dimensional embedding vector (spatially pooled).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class FoundationModelBase(ABC, nn.Module):
    """Abstract interface for EO foundation model encoders.
    
    Foundation models serve as frozen or fine-tunable feature extractors.
    They encode a single-timestep multi-modal satellite observation into
    a dense embedding vector.
    
    Input:  (B, C, H, W) — a batch of single-timestep patches
    Output: (B, D) — embedding vectors, where D = embedding_dim
    """

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """Encode a single-timestep multi-band patch into an embedding.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
               C includes both optical bands and SAR channels.
               
        Returns:
            Embedding tensor of shape (B, D).
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding."""
        ...

    @property
    @abstractmethod
    def supports_modalities(self) -> list[str]:
        """List of supported input modalities.
        
        Examples: ["optical"], ["optical", "sar"], ["optical", "sar", "dem"]
        """
        ...

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self) -> None:
        """Freeze all parameters (for use as feature extractor)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters (for fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass delegates to encode()."""
        return self.encode(x)
