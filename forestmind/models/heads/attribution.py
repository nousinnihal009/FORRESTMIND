"""
Cause Attribution prediction head.

Multi-label classification predicting which degradation drivers are
active in the observed area. Drivers are not mutually exclusive —
an area can simultaneously experience logging AND drought.

Drivers:
  0: agriculture
  1: cattle_pasture
  2: mining
  3: logging
  4: road_construction
  5: wildfire
  6: drought
  7: urban_expansion
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CauseAttributionHead(nn.Module):
    """Multi-label classification of degradation drivers.
    
    Uses a shared trunk with per-driver attention mechanism to
    capture driver-specific signals from the embedding.
    
    Args:
        embed_dim: Input embedding dimension.
        num_drivers: Number of degradation driver classes.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        predict_strength: Also predict per-driver strength [0, 1].
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_drivers: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        predict_strength: bool = True,
    ):
        super().__init__()
        self.num_drivers = num_drivers
        self.predict_strength = predict_strength

        # Shared feature backbone
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-driver classifier (multi-label, each independent)
        self.driver_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for _ in range(num_drivers)
        ])

        # Per-driver strength regressor
        if predict_strength:
            self.driver_regressors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_drivers)
            ])

    def forward(self, embedding: Tensor) -> dict[str, Tensor]:
        """Predict active drivers and their strengths.
        
        Args:
            embedding: (B, D) summary embedding.
            
        Returns:
            Dictionary with:
              logits: (B, 8) raw logits per driver
              probs: (B, 8) sigmoid probabilities
              active: (B, 8) binary predictions (threshold=0.5)
              strength: (B, 8) per-driver strength [0, 1] (if predict_strength)
        """
        features = self.shared(embedding)

        # Per-driver binary classification
        logits = torch.cat(
            [clf(features) for clf in self.driver_classifiers], dim=-1
        )  # (B, num_drivers)

        probs = torch.sigmoid(logits)
        active = (probs > 0.5).float()

        result = {
            "logits": logits,
            "probs": probs,
            "active": active,
        }

        if self.predict_strength:
            strength = torch.cat(
                [reg(features) for reg in self.driver_regressors], dim=-1
            )  # (B, num_drivers)
            result["strength"] = strength

        return result
