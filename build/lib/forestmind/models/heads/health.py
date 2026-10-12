"""
Forest Health prediction head.

Predicts both:
  1. Discrete health state (5-class classification):
     0=Healthy, 1=Stressed, 2=Degraded, 3=Severely Degraded, 4=Cleared
  2. Continuous health score (0-100)

The dual output provides both actionable categories and fine-grained
assessment for tracking gradual changes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ForestHealthHead(nn.Module):
    """Predicts forest health state and continuous score.
    
    Args:
        embed_dim: Input embedding dimension.
        num_classes: Number of discrete health states.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        predict_continuous: Also predict continuous 0-100 score.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        predict_continuous: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.predict_continuous = predict_continuous

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classification head (discrete state)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Regression head (continuous score)
        if predict_continuous:
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),  # outputs [0, 1], multiply by 100 for 0-100 scale
            )

    def forward(self, embedding: Tensor) -> dict[str, Tensor]:
        """Predict health state and score.
        
        Args:
            embedding: (B, D) summary embedding from temporal transformer.
            
        Returns:
            Dictionary with:
              logits: (B, 5) class logits
              probs: (B, 5) class probabilities
              state: (B,) predicted class indices
              score: (B,) continuous score [0, 100] (if predict_continuous)
        """
        features = self.shared(embedding)

        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=-1)
        state = torch.argmax(probs, dim=-1)

        result = {
            "logits": logits,
            "probs": probs,
            "state": state,
        }

        if self.predict_continuous:
            score = self.regressor(features).squeeze(-1) * 100.0
            result["score"] = score

        return result
