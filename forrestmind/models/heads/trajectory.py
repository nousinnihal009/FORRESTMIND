"""
Future Trajectory prediction head.

Predicts forest health scores at multiple future horizons:
  - 30 days
  - 6 months (180 days)
  - 1 year (365 days)
  - 3 years (1095 days)
  - 5 years (1825 days)

Uses the full temporal sequence (not just the summary) to capture
degradation momentum and trend information for extrapolation.
Optionally predicts uncertainty (aleatoric) via a heteroscedastic
output (predicting both mean and variance).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TrajectoryHead(nn.Module):
    """Multi-horizon future health trajectory prediction.
    
    Takes the full temporal sequence from the transformer and predicts
    future health scores. Uses an attention-based aggregation over the
    sequence to weight recent vs historical observations.
    
    Args:
        embed_dim: Input embedding dimension.
        num_horizons: Number of future prediction horizons.
        horizon_days: List of horizon offsets in days.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        predict_uncertainty: Predict per-horizon uncertainty.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_horizons: int = 5,
        horizon_days: list[int] | None = None,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        predict_uncertainty: bool = True,
    ):
        super().__init__()
        self.num_horizons = num_horizons
        self.horizon_days = horizon_days or [30, 180, 365, 1095, 1825]
        self.predict_uncertainty = predict_uncertainty

        # Temporal attention: learn which timesteps matter for prediction
        self.temporal_attention = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Horizon embedding: encode the prediction horizon
        self.horizon_embed = nn.Embedding(num_horizons, embed_dim)

        # Prediction MLP (shared trunk + per-horizon output)
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # concatenate summary + horizon embedding
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Mean prediction
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # [0, 1] health score
        )

        # Uncertainty prediction (log variance)
        if predict_uncertainty:
            self.logvar_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

    def _temporal_pool(self, sequence: Tensor) -> Tensor:
        """Attention-weighted temporal pooling.
        
        Args:
            sequence: (B, T, D) temporal sequence.
            
        Returns:
            (B, D) weighted summary.
        """
        # Compute attention weights
        weights = self.temporal_attention(sequence)  # (B, T, 1)
        weights = torch.softmax(weights, dim=1)

        # Weighted sum
        pooled = (sequence * weights).sum(dim=1)  # (B, D)
        return pooled

    def forward(self, sequence: Tensor, summary: Tensor) -> dict[str, Tensor]:
        """Predict future health trajectory.
        
        Args:
            sequence: (B, T, D) full temporal sequence.
            summary: (B, D) summary embedding (last timestep).
            
        Returns:
            Dictionary with:
              mean: (B, 5) predicted health scores [0, 1] at each horizon
              score_100: (B, 5) predicted scores scaled to [0, 100]
              uncertainty: (B, 5) prediction std dev (if predict_uncertainty)
        """
        # Attention-weighted temporal summary
        temporal_summary = self._temporal_pool(sequence)

        # Use both attention summary and last-step summary
        combined_summary = temporal_summary + summary  # residual combination

        # Predict at each horizon
        means = []
        logvars = []

        horizon_indices = torch.arange(
            self.num_horizons, device=sequence.device
        )
        horizon_embeds = self.horizon_embed(horizon_indices)  # (H, D)

        for h_idx in range(self.num_horizons):
            h_embed = horizon_embeds[h_idx].unsqueeze(0).expand(sequence.shape[0], -1)
            # Concatenate summary with horizon embedding
            h_input = torch.cat([combined_summary, h_embed], dim=-1)  # (B, 2D)

            features = self.trunk(h_input)
            mean = self.mean_head(features).squeeze(-1)  # (B,)
            means.append(mean)

            if self.predict_uncertainty:
                logvar = self.logvar_head(features).squeeze(-1)
                logvars.append(logvar)

        means = torch.stack(means, dim=-1)  # (B, H)

        result = {
            "mean": means,
            "score_100": means * 100.0,
        }

        if self.predict_uncertainty:
            logvars = torch.stack(logvars, dim=-1)
            result["logvar"] = logvars
            result["uncertainty"] = torch.exp(0.5 * logvars)  # std dev

        return result
