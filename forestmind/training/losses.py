"""
Multi-task loss functions for ForestMind.

Combines losses from all three prediction heads:
  1. Health: Cross-entropy (state) + MSE (continuous score)
  2. Attribution: Binary cross-entropy (multi-label drivers)
  3. Trajectory: MSE + Huber (future health prediction)

Optional: Heteroscedastic loss for trajectory uncertainty.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HealthLoss(nn.Module):
    """Combined classification + regression loss for forest health.
    
    L = w_ce * CrossEntropy(logits, state) + w_mse * MSE(score_pred, score_gt)
    """

    def __init__(self, ce_weight: float = 1.0, mse_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.mse_weight = mse_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: Tensor,
        score_pred: Tensor,
        state_gt: Tensor,
        score_gt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Args:
            logits: (B, 5) class logits.
            score_pred: (B,) predicted continuous score [0, 100].
            state_gt: (B,) ground-truth class indices.
            score_gt: (B,) ground-truth continuous score [0, 100].
        """
        ce = self.ce_loss(logits, state_gt)
        # Scores are reported on a 0-100 scale, while the other tasks use
        # probabilities. Normalize this term so regression does not overwhelm
        # classification and attribution by four orders of magnitude.
        mse = F.mse_loss(score_pred / 100.0, score_gt / 100.0)

        total = self.ce_weight * ce + self.mse_weight * mse

        return {
            "health_total": total,
            "health_ce": ce,
            "health_mse": mse,
        }


class AttributionLoss(nn.Module):
    """Binary cross-entropy loss for multi-label driver attribution.
    
    Each driver is an independent binary classification.
    Optionally includes strength regression loss.
    """

    def __init__(self, bce_weight: float = 1.0, strength_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.strength_weight = strength_weight

    def forward(
        self,
        logits: Tensor,
        strength_pred: Tensor | None,
        drivers_gt: Tensor,
        strength_gt: Tensor | None,
    ) -> dict[str, Tensor]:
        """
        Args:
            logits: (B, 8) raw logits per driver.
            strength_pred: (B, 8) predicted strengths [0, 1].
            drivers_gt: (B, 8) binary ground-truth.
            strength_gt: (B, 8) ground-truth strengths.
        """
        bce = F.binary_cross_entropy_with_logits(logits, drivers_gt)

        total = self.bce_weight * bce

        result = {
            "attribution_total": total,
            "attribution_bce": bce,
        }

        if strength_pred is not None and strength_gt is not None:
            strength_mse = F.mse_loss(strength_pred, strength_gt)
            total = total + self.strength_weight * strength_mse
            result["attribution_strength_mse"] = strength_mse
            result["attribution_total"] = total

        return result


class TrajectoryLoss(nn.Module):
    """Combined MSE + Huber loss for multi-horizon trajectory prediction.
    
    Optionally uses heteroscedastic loss when uncertainty is predicted:
      L = 0.5 * exp(-logvar) * (y - y_hat)^2 + 0.5 * logvar
    
    This naturally learns the uncertainty calibration.
    """

    def __init__(
        self,
        mse_weight: float = 2.0,
        huber_weight: float = 1.0,
        use_heteroscedastic: bool = True,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.use_heteroscedastic = use_heteroscedastic

    def forward(
        self,
        mean_pred: Tensor,
        logvar_pred: Tensor | None,
        trajectory_gt: Tensor,
    ) -> dict[str, Tensor]:
        """
        Args:
            mean_pred: (B, H) predicted health at each horizon [0, 1].
            logvar_pred: (B, H) predicted log-variance (for uncertainty).
            trajectory_gt: (B, H) ground-truth future health [0, 1].
        """
        # Normalize gt to [0, 1] if needed
        if trajectory_gt.max() > 1.5:
            trajectory_gt = trajectory_gt / 100.0

        mse = F.mse_loss(mean_pred, trajectory_gt)
        huber = F.smooth_l1_loss(mean_pred, trajectory_gt)

        result = {
            "trajectory_mse": mse,
            "trajectory_huber": huber,
        }

        if self.use_heteroscedastic and logvar_pred is not None:
            # Heteroscedastic loss: learns uncertainty
            logvar_pred = logvar_pred.clamp(-10.0, 2.0)
            precision = torch.exp(-logvar_pred)
            hetero_loss = 0.5 * (precision * (mean_pred - trajectory_gt) ** 2 + logvar_pred).mean()
            total = self.mse_weight * hetero_loss + self.huber_weight * huber
            result["trajectory_hetero"] = hetero_loss
        else:
            total = self.mse_weight * mse + self.huber_weight * huber

        result["trajectory_total"] = total
        return result


class ForestMindLoss(nn.Module):
    """Unified multi-task loss for ForestMind.
    
    Combines health, attribution, and trajectory losses with
    configurable weights.
    
    Args:
        loss_weights: Dictionary of loss component weights.
    """

    def __init__(self, loss_weights: dict | None = None):
        super().__init__()
        weights = loss_weights or {}

        self.health_loss = HealthLoss(
            ce_weight=weights.get("health_ce", 1.0),
            mse_weight=weights.get("health_mse", 0.5),
        )
        self.attribution_loss = AttributionLoss(
            bce_weight=weights.get("attribution_bce", 1.0),
            strength_weight=weights.get("attribution_strength", 0.5),
        )
        self.trajectory_loss = TrajectoryLoss(
            mse_weight=weights.get("trajectory_mse", 2.0),
            huber_weight=weights.get("trajectory_huber", 1.0),
        )

        # Task weights for combining
        self.w_health = weights.get("task_health", 1.0)
        self.w_attribution = weights.get("task_attribution", 1.0)
        self.w_trajectory = weights.get("task_trajectory", 1.5)

    def forward(
        self,
        outputs: dict[str, dict[str, Tensor]],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute combined multi-task loss.
        
        Args:
            outputs: Model outputs with keys "health", "attribution", "trajectory".
            targets: Ground-truth targets.
            
        Returns:
            Dictionary with all individual and total losses.
        """
        all_losses = {}
        if "label_valid" in targets and not torch.as_tensor(targets["label_valid"]).all():
            raise ValueError("Supervised loss requires observed labels, not unlabeled acquisitions")

        # Health loss
        health_losses = self.health_loss(
            logits=outputs["health"]["logits"],
            score_pred=outputs["health"]["score"],
            state_gt=targets["health_state"],
            score_gt=targets["health_score"],
        )
        all_losses.update(health_losses)

        # Attribution loss
        attr_losses = self.attribution_loss(
            logits=outputs["attribution"]["logits"],
            strength_pred=outputs["attribution"].get("strength"),
            drivers_gt=targets["drivers"],
            strength_gt=targets.get("driver_strengths"),
        )
        all_losses.update(attr_losses)

        # Trajectory loss
        traj_losses = self.trajectory_loss(
            mean_pred=outputs["trajectory"]["mean"],
            logvar_pred=outputs["trajectory"].get("logvar"),
            trajectory_gt=targets["trajectory"],
        )
        all_losses.update(traj_losses)

        # Total
        total = (
            self.w_health * health_losses["health_total"]
            + self.w_attribution * attr_losses["attribution_total"]
            + self.w_trajectory * traj_losses["trajectory_total"]
        )
        all_losses["total"] = total

        return all_losses
