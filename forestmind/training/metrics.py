"""
Evaluation metrics for ForestMind.

Computes per-task and aggregate metrics:
  - Health: accuracy, F1 (macro), MAE of continuous score
  - Attribution: precision, recall, F1 (per-driver and macro)
  - Trajectory: MAE, RMSE, calibration (if uncertainty available)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor


@dataclass
class MetricsResult:
    """Aggregated evaluation metrics."""
    # Health metrics
    health_accuracy: float = 0.0
    health_f1_macro: float = 0.0
    health_mae: float = 0.0

    # Attribution metrics
    attribution_precision: float = 0.0
    attribution_recall: float = 0.0
    attribution_f1: float = 0.0
    attribution_per_driver_f1: list[float] = field(default_factory=list)

    # Trajectory metrics
    trajectory_mae: float = 0.0
    trajectory_rmse: float = 0.0
    trajectory_per_horizon_mae: list[float] = field(default_factory=list)

    # Aggregate
    aggregate_score: float = 0.0


class ForestMindMetrics:
    """Accumulates and computes evaluation metrics across batches."""

    def __init__(self, num_drivers: int = 8, num_horizons: int = 5):
        self.num_drivers = num_drivers
        self.num_horizons = num_horizons
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self._health_correct = 0
        self._health_total = 0
        self._health_preds = []
        self._health_targets = []
        self._health_score_errors = []

        self._attr_preds = []
        self._attr_targets = []

        self._traj_preds = []
        self._traj_targets = []

    @torch.no_grad()
    def update(
        self,
        outputs: dict[str, dict[str, Tensor]],
        targets: dict[str, Tensor],
    ):
        """Accumulate predictions from one batch.
        
        Args:
            outputs: Model outputs.
            targets: Ground-truth targets.
        """
        if "label_valid" in targets and not torch.as_tensor(targets["label_valid"]).all():
            raise ValueError("Metrics require observed labels, not unlabeled acquisitions")
        # Health
        health_state_pred = outputs["health"]["state"].cpu().numpy()
        health_state_gt = targets["health_state"].cpu().numpy()
        self._health_preds.extend(health_state_pred.tolist())
        self._health_targets.extend(health_state_gt.tolist())
        self._health_correct += (health_state_pred == health_state_gt).sum()
        self._health_total += len(health_state_gt)

        if "score" in outputs["health"]:
            score_pred = outputs["health"]["score"].cpu().numpy()
            score_gt = targets["health_score"].cpu().numpy()
            self._health_score_errors.extend(np.abs(score_pred - score_gt).tolist())

        # Attribution
        attr_pred = outputs["attribution"]["active"].cpu().numpy()
        attr_gt = targets["drivers"].cpu().numpy()
        self._attr_preds.append(attr_pred)
        self._attr_targets.append(attr_gt)

        # Trajectory
        traj_pred = outputs["trajectory"]["mean"].cpu().numpy()
        traj_gt = targets["trajectory"].cpu().numpy()
        # Normalize gt to [0, 1] if needed
        if traj_gt.max() > 1.5:
            traj_gt = traj_gt / 100.0
        self._traj_preds.append(traj_pred)
        self._traj_targets.append(traj_gt)

    def compute(self) -> MetricsResult:
        """Compute all metrics from accumulated predictions."""
        result = MetricsResult()
        if self._health_total == 0:
            return result

        # ─── Health Metrics ────────────────────────────────
        if self._health_total > 0:
            result.health_accuracy = float(self._health_correct / self._health_total)

            # Macro F1
            preds = np.array(self._health_preds)
            targets = np.array(self._health_targets)
            f1s = []
            for cls in range(5):
                tp = ((preds == cls) & (targets == cls)).sum()
                fp = ((preds == cls) & (targets != cls)).sum()
                fn = ((preds != cls) & (targets == cls)).sum()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                f1s.append(f1)
            result.health_f1_macro = float(np.mean(f1s))

        if self._health_score_errors:
            result.health_mae = float(np.mean(self._health_score_errors))

        # ─── Attribution Metrics ───────────────────────────
        if self._attr_preds:
            all_preds = np.vstack(self._attr_preds)
            all_targets = np.vstack(self._attr_targets)

            # Per-driver F1
            per_driver_f1 = []
            for d in range(self.num_drivers):
                tp = ((all_preds[:, d] == 1) & (all_targets[:, d] == 1)).sum()
                fp = ((all_preds[:, d] == 1) & (all_targets[:, d] == 0)).sum()
                fn = ((all_preds[:, d] == 0) & (all_targets[:, d] == 1)).sum()
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                per_driver_f1.append(float(f1))

            result.attribution_per_driver_f1 = per_driver_f1
            result.attribution_f1 = float(np.mean(per_driver_f1))

            # Overall precision/recall
            tp_all = ((all_preds == 1) & (all_targets == 1)).sum()
            fp_all = ((all_preds == 1) & (all_targets == 0)).sum()
            fn_all = ((all_preds == 0) & (all_targets == 1)).sum()
            result.attribution_precision = float(tp_all / (tp_all + fp_all + 1e-8))
            result.attribution_recall = float(tp_all / (tp_all + fn_all + 1e-8))

        # ─── Trajectory Metrics ────────────────────────────
        if self._traj_preds:
            all_traj_preds = np.vstack(self._traj_preds)
            all_traj_targets = np.vstack(self._traj_targets)

            errors = all_traj_preds - all_traj_targets
            result.trajectory_mae = float(np.mean(np.abs(errors)))
            result.trajectory_rmse = float(np.sqrt(np.mean(errors ** 2)))

            # Per-horizon MAE
            per_horizon = []
            for h in range(min(self.num_horizons, all_traj_preds.shape[1])):
                h_mae = float(np.mean(np.abs(all_traj_preds[:, h] - all_traj_targets[:, h])))
                per_horizon.append(h_mae)
            result.trajectory_per_horizon_mae = per_horizon

        # ─── Aggregate Score ───────────────────────────────
        # Weighted combination of key metrics (higher = better)
        result.aggregate_score = (
            0.3 * result.health_accuracy
            + 0.2 * result.health_f1_macro
            + 0.2 * result.attribution_f1
            + 0.3 * max(0, 1.0 - result.trajectory_mae * 5)  # lower MAE = better
        )

        return result

    def summary_str(self, result: MetricsResult | None = None) -> str:
        """Format metrics as a human-readable string."""
        if result is None:
            result = self.compute()

        lines = [
            "─── Evaluation Metrics ──────────────────────",
            f"  Health Accuracy:     {result.health_accuracy:.3f}",
            f"  Health F1 (macro):   {result.health_f1_macro:.3f}",
            f"  Health MAE:          {result.health_mae:.2f}",
            f"  Attribution F1:      {result.attribution_f1:.3f}",
            f"  Attribution P/R:     {result.attribution_precision:.3f} / {result.attribution_recall:.3f}",
            f"  Trajectory MAE:      {result.trajectory_mae:.4f}",
            f"  Trajectory RMSE:     {result.trajectory_rmse:.4f}",
        ]

        horizon_names = ["30d", "6mo", "1yr", "3yr", "5yr"]
        if result.trajectory_per_horizon_mae:
            lines.append("  Per-horizon MAE:")
            for name, mae in zip(horizon_names, result.trajectory_per_horizon_mae):
                lines.append(f"    {name:>4s}: {mae:.4f}")

        lines.append(f"  Aggregate Score:     {result.aggregate_score:.3f}")
        lines.append("─────────────────────────────────────────────")

        return "\n".join(lines)
