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
from forestmind.training.targets import supervision_masks


@dataclass
class MetricsResult:
    """Aggregated evaluation metrics."""
    # Health metrics
    health_accuracy: float | None = None
    health_f1_macro: float | None = None
    health_mae: float | None = None

    # Attribution metrics
    attribution_precision: float | None = None
    attribution_recall: float | None = None
    attribution_f1: float | None = None
    attribution_per_driver_f1: list[float] = field(default_factory=list)

    # Trajectory metrics
    trajectory_mae: float | None = None
    trajectory_rmse: float | None = None
    trajectory_per_horizon_mae: list[float] = field(default_factory=list)

    # Aggregate
    aggregate_score: float = 0.0
    observed_counts: dict = field(default_factory=dict)


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
        masks = {name: value.cpu().numpy() for name, value in supervision_masks(targets).items()}
        # Health
        health_state_pred = outputs["health"]["state"].cpu().numpy()[masks["health_state"]]
        health_state_gt = targets["health_state"].cpu().numpy()[masks["health_state"]]
        self._health_preds.extend(health_state_pred.tolist())
        self._health_targets.extend(health_state_gt.tolist())
        self._health_correct += (health_state_pred == health_state_gt).sum()
        self._health_total += len(health_state_gt)

        if "score" in outputs["health"]:
            score_pred = outputs["health"]["score"].cpu().numpy()
            score_gt = targets["health_score"].cpu().numpy()
            self._health_score_errors.extend(np.abs(score_pred - score_gt)[masks["health_score"]].tolist())

        # Attribution
        attr_pred = outputs["attribution"]["active"].cpu().numpy()
        attr_gt = targets["drivers"].cpu().numpy()
        self._attr_preds.append(np.where(masks["drivers"], attr_pred, np.nan))
        self._attr_targets.append(np.where(masks["drivers"], attr_gt, np.nan))

        # Trajectory
        traj_pred = outputs["trajectory"]["mean"].cpu().numpy()
        traj_gt = targets["trajectory"].cpu().numpy()
        # Normalize gt to [0, 1] if needed
        if masks["trajectory"].any() and traj_gt[masks["trajectory"]].max() > 1.5:
            traj_gt = traj_gt / 100.0
        self._traj_preds.append(np.where(masks["trajectory"], traj_pred, np.nan))
        self._traj_targets.append(np.where(masks["trajectory"], traj_gt, np.nan))

    def compute(self) -> MetricsResult:
        """Compute all metrics from accumulated predictions."""
        result = MetricsResult()
        result.observed_counts = {"health_state": self._health_total,
                                  "health_score": len(self._health_score_errors),
                                  "drivers": [0] * self.num_drivers,
                                  "trajectory": [0] * self.num_horizons}

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
                count = int(np.isfinite(all_targets[:, d]).sum())
                result.observed_counts["drivers"][d] = count
                if not count:
                    per_driver_f1.append(None)
                    continue
                tp = ((all_preds[:, d] == 1) & (all_targets[:, d] == 1)).sum()
                fp = ((all_preds[:, d] == 1) & (all_targets[:, d] == 0)).sum()
                fn = ((all_preds[:, d] == 0) & (all_targets[:, d] == 1)).sum()
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                per_driver_f1.append(float(f1))

            result.attribution_per_driver_f1 = per_driver_f1
            observed_f1 = [value for value in per_driver_f1 if value is not None]
            result.attribution_f1 = float(np.mean(observed_f1)) if observed_f1 else None

            # Overall precision/recall
            tp_all = ((all_preds == 1) & (all_targets == 1)).sum()
            fp_all = ((all_preds == 1) & (all_targets == 0)).sum()
            fn_all = ((all_preds == 0) & (all_targets == 1)).sum()
            if observed_f1:
                result.attribution_precision = float(tp_all / (tp_all + fp_all + 1e-8))
                result.attribution_recall = float(tp_all / (tp_all + fn_all + 1e-8))

        # ─── Trajectory Metrics ────────────────────────────
        if self._traj_preds:
            all_traj_preds = np.vstack(self._traj_preds)
            all_traj_targets = np.vstack(self._traj_targets)

            errors = all_traj_preds - all_traj_targets
            finite = np.isfinite(errors)
            if finite.any():
                result.trajectory_mae = float(np.abs(errors[finite]).mean())
                result.trajectory_rmse = float(np.sqrt(np.mean(errors[finite] ** 2)))

            # Per-horizon MAE
            per_horizon = []
            for h in range(min(self.num_horizons, all_traj_preds.shape[1])):
                count = int(finite[:, h].sum())
                result.observed_counts["trajectory"][h] = count
                h_mae = float(np.abs(errors[finite[:, h], h]).mean()) if count else None
                per_horizon.append(h_mae)
            result.trajectory_per_horizon_mae = per_horizon

        # ─── Aggregate Score ───────────────────────────────
        # Weighted combination of key metrics (higher = better)
        terms = [(0.3, result.health_accuracy), (0.2, result.health_f1_macro),
                 (0.2, result.attribution_f1),
                 (0.3, max(0, 1 - result.trajectory_mae * 5) if result.trajectory_mae is not None else None)]
        # Score-only studies must also have a usable checkpoint-selection metric.
        if result.health_accuracy is None and result.health_mae is not None:
            terms.append((0.5, max(0, 1 - result.health_mae / 100)))
        known_terms = [(weight, value) for weight, value in terms if value is not None]
        result.aggregate_score = (sum(w * v for w, v in known_terms) / sum(w for w, _ in known_terms)
                                  if known_terms else 0.0)

        return result

    def summary_str(self, result: MetricsResult | None = None) -> str:
        """Format metrics as a human-readable string."""
        if result is None:
            result = self.compute()

        def fmt(value, precision=3):
            return "unavailable" if value is None else f"{value:.{precision}f}"
        lines = [
            "─── Evaluation Metrics ──────────────────────",
            f"  Health Accuracy:     {fmt(result.health_accuracy)}",
            f"  Health F1 (macro):   {fmt(result.health_f1_macro)}",
            f"  Health MAE:          {fmt(result.health_mae, 2)}",
            f"  Attribution F1:      {fmt(result.attribution_f1)}",
            f"  Attribution P/R:     {fmt(result.attribution_precision)} / {fmt(result.attribution_recall)}",
            f"  Trajectory MAE:      {fmt(result.trajectory_mae, 4)}",
            f"  Trajectory RMSE:     {fmt(result.trajectory_rmse, 4)}",
        ]

        horizon_names = ["30d", "6mo", "1yr", "3yr", "5yr"]
        if result.trajectory_per_horizon_mae:
            lines.append("  Per-horizon MAE:")
            for name, mae in zip(horizon_names, result.trajectory_per_horizon_mae):
                lines.append(f"    {name:>4s}: {fmt(mae, 4)}")

        lines.append(f"  Aggregate Score:     {result.aggregate_score:.3f}")
        lines.append("─────────────────────────────────────────────")

        return "\n".join(lines)
