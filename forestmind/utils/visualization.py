"""
Visualization utilities for ForestMind predictions.

Generates exploratory plots for:
  - Multi-spectral patch visualization (RGB composites)
  - Temporal health trajectories with predictions
  - Candidate pressure scores
  - Uncalibrated ecological screening dashboards
  - Hypothetical intervention comparison charts
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─── Color Palettes ──────────────────────────────────────────────────────────

HEALTH_COLORS = {
    "Healthy": "#2ecc71",
    "Stressed": "#f39c12",
    "Degraded": "#e67e22",
    "Severely Degraded": "#e74c3c",
    "Cleared": "#7f8c8d",
}

DRIVER_COLORS = {
    "Agriculture": "#27ae60",
    "Cattle Pasture": "#f1c40f",
    "Mining": "#8e44ad",
    "Logging": "#d35400",
    "Road Construction": "#7f8c8d",
    "Wildfire": "#c0392b",
    "Drought": "#e67e22",
    "Urban Expansion": "#2c3e50",
}

RISK_COLORS = {
    "Low": "#2ecc71",
    "Moderate": "#f39c12",
    "Elevated": "#e67e22",
    "High": "#e74c3c",
    "Critical": "#8e44ad",
}


def _check_matplotlib():
    if not HAS_MPL:
        raise ImportError(
            "Visualization requires matplotlib. Install with: pip install matplotlib"
        )


def _finite_value(value) -> float | None:
    """Treat missing diagnostics as unavailable, never as a numeric zero."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def plot_rgb_composite(
    optical: np.ndarray,
    timestep: int = -1,
    bands: tuple[int, int, int] = (3, 2, 1),  # Red, Green, Blue
    gamma: float = 2.2,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot an RGB composite from multi-spectral data.
    
    Args:
        optical: (T, 12, H, W) multi-spectral data.
        timestep: Which timestep to visualize (-1 = last).
        bands: Band indices for R, G, B channels.
        gamma: Gamma correction for display.
        save_path: Optional path to save the figure.
    """
    _check_matplotlib()

    img = optical[timestep]  # (12, H, W)
    rgb = np.stack([img[b] for b in bands], axis=-1)  # (H, W, 3)

    # Normalize and gamma correct
    rgb = np.clip(rgb, 0, 1)
    rgb = np.power(rgb, 1.0 / gamma)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title(f"RGB Composite (Bands {bands})", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_health_trajectory(
    health_history: np.ndarray,
    predicted_trajectory: np.ndarray | None = None,
    predicted_uncertainty: np.ndarray | None = None,
    horizon_labels: list[str] | None = None,
    title: str = "Forest Condition Trajectory (uncalibrated)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot health trajectory with historical data and future predictions.
    
    Args:
        health_history: (T,) observed health scores [0, 1].
        predicted_trajectory: (H,) predicted future scores [0, 1].
        predicted_uncertainty: (H,) prediction std dev.
        horizon_labels: Labels for prediction horizons.
        title: Plot title.
        save_path: Optional path to save the figure.
    """
    _check_matplotlib()

    health_history = np.asarray(health_history, dtype=float)
    if health_history.ndim != 1:
        raise ValueError("health_history must be one-dimensional")
    if predicted_trajectory is not None:
        predicted_trajectory = np.asarray(predicted_trajectory, dtype=float)
        if predicted_trajectory.ndim != 1:
            raise ValueError("predicted_trajectory must be one-dimensional")
        if horizon_labels is not None and len(horizon_labels) != len(predicted_trajectory):
            raise ValueError("horizon_labels must match predicted_trajectory")
        if predicted_uncertainty is not None:
            predicted_uncertainty = np.asarray(predicted_uncertainty, dtype=float)
            if predicted_uncertainty.shape != predicted_trajectory.shape:
                raise ValueError("predicted_uncertainty must match predicted_trajectory")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Historical trajectory
    t_hist = np.arange(len(health_history))
    ax.plot(
        t_hist, health_history * 100, "o-",
        color="#2c3e50", linewidth=2, markersize=5,
        label="Historical condition estimates", zorder=5,
    )

    # Predicted trajectory
    if predicted_trajectory is not None and len(predicted_trajectory):
        anchored = bool(len(health_history))
        t_pred_start = len(health_history) - 1 if anchored else 0
        pred_with_current = (np.concatenate([[health_history[-1]], predicted_trajectory])
                             if anchored else predicted_trajectory)
        t_pred = np.arange(t_pred_start, t_pred_start + len(pred_with_current))
        pred_scores = pred_with_current * 100

        ax.plot(
            t_pred, pred_scores, "s--",
            color="#e74c3c", linewidth=2, markersize=6,
            label="Model projection (uncalibrated)", zorder=5,
        )

        # Uncertainty band
        if predicted_uncertainty is not None:
            unc_with_zero = (np.concatenate([[0], predicted_uncertainty])
                             if anchored else predicted_uncertainty)
            upper = pred_scores + unc_with_zero * 100
            lower = pred_scores - unc_with_zero * 100
            ax.fill_between(
                t_pred, lower, upper,
                color="#e74c3c", alpha=0.15, label="Model spread (uncalibrated)",
            )

        # Horizon labels
        if horizon_labels:
            for i, label in enumerate(horizon_labels):
                ax.annotate(
                    label,
                    (t_pred[i + int(anchored)], pred_scores[i + int(anchored)]),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=8, ha="center", color="#e74c3c",
                )

    if not len(health_history):
        ax.text(0.02, 0.95, "Historical condition unavailable", transform=ax.transAxes,
                ha="left", va="top", color="#7f8c8d")
    ax.set_xlabel("Observation / forecast index (forecast spacing is not elapsed time)", fontsize=10)
    ax.set_ylabel("Condition score (uncalibrated)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_driver_attribution(
    driver_names: list[str],
    driver_probs: np.ndarray,
    driver_strengths: np.ndarray | None = None,
    title: str = "Candidate Pressure Scores (uncalibrated)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot driver attribution as horizontal bar chart.
    
    Args:
        driver_names: List of driver names.
        driver_probs: (8,) detection probabilities [0, 1].
        driver_strengths: (8,) estimated strengths [0, 1].
        title: Plot title.
        save_path: Optional path to save.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    colors = [DRIVER_COLORS.get(name, "#95a5a6") for name in driver_names]

    # Sort by probability
    sort_idx = np.argsort(driver_probs)[::-1]
    sorted_names = [driver_names[i] for i in sort_idx]
    sorted_probs = driver_probs[sort_idx]
    sorted_colors = [colors[i] for i in sort_idx]

    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_probs * 100, color=sorted_colors, alpha=0.8, height=0.6)

    # Add strength overlay if available
    if driver_strengths is not None:
        sorted_strengths = driver_strengths[sort_idx]
        for i, (prob, strength) in enumerate(zip(sorted_probs, sorted_strengths)):
            if prob > 0.3:
                ax.text(
                    prob * 100 + 1, i,
                    f"Strength: {strength * 100:.0f}%",
                    va="center", fontsize=9, color="#2c3e50",
                )

    # An operating cutoff does not establish causal attribution or calibration.
    ax.axvline(x=50, color="#e74c3c", linestyle="--", alpha=0.5, label="Illustrative score cutoff")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Model score (0–100; uncalibrated, not causal contribution)", fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_intervention_comparison(
    intervention_analysis,
    title: str = "Hypothetical Intervention Scenarios (uncalibrated)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot comparison of intervention scenarios.
    
    Args:
        intervention_analysis: InterventionAnalysis result.
        title: Plot title.
        save_path: Optional path to save.
    """
    _check_matplotlib()

    from forestmind.risk.intervention import InterventionAnalysis

    analysis: InterventionAnalysis = intervention_analysis
    horizons = analysis.horizons
    num_horizons = len(horizons)
    scenarios = analysis.scenarios or [analysis.baseline]
    if any(len(result.projected_health_100) != num_horizons for result in scenarios):
        raise ValueError("Each scenario must have one score per configured horizon")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: trajectory comparison
    x = np.arange(num_horizons)
    width = 0.8 / len(scenarios)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(scenarios)))

    for i, result in enumerate(scenarios):
        scores = result.projected_health_100
        ax1.bar(
            x + i * width - 0.4 + width / 2,
            scores,
            width=width,
            label=result.scenario.name,
            color=colors[i],
            alpha=0.85,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons, fontsize=10)
    ax1.set_ylabel("Conditional health score", fontsize=12)
    ax1.set_title("Health Under Scenario Assumptions", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)

    # Right: cost-effectiveness
    scenario_names = [r.scenario.name for r in analysis.scenarios if r.scenario.driver_reduction]
    cost_eff = [r.cost_effectiveness for r in analysis.scenarios if r.scenario.driver_reduction]

    if scenario_names:
        bar_colors = ["#2ecc71" if r == analysis.recommended else "#3498db"
                      for r in analysis.scenarios if r.scenario.driver_reduction]

        y_pos = np.arange(len(scenario_names))
        ax2.barh(y_pos, cost_eff, color=bar_colors, alpha=0.8, height=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(scenario_names, fontsize=10)
        ax2.set_xlabel("Assumed mean benefit / relative cost", fontsize=11)
        ax2.set_title("Illustrative Scenario Ranking", fontsize=13, fontweight="bold")
        ax2.invert_yaxis()

        if analysis.recommended:
            ax2.annotate(
                "Highest assumed benefit/cost",
                xy=(0.95, 0.02), xycoords="axes fraction",
                fontsize=10, color="#2ecc71", fontweight="bold",
                ha="right",
            )

    else:
        ax2.text(0.5, 0.5, "No active intervention scenarios", transform=ax2.transAxes,
                 ha="center", va="center", color="#7f8c8d")
        ax2.set_axis_off()

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "Scenario differences are conditional assumptions, not validated causal intervention effects.",
             ha="center", fontsize=10, color="#7f8c8d")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_tipping_point_dashboard(
    assessment,
    health_history: np.ndarray,
    title: str = "Exploratory Ecological Screening (uncalibrated)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot available screening indicators without claiming tipping probabilities.
    
    Args:
        assessment: TippingPointAssessment result.
        health_history: (T,) historical health scores.
        title: Plot title.
        save_path: Optional path to save.
    """
    _check_matplotlib()

    from forestmind.risk.tipping_point import TippingPointAssessment
    tp: TippingPointAssessment = assessment

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ─── Top left: Risk gauge ──────────────────────────
    ax_gauge = fig.add_subplot(gs[0, 0])
    risk_color = RISK_COLORS.get(tp.risk_category, "#95a5a6")

    # Draw semicircular gauge
    theta = np.linspace(np.pi, 0, 100)
    ax_gauge.plot(np.cos(theta), np.sin(theta), color="#ecf0f1", linewidth=20, solid_capstyle="round")
    
    # A missing screening score is unavailable, never represented as zero risk.
    risk_score = _finite_value(tp.risk_score)
    fill_idx = int(np.clip(risk_score, 0, 1) * len(theta)) if risk_score is not None else 0
    if fill_idx > 0:
        theta_fill = theta[:fill_idx]
        ax_gauge.plot(np.cos(theta_fill), np.sin(theta_fill), color=risk_color, linewidth=20, solid_capstyle="round")

    ax_gauge.text(0, -0.1, f"{risk_score:.2f}" if risk_score is not None else "Unavailable",
                  fontsize=28 if risk_score is not None else 18, ha="center", va="center",
                  fontweight="bold", color=risk_color)
    ax_gauge.text(0, -0.4, tp.risk_category, fontsize=14, ha="center", va="center", color=risk_color)
    ax_gauge.set_xlim(-1.3, 1.3)
    ax_gauge.set_ylim(-0.6, 1.3)
    ax_gauge.set_aspect("equal")
    ax_gauge.axis("off")
    ax_gauge.set_title("Heuristic screening index", fontsize=12, fontweight="bold")

    # Bars allow missing dimensions to remain visibly unavailable. A closed
    # radar polygon would imply a numeric value for every component.
    ax_components = fig.add_subplot(gs[0, 1])
    resilience = _finite_value(tp.resilience_score)
    components = {
        "CSD heuristic": _finite_value(tp.csd_score),
        "Limited rebound proxy": 1 - resilience if resilience is not None else None,
        "Spatial fragmentation": _finite_value(tp.fragmentation_score),
        "Decline proxy": _finite_value(tp.momentum_score),
        "Drought-coincident drop": _finite_value(tp.drought_sensitivity_score),
    }
    for index, value in enumerate(components.values()):
        if value is None:
            ax_components.text(0.02, index, "Unavailable", va="center", fontsize=9,
                               color="#7f8c8d")
        else:
            ax_components.barh(index, value, color=risk_color, alpha=0.8, height=0.5)
    ax_components.set_yticks(np.arange(len(components)))
    ax_components.set_yticklabels(list(components), fontsize=8)
    ax_components.set_ylim(len(components) - 0.5, -0.5)
    ax_components.set_xlim(0, 1)
    ax_components.set_title("Selected screening components", fontsize=11, fontweight="bold")

    # ─── Top right: CSD details ───────────────────────
    ax_csd = fig.add_subplot(gs[0, 2])
    if tp.csd_result and getattr(tp.csd_result, "status", "available") == "available":
        csd = tp.csd_result
        metrics = {
            "AR(1) coefficient": _finite_value(csd.ar1_coefficient),
            "AR(1) trend (tau)": _finite_value(csd.ar1_trend),
            "Variance trend (tau)": _finite_value(csd.variance_trend),
            "Correlation trend (tau)": _finite_value(csd.autocorrelation_trend),
            "Persistence proxy": _finite_value(csd.recovery_rate),
        }
        names = list(metrics.keys())
        y_pos = np.arange(len(names))
        vals = [value for value in metrics.values() if value is not None]
        for index, value in enumerate(metrics.values()):
            if value is None:
                ax_csd.text(0.02, index, "Unavailable", va="center", fontsize=9,
                            color="#7f8c8d", transform=ax_csd.get_yaxis_transform())
            else:
                ax_csd.barh(index, value, color="#3498db", alpha=0.8, height=0.5)
        ax_csd.set_yticks(y_pos)
        ax_csd.set_yticklabels(names, fontsize=9)
        ax_csd.set_xlim(min([-1.1] + [value * 1.1 for value in vals]),
                         max([1.1] + [value * 1.1 for value in vals]))
        ax_csd.set_ylim(len(names) - 0.5, -0.5)
        ax_csd.axvline(0, color="#95a5a6", linewidth=0.8)
        ax_csd.set_xlabel("Raw statistics; proxy is not a measured recovery rate", fontsize=8)
    else:
        status = getattr(tp.csd_result, "status", "unavailable").replace("_", " ")
        ax_csd.text(0.5, 0.5, f"Diagnostics unavailable\n{status}", transform=ax_csd.transAxes,
                    ha="center", va="center", fontsize=11, color="#7f8c8d")
        ax_csd.set_axis_off()
    ax_csd.set_title("Descriptive CSD diagnostics", fontsize=11, fontweight="bold")

    # ─── Bottom: Health time series ───────────────────
    ax_ts = fig.add_subplot(gs[1, :])
    history = np.asarray(health_history, dtype=float)
    if history.ndim != 1:
        raise ValueError("health_history must be one-dimensional")
    t = np.arange(len(history))
    ax_ts.plot(t, history * 100, "o-", color="#2c3e50", linewidth=2, markersize=4,
               label="Historical condition estimates")
    if not np.isfinite(history).any():
        ax_ts.text(0.5, 0.5, "Historical condition unavailable", transform=ax_ts.transAxes,
                   ha="center", va="center", color="#7f8c8d")

    ax_ts.set_xlabel("Observation index", fontsize=11)
    ax_ts.set_ylabel("Condition score (uncalibrated)", fontsize=11)
    ax_ts.set_title("Historical Condition", fontsize=12, fontweight="bold")
    ax_ts.set_ylim(-5, 105)
    ax_ts.legend(fontsize=9)
    ax_ts.grid(alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.01,
             "Screening indicators do not establish tipping probabilities, tipping dates or ecological irreversibility.",
             ha="center", fontsize=10, color="#7f8c8d")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
