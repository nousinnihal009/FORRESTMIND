"""
Visualization utilities for ForestMind predictions.

Generates publication-quality plots for:
  - Multi-spectral patch visualization (RGB composites)
  - Temporal health trajectories with predictions
  - Driver attribution bar charts
  - Tipping-point risk dashboards
  - Intervention comparison charts
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
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
    title: str = "Forest Health Trajectory",
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

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Health state background colors
    health_bands = [
        (0.80, 1.00, "#2ecc7133", "Healthy"),
        (0.60, 0.80, "#f39c1233", "Stressed"),
        (0.40, 0.60, "#e67e2233", "Degraded"),
        (0.15, 0.40, "#e74c3c33", "Severely Degraded"),
        (0.00, 0.15, "#7f8c8d33", "Cleared"),
    ]
    for low, high, color, label in health_bands:
        ax.axhspan(low * 100, high * 100, color=color, alpha=0.3)

    # Historical trajectory
    t_hist = np.arange(len(health_history))
    ax.plot(
        t_hist, health_history * 100, "o-",
        color="#2c3e50", linewidth=2, markersize=5,
        label="Observed", zorder=5,
    )

    # Predicted trajectory
    if predicted_trajectory is not None:
        t_pred_start = len(health_history) - 1
        t_pred = np.arange(t_pred_start, t_pred_start + len(predicted_trajectory) + 1)
        pred_with_current = np.concatenate([[health_history[-1]], predicted_trajectory])
        pred_scores = pred_with_current * 100

        ax.plot(
            t_pred, pred_scores, "s--",
            color="#e74c3c", linewidth=2, markersize=6,
            label="Predicted", zorder=5,
        )

        # Uncertainty band
        if predicted_uncertainty is not None:
            unc_with_zero = np.concatenate([[0], predicted_uncertainty])
            upper = pred_scores + unc_with_zero * 100
            lower = pred_scores - unc_with_zero * 100
            ax.fill_between(
                t_pred, lower, upper,
                color="#e74c3c", alpha=0.15, label="Uncertainty",
            )

        # Horizon labels
        if horizon_labels:
            for i, label in enumerate(horizon_labels):
                ax.annotate(
                    label,
                    (t_pred[i + 1], pred_scores[i + 1]),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=8, ha="center", color="#e74c3c",
                )

    # Tipping point threshold
    ax.axhline(y=30, color="#8e44ad", linestyle=":", linewidth=1.5, alpha=0.7, label="Tipping threshold")

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Forest Health Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_driver_attribution(
    driver_names: list[str],
    driver_probs: np.ndarray,
    driver_strengths: np.ndarray | None = None,
    title: str = "Degradation Driver Attribution",
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

    # Detection threshold line
    ax.axvline(x=50, color="#e74c3c", linestyle="--", alpha=0.5, label="Detection threshold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Detection Probability (%)", fontsize=12)
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
    title: str = "Intervention Scenario Comparison",
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: trajectory comparison
    x = np.arange(num_horizons)
    width = 0.8 / len(analysis.scenarios)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(analysis.scenarios)))

    for i, result in enumerate(analysis.scenarios):
        scores = result.projected_health_100[:num_horizons]
        ax1.bar(
            x + i * width - 0.4 + width / 2,
            scores,
            width=width,
            label=result.scenario.name,
            color=colors[i],
            alpha=0.85,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons[:num_horizons], fontsize=10)
    ax1.set_ylabel("Projected Health Score", fontsize=12)
    ax1.set_title("Projected Health by Scenario", fontsize=13, fontweight="bold")
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
        ax2.set_xlabel("Cost-Effectiveness Score", fontsize=12)
        ax2.set_title("Cost-Effectiveness Analysis", fontsize=13, fontweight="bold")
        ax2.invert_yaxis()

        if analysis.recommended:
            ax2.annotate(
                "★ Recommended",
                xy=(0.95, 0.02), xycoords="axes fraction",
                fontsize=10, color="#2ecc71", fontweight="bold",
                ha="right",
            )

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_tipping_point_dashboard(
    assessment,
    health_history: np.ndarray,
    title: str = "Tipping-Point Risk Assessment",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot a comprehensive tipping-point risk dashboard.
    
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
    
    # Fill portion based on risk
    fill_idx = int(tp.risk_score * 100)
    if fill_idx > 0:
        theta_fill = theta[:fill_idx]
        ax_gauge.plot(np.cos(theta_fill), np.sin(theta_fill), color=risk_color, linewidth=20, solid_capstyle="round")

    ax_gauge.text(0, -0.1, f"{tp.risk_score:.2f}", fontsize=28, ha="center", va="center", fontweight="bold", color=risk_color)
    ax_gauge.text(0, -0.4, tp.risk_category, fontsize=14, ha="center", va="center", color=risk_color)
    ax_gauge.set_xlim(-1.3, 1.3)
    ax_gauge.set_ylim(-0.6, 1.3)
    ax_gauge.set_aspect("equal")
    ax_gauge.axis("off")
    ax_gauge.set_title("Risk Score", fontsize=12, fontweight="bold")

    # ─── Top middle: Component radar ──────────────────
    ax_radar = fig.add_subplot(gs[0, 1], projection="polar")
    categories = ["CSD", "Resilience\n(inverse)", "Fragmentation", "Momentum", "Drought\nSensitivity"]
    values = [
        tp.csd_score,
        1 - tp.resilience_score,
        tp.fragmentation_score,
        tp.momentum_score,
        tp.drought_sensitivity_score,
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]
    
    ax_radar.fill(angles_plot, values_plot, alpha=0.25, color=risk_color)
    ax_radar.plot(angles_plot, values_plot, "o-", color=risk_color, linewidth=2)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(categories, fontsize=8)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Risk Components", fontsize=12, fontweight="bold", pad=20)

    # ─── Top right: CSD details ───────────────────────
    ax_csd = fig.add_subplot(gs[0, 2])
    if tp.csd_result:
        csd = tp.csd_result
        metrics = {
            "AR(1) Coeff": csd.ar1_coefficient,
            "AR(1) Trend": (csd.ar1_trend + 1) / 2,
            "Variance Trend": (csd.variance_trend + 1) / 2,
            "AC Trend": (csd.autocorrelation_trend + 1) / 2,
            "Recovery Rate": csd.recovery_rate,
        }
        names = list(metrics.keys())
        vals = list(metrics.values())
        y_pos = np.arange(len(names))
        bar_colors = ["#e74c3c" if v > 0.7 else "#f39c12" if v > 0.4 else "#2ecc71" for v in vals]
        ax_csd.barh(y_pos, vals, color=bar_colors, alpha=0.8, height=0.5)
        ax_csd.set_yticks(y_pos)
        ax_csd.set_yticklabels(names, fontsize=9)
        ax_csd.set_xlim(0, 1.1)
        ax_csd.set_title("CSD Indicators", fontsize=12, fontweight="bold")
        ax_csd.invert_yaxis()

    # ─── Bottom: Health time series ───────────────────
    ax_ts = fig.add_subplot(gs[1, :])
    t = np.arange(len(health_history))
    ax_ts.plot(t, health_history * 100, "o-", color="#2c3e50", linewidth=2, markersize=4)
    ax_ts.axhline(y=30, color="#8e44ad", linestyle=":", linewidth=1.5, alpha=0.7, label="Tipping threshold")
    
    # Color background by health bands
    for low, high, color_hex, label in [
        (80, 100, "#2ecc71", "Healthy"), (60, 80, "#f39c12", "Stressed"),
        (40, 60, "#e67e22", "Degraded"), (15, 40, "#e74c3c", "Severe"),
        (0, 15, "#7f8c8d", "Cleared"),
    ]:
        ax_ts.axhspan(low, high, color=color_hex, alpha=0.08)

    if tp.time_to_tipping is not None and tp.time_to_tipping > 0:
        ttp_steps = tp.time_to_tipping / 30
        ax_ts.axvline(x=len(health_history) - 1 + ttp_steps, color="#8e44ad",
                      linestyle="--", linewidth=2, alpha=0.7, label=f"Est. tipping: ~{tp.time_to_tipping/365:.1f}yr")

    ax_ts.set_xlabel("Time Step", fontsize=11)
    ax_ts.set_ylabel("Health Score", fontsize=11)
    ax_ts.set_title("Health Trajectory", fontsize=12, fontweight="bold")
    ax_ts.set_ylim(-5, 105)
    ax_ts.legend(fontsize=9)
    ax_ts.grid(alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
