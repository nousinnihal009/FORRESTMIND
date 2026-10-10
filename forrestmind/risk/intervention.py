"""
Intervention Simulator for ForestMind.

Simulates counterfactual trajectories under hypothetical conservation
interventions. This is the "digital twin" component that enables
decision support:

  "If we implement a conservation corridor, what happens to forest
   health compared to no intervention?"

Each intervention is modeled as a reduction in one or more degradation
drivers, which modifies the predicted trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InterventionScenario:
    """Definition of a hypothetical intervention."""
    name: str
    description: str
    driver_reduction: dict[str, float]  # driver_name → reduction factor [0, 1]
    cost_relative: float = 1.0          # relative cost (1 = baseline)
    implementation_delay_days: int = 0  # delay before intervention takes effect


@dataclass
class InterventionResult:
    """Result of simulating one intervention scenario."""
    scenario: InterventionScenario
    projected_health: list[float]       # health scores at each horizon
    projected_health_100: list[float]   # scaled to 0-100
    improvement_vs_baseline: list[float]  # absolute improvement over no-intervention
    relative_improvement: list[float]   # % improvement over no-intervention
    cost_effectiveness: float           # improvement per unit cost
    summary: str


@dataclass
class InterventionAnalysis:
    """Complete intervention analysis with all scenarios."""
    baseline: InterventionResult        # no-intervention trajectory
    scenarios: list[InterventionResult]
    recommended: InterventionResult | None  # best cost-effective scenario
    horizons: list[str]
    current_health: float


# ─── Default Intervention Scenarios ───────────────────────────────────────────

DEFAULT_SCENARIOS = [
    InterventionScenario(
        name="No Intervention",
        description="Continuation of current trends without any conservation action.",
        driver_reduction={},
        cost_relative=0.0,
    ),
    InterventionScenario(
        name="Conservation Corridor",
        description="Establish protected wildlife corridors connecting forest fragments. "
                    "Reduces logging and road construction pressure.",
        driver_reduction={"logging": 0.6, "road_construction": 0.4},
        cost_relative=0.6,
    ),
    InterventionScenario(
        name="Fire Management",
        description="Implement fire prevention and controlled burn programs. "
                    "Reduces wildfire risk and associated drought damage.",
        driver_reduction={"wildfire": 0.7, "drought": 0.2},
        cost_relative=0.4,
    ),
    InterventionScenario(
        name="Agricultural Buffer Zone",
        description="Create buffer zones between agricultural areas and forest. "
                    "Reduces agricultural encroachment and cattle grazing.",
        driver_reduction={"agriculture": 0.5, "cattle_pasture": 0.6},
        cost_relative=0.5,
    ),
    InterventionScenario(
        name="Mining Regulation",
        description="Enforce strict mining regulations and restoration requirements. "
                    "Reduces mining-related deforestation.",
        driver_reduction={"mining": 0.8, "road_construction": 0.3},
        cost_relative=0.7,
    ),
    InterventionScenario(
        name="Full Protection",
        description="Comprehensive protection program: enforce anti-deforestation laws, "
                    "establish protected area status, active monitoring.",
        driver_reduction={
            "agriculture": 0.8, "cattle_pasture": 0.8, "mining": 0.9,
            "logging": 0.9, "road_construction": 0.7, "urban_expansion": 0.5,
        },
        cost_relative=1.0,
    ),
    InterventionScenario(
        name="Reforestation + Protection",
        description="Full protection plus active reforestation in degraded areas. "
                    "Maximum recovery potential.",
        driver_reduction={
            "agriculture": 0.8, "cattle_pasture": 0.8, "mining": 0.9,
            "logging": 0.9, "road_construction": 0.7, "wildfire": 0.5,
            "urban_expansion": 0.5,
        },
        cost_relative=1.5,
        implementation_delay_days=90,
    ),
]

DRIVER_NAMES = [
    "agriculture", "cattle_pasture", "mining", "logging",
    "road_construction", "wildfire", "drought", "urban_expansion",
]


def simulate_intervention(
    baseline_trajectory: np.ndarray,
    driver_strengths: np.ndarray,
    scenario: InterventionScenario,
    horizon_days: list[int] | None = None,
) -> InterventionResult:
    """Simulate a single intervention scenario.
    
    Models intervention as reducing the contribution of specific drivers
    to the degradation trajectory. The modified trajectory is computed
    by attenuating driver impacts.
    
    Args:
        baseline_trajectory: (H,) no-intervention health scores at horizons.
        driver_strengths: (8,) current driver strengths [0, 1].
        scenario: Intervention definition.
        horizon_days: List of horizon offsets in days.
        
    Returns:
        InterventionResult with projected outcomes.
    """
    horizon_days = horizon_days or [30, 180, 365, 1095, 1825]

    if not scenario.driver_reduction:
        # No intervention — return baseline
        projected = baseline_trajectory.tolist()
        return InterventionResult(
            scenario=scenario,
            projected_health=[float(h) for h in projected],
            projected_health_100=[float(h * 100) for h in projected],
            improvement_vs_baseline=[0.0] * len(projected),
            relative_improvement=[0.0] * len(projected),
            cost_effectiveness=0.0,
            summary="Baseline trajectory without intervention.",
        )

    # Compute total driver pressure before and after intervention
    total_pressure_before = 0.0
    total_pressure_after = 0.0

    for i, driver_name in enumerate(DRIVER_NAMES):
        strength = float(driver_strengths[i]) if i < len(driver_strengths) else 0.0
        total_pressure_before += strength
        reduction = scenario.driver_reduction.get(driver_name, 0.0)
        total_pressure_after += strength * (1.0 - reduction)

    # Pressure reduction factor
    if total_pressure_before > 0:
        pressure_reduction = 1.0 - (total_pressure_after / total_pressure_before)
    else:
        pressure_reduction = 0.0

    # Apply intervention effect to trajectory
    current_health = float(baseline_trajectory[0]) if len(baseline_trajectory) > 0 else 0.5
    projected = []

    for i, (base_h, horizon_d) in enumerate(zip(baseline_trajectory, horizon_days)):
        base_h = float(base_h)

        # Intervention effect grows over time (cumulative benefit)
        time_factor = min(1.0, horizon_d / 365.0)

        # Account for implementation delay
        if horizon_d < scenario.implementation_delay_days:
            effective_reduction = 0.0
        else:
            delay_factor = min(1.0, (horizon_d - scenario.implementation_delay_days) / 180.0)
            effective_reduction = pressure_reduction * delay_factor

        # Degradation that would have occurred
        degradation = max(0.0, current_health - base_h)

        # Reduce degradation by intervention effect
        mitigated_degradation = degradation * (1.0 - effective_reduction * time_factor)

        # Also add slight recovery effect from reduced pressure
        recovery_boost = effective_reduction * 0.05 * time_factor

        projected_h = current_health - mitigated_degradation + recovery_boost
        projected_h = float(np.clip(projected_h, 0.0, 1.0))
        projected.append(projected_h)

    # Compute improvements
    improvement = [p - b for p, b in zip(projected, baseline_trajectory.tolist())]
    relative = [
        (imp / max(0.01, abs(current_health - b))) * 100 if abs(current_health - b) > 0.01 else 0.0
        for imp, b in zip(improvement, baseline_trajectory.tolist())
    ]

    # Cost effectiveness (average improvement per unit cost)
    avg_improvement = np.mean(improvement)
    cost_eff = float(avg_improvement / max(0.01, scenario.cost_relative))

    # Summary
    final_health = projected[-1] * 100
    baseline_final = float(baseline_trajectory[-1]) * 100
    summary = (
        f"{scenario.name}: projected health at 5yr = {final_health:.1f}% "
        f"(vs {baseline_final:.1f}% without intervention). "
        f"Improvement: +{(final_health - baseline_final):.1f}pp"
    )

    return InterventionResult(
        scenario=scenario,
        projected_health=projected,
        projected_health_100=[h * 100 for h in projected],
        improvement_vs_baseline=improvement,
        relative_improvement=relative,
        cost_effectiveness=cost_eff,
        summary=summary,
    )


def run_intervention_analysis(
    baseline_trajectory: np.ndarray,
    driver_strengths: np.ndarray,
    current_health: float,
    scenarios: list[InterventionScenario] | None = None,
    horizon_days: list[int] | None = None,
    config: dict | None = None,
) -> InterventionAnalysis:
    """Run full intervention analysis across all scenarios.
    
    Args:
        baseline_trajectory: (H,) no-intervention health scores.
        driver_strengths: (8,) current driver strengths.
        current_health: Current health score [0, 1].
        scenarios: List of interventions (uses defaults if None).
        horizon_days: Horizon offsets in days.
        config: Configuration dict.
        
    Returns:
        InterventionAnalysis with all scenario results and recommendation.
    """
    if scenarios is None:
        # Build from config or use defaults
        if config and "scenarios" in config:
            scenarios = []
            for s_cfg in config["scenarios"]:
                scenarios.append(InterventionScenario(
                    name=s_cfg.get("name", "Unknown"),
                    description="",
                    driver_reduction=s_cfg.get("driver_reduction", {}),
                ))
        else:
            scenarios = DEFAULT_SCENARIOS

    horizon_days = horizon_days or [30, 180, 365, 1095, 1825]
    horizon_names = ["30 days", "6 months", "1 year", "3 years", "5 years"]

    results = []
    baseline_result = None

    for scenario in scenarios:
        result = simulate_intervention(
            baseline_trajectory, driver_strengths, scenario, horizon_days
        )
        results.append(result)
        if scenario.name == "No Intervention" or not scenario.driver_reduction:
            baseline_result = result

    if baseline_result is None:
        baseline_result = results[0]

    # Find recommended intervention (best cost-effectiveness, excluding no-intervention)
    intervention_results = [r for r in results if r.scenario.driver_reduction]
    recommended = None
    if intervention_results:
        recommended = max(intervention_results, key=lambda r: r.cost_effectiveness)

    return InterventionAnalysis(
        baseline=baseline_result,
        scenarios=results,
        recommended=recommended,
        horizons=horizon_names[:len(horizon_days)],
        current_health=current_health,
    )
