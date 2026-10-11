"""Conditional conservation scenarios with explicit, user-reviewable assumptions.

This is a sensitivity calculator. Driver scores are not causal effect estimates,
scenario reduction factors are assumptions, and rankings are not policy advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DRIVER_NAMES = [
    "agriculture", "cattle_pasture", "mining", "logging",
    "road_construction", "wildfire", "drought", "urban_expansion",
]
DRIVER_ALIASES = {"cattle": "cattle_pasture", "road": "road_construction", "fire": "wildfire"}
DEFAULT_HORIZONS = [30, 180, 365, 1095, 1825]


def resolve_horizons(horizon_days: list[int] | None, length: int) -> list[int]:
    days = DEFAULT_HORIZONS.copy() if horizon_days is None else list(horizon_days)
    if len(days) != length or not days:
        raise ValueError("horizon_days must have one entry per nonempty trajectory value")
    if any(isinstance(day, (bool, np.bool_)) or not isinstance(day, (int, np.integer))
           or day <= 0 for day in days) or any(b <= a for a, b in zip(days, days[1:])):
        raise ValueError("horizon_days must be positive, strictly increasing integers")
    return [int(day) for day in days]


def horizon_label(day: int) -> str:
    if day == 365:
        return "1 year"
    if day % 365 == 0:
        return f"{day // 365} years"
    return f"{day} days"


def _unit_array(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    if np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


@dataclass
class InterventionScenario:
    """Assumed driver attenuation and an arbitrary relative cost, not measured effects."""

    name: str
    description: str
    driver_reduction: dict[str, float]
    cost_relative: float = 1.0
    implementation_delay_days: int = 0

    def __post_init__(self):
        normalized = {}
        for driver, reduction in self.driver_reduction.items():
            driver = DRIVER_ALIASES.get(driver, driver)
            if driver not in DRIVER_NAMES:
                raise ValueError(f"Unknown intervention driver: {driver}")
            reduction = float(reduction)
            if not np.isfinite(reduction) or not 0 <= reduction <= 1:
                raise ValueError("driver_reduction values must be finite and lie in [0, 1]")
            if driver in normalized:
                raise ValueError(f"Duplicate intervention driver after alias resolution: {driver}")
            normalized[driver] = reduction
        self.driver_reduction = normalized
        self.cost_relative = float(self.cost_relative)
        if not np.isfinite(self.cost_relative) or self.cost_relative < 0:
            raise ValueError("cost_relative must be finite and non-negative")
        if any(normalized.values()) and self.cost_relative == 0:
            raise ValueError("Active scenarios require a positive relative cost for ranking")
        if (not isinstance(self.implementation_delay_days, (int, np.integer))
                or self.implementation_delay_days < 0):
            raise ValueError("implementation_delay_days must be a non-negative integer")


@dataclass
class InterventionResult:
    scenario: InterventionScenario
    projected_health: list[float]
    projected_health_100: list[float]
    improvement_vs_baseline: list[float]
    relative_improvement: list[float]
    cost_effectiveness: float
    summary: str
    assumptions: list[str] = field(default_factory=list)
    evidence_status: str = "hypothetical_sensitivity_analysis"


@dataclass
class InterventionAnalysis:
    baseline: InterventionResult
    scenarios: list[InterventionResult]
    recommended: InterventionResult | None  # Highest modeled benefit/cost, not a causal recommendation.
    horizons: list[str]
    current_health: float
    limitations: list[str] = field(default_factory=list)
    ranking_basis: str = "mean_assumed_health_difference_per_relative_cost"


DEFAULT_SCENARIOS = [
    InterventionScenario("No Intervention", "Provided baseline projection.", {}, 0),
    InterventionScenario("Conservation Corridor", "Assume reduced logging and road pressure.",
                         {"logging": 0.6, "road_construction": 0.4}, 0.6),
    InterventionScenario("Fire Management", "Assume reduced wildfire pressure.", {"wildfire": 0.7}, 0.4),
    InterventionScenario("Agricultural Buffer Zone", "Assume reduced agricultural and grazing pressure.",
                         {"agriculture": 0.5, "cattle_pasture": 0.6}, 0.5),
    InterventionScenario("Mining Regulation", "Assume reduced mining and road pressure.",
                         {"mining": 0.8, "road_construction": 0.3}, 0.7),
    InterventionScenario("Full Protection", "Illustrative combined reduction in human pressure.",
                         {"agriculture": 0.8, "cattle_pasture": 0.8, "mining": 0.9,
                          "logging": 0.9, "road_construction": 0.7, "urban_expansion": 0.5}, 1.0),
    InterventionScenario("Reforestation + Protection", "Illustrative protection assumptions; no growth benefit is assumed.",
                         {"agriculture": 0.8, "cattle_pasture": 0.8, "mining": 0.9,
                          "logging": 0.9, "road_construction": 0.7, "wildfire": 0.5,
                          "urban_expansion": 0.5}, 1.5, 90),
]


def simulate_intervention(
    baseline_trajectory: np.ndarray,
    driver_strengths: np.ndarray,
    scenario: InterventionScenario,
    horizon_days: list[int] | None = None,
    *,
    current_health: float | None = None,
) -> InterventionResult:
    """Attenuate baseline decline under assumed driver reductions and a 180-day ramp.

    Improvement is bounded by the baseline decline from current health. No
    additional recovery is invented, and an improving baseline stays intact.
    Legacy calls omitting current_health use the first forecast as a reference,
    explicitly recorded in the result assumptions.
    """
    baseline = _unit_array(baseline_trajectory, "baseline_trajectory")
    drivers = _unit_array(driver_strengths, "driver_strengths")
    if len(drivers) != len(DRIVER_NAMES):
        raise ValueError(f"driver_strengths must contain {len(DRIVER_NAMES)} values")
    days = resolve_horizons(horizon_days, len(baseline))
    assumptions = [
        "Driver scores and reduction factors are scenario assumptions, not identified causal effects.",
        "Assumed driver reductions affect baseline decline proportionally, with a 180-day implementation ramp.",
        "No additional recovery, displacement, ecological side effects or policy uncertainty are modeled.",
        "Relative costs are illustrative units; ranking does not establish real-world cost effectiveness.",
    ]
    if current_health is None:
        current_health = float(baseline[0])
        assumptions.append("Current health was omitted; the first forecast is used as the comparison reference.")
    current = float(current_health)
    if not np.isfinite(current) or not 0 <= current <= 1:
        raise ValueError("current_health must be finite and lie in [0, 1]")
    reductions = np.asarray([scenario.driver_reduction.get(name, 0) for name in DRIVER_NAMES])
    pressure_reduction = float(np.dot(drivers, reductions) / drivers.sum()) if drivers.sum() else 0.0
    # A zero-pressure scenario must reproduce every baseline value, including increases.
    ramp = np.clip((np.asarray(days) - scenario.implementation_delay_days) / 180, 0, 1)
    decline = np.maximum(0, current - baseline)
    improvement = decline * pressure_reduction * ramp
    projected = np.clip(baseline + improvement, 0, 1)
    relative = np.divide(improvement * 100, decline, out=np.zeros_like(decline), where=decline > 0)
    cost_effectiveness = float(improvement.mean() / scenario.cost_relative) if scenario.cost_relative else 0.0
    summary = (
        f"{scenario.name}: conditional health at {horizon_label(days[-1])} = {projected[-1] * 100:.1f}/100 "
        f"(baseline {baseline[-1] * 100:.1f}/100); assumed difference +{improvement[-1] * 100:.1f} points. "
        "Hypothetical scenario, not a validated intervention effect."
    )
    return InterventionResult(
        scenario=scenario, projected_health=projected.tolist(),
        projected_health_100=(projected * 100).tolist(),
        improvement_vs_baseline=improvement.tolist(), relative_improvement=relative.tolist(),
        cost_effectiveness=cost_effectiveness, summary=summary, assumptions=assumptions,
    )


def run_intervention_analysis(
    baseline_trajectory: np.ndarray,
    driver_strengths: np.ndarray,
    current_health: float,
    scenarios: list[InterventionScenario] | None = None,
    horizon_days: list[int] | None = None,
    config: dict | None = None,
) -> InterventionAnalysis:
    """Compare assumptions and rank positive modeled benefits per relative cost."""
    config = config or {}
    if scenarios is None:
        if "scenarios" in config:
            scenarios = [InterventionScenario(
                name=item.get("name", "Unnamed scenario"),
                description=item.get("description", "User-defined hypothetical scenario."),
                driver_reduction=item.get("driver_reduction", {}),
                cost_relative=item.get("cost_relative", 1.0),
                implementation_delay_days=item.get("implementation_delay_days", 0),
            ) for item in config["scenarios"]]
        else:
            scenarios = list(DEFAULT_SCENARIOS)
    days = resolve_horizons(horizon_days, len(baseline_trajectory))
    # Always construct a real no-intervention baseline, even for an empty scenario list.
    baseline = simulate_intervention(
        baseline_trajectory, driver_strengths,
        InterventionScenario("No Intervention", "Provided baseline projection.", {}, 0),
        days, current_health=current_health,
    )
    results = [simulate_intervention(baseline_trajectory, driver_strengths, scenario, days,
                                    current_health=current_health) for scenario in scenarios]
    if not any(not result.scenario.driver_reduction for result in results):
        results.insert(0, baseline)
    candidates = [result for result in results
                  if result.scenario.driver_reduction and result.cost_effectiveness > 0]
    ranked = max(candidates, key=lambda result: result.cost_effectiveness) if candidates else None
    return InterventionAnalysis(
        baseline=baseline, scenarios=results, recommended=ranked,
        horizons=[horizon_label(day) for day in days], current_health=float(current_health),
        limitations=[
            "Scenario differences are conditional on explicit assumptions; no causal intervention effects are established.",
            "The recommended field is the highest assumed benefit/cost scenario, not an evidence-based policy recommendation.",
        ],
    )
