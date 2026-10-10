"""
ForestMind demo — run the full prediction pipeline on synthetic data.

Generates a synthetic sample, runs the model (untrained weights),
then performs ecological risk analysis and intervention simulation.
Demonstrates the complete ForestMind pipeline end-to-end.

Usage:
    python scripts/demo_prediction.py [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    # Force UTF-8 output on Windows
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="ForestMind demo prediction")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional trained checkpoint")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for sample generation")
    parser.add_argument("--visualize", action="store_true", help="Show matplotlib plots")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("demo")

    # ─── Generate a Synthetic Sample ──────────────────────
    logger.info("Generating synthetic satellite observation...")
    from forestmind.data.synthetic import SyntheticForestGenerator, HEALTH_STATES, DRIVER_CLASSES
    from forestmind.data.transforms import Normalize, ToTensor, Compose

    generator = SyntheticForestGenerator(seed=args.seed)
    sample = generator.generate_sample()

    logger.info(f"  Patch: {sample['optical'].shape[0]} timesteps × "
                f"{sample['optical'].shape[1]} bands × "
                f"{sample['optical'].shape[2]}×{sample['optical'].shape[3]} pixels")
    logger.info(f"  Ground-truth health: {HEALTH_STATES[sample['health_state']]} "
                f"({sample['health_score']:.1f}/100)")
    active = [DRIVER_CLASSES[i] for i, d in enumerate(sample['drivers']) if d > 0]
    logger.info(f"  Ground-truth drivers: {active if active else 'None'}")

    # ─── Build Model ─────────────────────────────────────
    import yaml
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Use smaller config for demo
    config["foundation"]["custom_vit"]["depth"] = 4
    config["temporal"]["depth"] = 2

    logger.info("Building ForestMind model...")
    from forestmind.models.forestmind_model import ForestMindModel

    model = ForestMindModel(config)

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    # ─── Run Prediction ──────────────────────────────────
    logger.info("Running model prediction...")

    # Prepare input
    transform = Compose([Normalize(), ToTensor()])
    transformed = transform(sample)

    optical = transformed["optical"].unsqueeze(0)   # (1, T, 12, H, W)
    sar = transformed["sar"].unsqueeze(0)            # (1, T, 2, H, W)
    timestamps = transformed["timestamps"].unsqueeze(0)  # (1, T)

    prediction = model.predict(optical, sar, timestamps)

    print("\n" + "=" * 60)
    print("  FORESTMIND MODEL PREDICTION")
    print("=" * 60)
    print(f"\n  Health State:  {prediction['health_state']}")
    print(f"  Health Score:  {prediction['health_score']}/100")
    print(f"  Confidence:    {prediction['health_confidence']}%")

    if prediction["active_drivers"]:
        print("\n  Active Drivers:")
        for d in prediction["active_drivers"]:
            print(f"    - {d['name']:20s} conf={d['confidence']:.1f}%  strength={d['strength']:.1f}%")
    else:
        print("\n  No active degradation drivers detected")

    print("\n  Predicted Trajectory:")
    for horizon, score in zip(prediction["trajectory"]["horizons"], prediction["trajectory"]["health_scores"]):
        print(f"    {horizon:>10s} -> {score:.1f}/100")

    if "uncertainty" in prediction["trajectory"]:
        print("\n  Uncertainty (±):")
        for horizon, unc in zip(prediction["trajectory"]["horizons"], prediction["trajectory"]["uncertainty"]):
            print(f"    {horizon:>10s} -> +/-{unc:.1f}")

    # ─── Ecological Risk Analysis ─────────────────────────
    logger.info("\nRunning ecological risk analysis...")
    from forestmind.risk.engine import EcologicalRiskEngine

    risk_engine = EcologicalRiskEngine(config.get("risk_engine", {}))

    # Convert prediction to risk engine input
    driver_names = [
        "Agriculture", "Cattle Pasture", "Mining", "Logging",
        "Road Construction", "Wildfire", "Drought", "Urban Expansion",
    ]

    report = risk_engine.analyze(
        health_history=sample["health_history"],
        health_state=model.predict(optical, sar, timestamps)["health_state"],
        health_score=prediction["health_score"],
        health_confidence=prediction["health_confidence"],
        active_drivers=prediction["active_drivers"],
        driver_strengths=sample["driver_strengths"],
        trajectory_scores=np.array(prediction["trajectory"]["health_scores"]),
        trajectory_uncertainty=np.array(prediction["trajectory"].get("uncertainty", [0]*5)),
        location="Synthetic Forest Patch (Demo)",
    )

    # Print full report
    print("\n" + report.summary())

    # ─── Visualization ───────────────────────────────────
    if args.visualize:
        logger.info("Generating visualizations...")
        try:
            from forestmind.utils.visualization import (
                plot_health_trajectory,
                plot_driver_attribution,
                plot_intervention_comparison,
                plot_tipping_point_dashboard,
                plot_rgb_composite,
            )

            plot_rgb_composite(sample["optical"])
            
            plot_health_trajectory(
                sample["health_history"],
                predicted_trajectory=np.array(prediction["trajectory"]["health_scores"]) / 100,
                predicted_uncertainty=np.array(prediction["trajectory"].get("uncertainty", [0]*5)) / 100,
                horizon_labels=prediction["trajectory"]["horizons"],
            )

            if prediction["active_drivers"]:
                all_driver_probs = np.zeros(8)
                all_driver_strengths = np.zeros(8)
                for d in prediction["active_drivers"]:
                    idx = driver_names.index(d["name"])
                    all_driver_probs[idx] = d["confidence"] / 100
                    all_driver_strengths[idx] = d["strength"] / 100

                plot_driver_attribution(driver_names, all_driver_probs, all_driver_strengths)

            if report.tipping_point:
                plot_tipping_point_dashboard(report.tipping_point, sample["health_history"])

            if report.interventions:
                plot_intervention_comparison(report.interventions)

        except ImportError as e:
            logger.warning(f"Visualization requires matplotlib: {e}")

    logger.info("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
