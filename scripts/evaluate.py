"""Evaluate a ForestMind checkpoint and optionally save an auditable JSON report.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split test --output artifacts/evaluation.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forestmind.training.evaluation import evaluate_checkpoint, write_evaluation_report


DRIVER_NAMES = [
    "Agriculture", "Cattle Pasture", "Mining", "Logging",
    "Road Construction", "Wildfire", "Drought", "Urban Expansion",
]


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate ForestMind with reproducible input provenance")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path (safe weights-only loading)")
    parser.add_argument("--config", default=None, help="Optional evaluation config override")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Explicit split to evaluate")
    parser.add_argument("--output", type=Path, help="Save metrics, provenance and every example's raw predictions as JSON")
    args = parser.parse_args(argv)
    setup_logging()
    try:
        report = evaluate_checkpoint(args.checkpoint, split=args.split, config_path=args.config)
        if args.output:
            saved = write_evaluation_report(report, args.output)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        logging.getLogger("evaluate").error("Evaluation failed: %s", exc)
        return 2
    result = report["metrics"]
    print("\n" + "=" * 60)
    print(f"  ForestMind Evaluation -- {report['dataset']['split']} split")
    print(f"  Checkpoint: {report['checkpoint']['path']}")
    print(f"  SHA256: {report['checkpoint']['sha256']}")
    print(f"  Epoch: {report['checkpoint']['epoch']}")
    print(f"  Samples: {report['counts']['samples']}; batches: {report['counts']['batches']}")
    print("=" * 60)
    if report["synthetic_warning"]:
        print(report["synthetic_warning"])
    if args.split == "val":
        print("VALIDATION SPLIT: development metrics, not held-out test performance.")
    print("Uncalibrated outputs; no ecological tipping or causal attribution claim.")
    print(f"\n  Average Loss:        {report['losses']['total']:.4f}")
    for label, key in (
        ("Health Accuracy", "health_accuracy"), ("Health F1 (macro)", "health_f1_macro"),
        ("Health MAE (0-100)", "health_mae"), ("Attribution F1", "attribution_f1"),
        ("Attribution Precision", "attribution_precision"), ("Attribution Recall", "attribution_recall"),
        ("Trajectory MAE (0-1)", "trajectory_mae"), ("Trajectory RMSE (0-1)", "trajectory_rmse"),
        ("Aggregate Score", "aggregate_score"),
    ):
        rendered = "unavailable" if result[key] is None else f"{result[key]:.4f}"
        print(f"  {label:23s}: {rendered}")
    print("  Per-horizon MAE (0-1):")
    for horizon in report["per_horizon"]:
        rendered = "unavailable" if horizon['mae'] is None else f"{horizon['mae']:.4f}"
        print(f"    {horizon['horizon_days']:>5d} days: {rendered} (n={horizon['n']})")
    print("  Per-driver F1:")
    for index, f1 in enumerate(result["attribution_per_driver_f1"]):
        name = DRIVER_NAMES[index] if index < len(DRIVER_NAMES) else f"Driver {index}"
        rendered = "unavailable" if f1 is None else f"{f1:.3f}"
        print(f"    {name:20s}: {rendered}")
    if args.output:
        print(f"\n  JSON report: {saved}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
