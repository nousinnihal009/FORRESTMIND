"""
Evaluate a trained ForestMind model.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forestmind.data.dataset import create_dataloaders
from forestmind.models.forestmind_model import ForestMindModel
from forestmind.training.losses import ForestMindLoss
from forestmind.training.metrics import ForestMindMetrics


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ForestMind")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Split to evaluate")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("evaluate")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Use checkpoint config if available
    if "config" in checkpoint:
        config = checkpoint["config"]
        logger.info("Using config from checkpoint")

    # Create data
    data_cfg = config.get("data", {})
    _, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=data_cfg.get("batch_size", 4),
        num_workers=data_cfg.get("num_workers", 0),
    )
    eval_loader = test_loader if args.split == "test" else val_loader

    # Create and load model
    model = ForestMindModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Evaluate
    criterion = ForestMindLoss(config.get("training", {}).get("loss_weights", {}))
    metrics = ForestMindMetrics()

    total_loss = 0.0
    num_batches = 0

    logger.info(f"Evaluating on {args.split} split ({len(eval_loader.dataset)} samples)...")

    with torch.no_grad():
        for batch in eval_loader:
            inputs = {
                "optical": batch["optical"].to(device),
                "sar": batch["sar"].to(device),
                "timestamps": batch["timestamps"].to(device),
            }
            targets = {
                "health_state": batch["health_state"].to(device),
                "health_score": batch["health_score"].to(device),
                "drivers": batch["drivers"].to(device),
                "driver_strengths": batch["driver_strengths"].to(device),
                "trajectory": batch["trajectory"].to(device),
            }

            outputs = model(**inputs)
            losses = criterion(outputs, targets)
            total_loss += losses["total"].item()
            num_batches += 1

            metrics.update(outputs, targets)

    # Report
    result = metrics.compute()
    avg_loss = total_loss / max(1, num_batches)

    print("\n" + "=" * 60)
    print(f"  ForestMind Evaluation — {args.split} split")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print("=" * 60)
    print(f"\n  Average Loss: {avg_loss:.4f}\n")
    print(metrics.summary_str(result))
    print()

    # Per-driver attribution F1
    driver_names = [
        "Agriculture", "Cattle Pasture", "Mining", "Logging",
        "Road Construction", "Wildfire", "Drought", "Urban Expansion",
    ]
    if result.attribution_per_driver_f1:
        print("  Per-driver F1:")
        for name, f1 in zip(driver_names, result.attribution_per_driver_f1):
            print(f"    {name:20s}: {f1:.3f}")


if __name__ == "__main__":
    main()
