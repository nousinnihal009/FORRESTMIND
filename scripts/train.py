"""
Train ForestMind model.

Usage:
    python scripts/train.py [--config configs/default.yaml]
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
from forestmind.training.trainer import ForestMindTrainer


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train ForestMind")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("train")

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Apply overrides
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr

    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Create data loaders
    data_cfg = config.get("data", {})
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=data_cfg.get("batch_size", 4),
        num_workers=data_cfg.get("num_workers", 0),
    )
    logger.info(
        f"Data: {len(train_loader.dataset)} train, "
        f"{len(val_loader.dataset)} val, "
        f"{len(test_loader.dataset)} test"
    )

    # Create model
    logger.info("Building model...")
    model = ForestMindModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Create trainer
    trainer = ForestMindTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from checkpoint: {args.resume}")

    # Train
    result = trainer.train()

    logger.info("\n" + "=" * 50)
    logger.info("  Training Summary")
    logger.info(f"  Best val score: {result['best_val_score']:.4f}")
    logger.info(f"  Total time:     {result['total_time'] / 60:.1f} minutes")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
