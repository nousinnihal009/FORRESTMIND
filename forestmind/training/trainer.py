"""
Training loop for ForestMind.

Handles:
  - Multi-task training with configurable loss weights
  - Learning rate scheduling (cosine, step, plateau)
  - Gradient clipping
  - Checkpoint saving/loading
  - Validation and metric logging
  - Mixed precision training (GPU only)
  - Early stopping
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from forestmind.models.forestmind_model import ForestMindModel
from forestmind.training.losses import ForestMindLoss
from forestmind.training.metrics import ForestMindMetrics

logger = logging.getLogger(__name__)


def _get_device(config: dict) -> torch.device:
    """Resolve device from config."""
    device_str = config.get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


class ForestMindTrainer:
    """Training orchestrator for ForestMind.
    
    Args:
        model: ForestMindModel instance.
        config: Full training configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
    """

    def __init__(
        self,
        model: ForestMindModel,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.config = config
        train_cfg = config.get("training", {})

        # Device
        self.device = _get_device(config)
        logger.info(f"Training on device: {self.device}")
        self.model = model.to(self.device)

        # Loss
        self.criterion = ForestMindLoss(train_cfg.get("loss_weights", {}))

        # Optimizer
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )

        # Scheduler
        self.epochs = train_cfg.get("epochs", 50)
        self.scheduler = self._build_scheduler(train_cfg)

        # Gradient clipping
        self.grad_clip = train_cfg.get("gradient_clip", 1.0)

        # Mixed precision
        self.use_amp = train_cfg.get("use_amp", False) and self.device.type == "cuda"
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = GradScaler(enabled=self.use_amp)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Metrics
        self.metrics = ForestMindMetrics()

        # Checkpointing
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = train_cfg.get("save_every", 5)

        # Warmup
        self.warmup_epochs = train_cfg.get("warmup_epochs", 5)
        self.base_lr = train_cfg.get("learning_rate", 1e-4)

        # State
        self.current_epoch = 0
        self.best_val_score = float("-inf")
        self.train_history = []
        self.val_history = []
        self.training_target_counts = {}

    def _build_scheduler(self, train_cfg: dict):
        """Build learning rate scheduler."""
        scheduler_type = train_cfg.get("scheduler", "cosine")
        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg.get("epochs", 50),
                eta_min=1e-6,
            )
        elif scheduler_type == "step":
            return StepLR(self.optimizer, step_size=15, gamma=0.1)
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5,
            )
        return None

    def _warmup_lr(self, epoch: int):
        """Linear learning rate warmup."""
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.base_lr * warmup_factor

    def _prepare_batch(self, batch: dict) -> tuple[dict, dict]:
        """Move batch to device and split into inputs/targets."""
        foundation = self.config.get("foundation", {})
        name = foundation.get("name")
        if name in ("prithvi", "terramind") and not foundation.get(name, {}).get("use_proxy", False):
            from forestmind.models.foundation.native import validate_native_sample
            for sample_index in range(batch["optical"].shape[0]):
                validate_native_sample(self.config, {key: batch[key][sample_index]
                                                     for key in ("optical", "observation_mask", "optical_mask", "sar_mask")
                                                     if key in batch})
        inputs = {
            "optical": batch["optical"].to(self.device),
            "sar": batch["sar"].to(self.device),
            "timestamps": batch["timestamps"].to(self.device),
        }
        if "label_valid" in batch and not torch.as_tensor(batch["label_valid"]).all():
            raise ValueError("Unlabeled satellite acquisitions cannot be used for supervised training/evaluation")
        required = {"health_state", "health_score", "drivers", "driver_strengths", "trajectory"}
        if not required.issubset(batch):
            raise ValueError(f"Supervised targets are missing: {sorted(required.difference(batch))}")
        for key in ("observation_mask", "optical_mask", "sar_mask", "quality_mask",
                    "sar_quality_mask", "climate_context", "activity_context", "climate_mask", "activity_mask"):
            if key in batch:
                inputs[key] = batch[key].to(self.device)
        targets = {
            "health_state": batch["health_state"].to(self.device),
            "health_score": batch["health_score"].to(self.device),
            "drivers": batch["drivers"].to(self.device),
            "driver_strengths": batch["driver_strengths"].to(self.device),
            "trajectory": batch["trajectory"].to(self.device),
        }
        from forestmind.training.targets import MASK_KEYS
        for key in MASK_KEYS:
            if key in batch:
                targets[key] = batch[key].to(self.device)
        scale = self.config.get("data", {}).get("trajectory_target_scale", "unit")
        if scale not in ("unit", "percent"):
            raise ValueError("trajectory_target_scale must be unit or percent")
        if scale == "percent":
            targets["trajectory"] = targets["trajectory"] / 100
        known = targets["trajectory"][targets.get("trajectory_mask", torch.ones_like(targets["trajectory"], dtype=torch.bool))]
        if not torch.isfinite(known).all() or ((known < 0) | (known > 1)).any():
            raise ValueError("Trajectory targets must use the declared scale, normalized to [0, 1]")
        return inputs, targets

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs}")

        for batch in pbar:
            inputs, targets = self._prepare_batch(batch)
            from forestmind.training.targets import supervision_masks
            for key, mask in supervision_masks(targets).items():
                count = mask.sum(dim=0).detach().cpu()
                self.training_target_counts[key] = self.training_target_counts.get(key, torch.zeros_like(count)) + count

            self.optimizer.zero_grad()

            # Forward
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(**inputs)
                losses = self.criterion(outputs, targets)

            # Backward
            self.scaler.scale(losses["total"]).backward()

            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        # Average losses
        avg_losses = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self) -> tuple[dict[str, float], float]:
        """Run validation and compute metrics.
        
        Returns:
            Tuple of (loss_dict, aggregate_score).
        """
        self.model.eval()
        self.metrics.reset()
        epoch_losses = {}
        num_batches = 0

        for batch in self.val_loader:
            inputs, targets = self._prepare_batch(batch)

            outputs = self.model(**inputs)
            losses = self.criterion(outputs, targets)

            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            num_batches += 1

            # Accumulate metrics
            self.metrics.update(outputs, targets)

        # Compute metrics
        metrics_result = self.metrics.compute()
        avg_losses = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}

        return avg_losses, metrics_result.aggregate_score

    def save_checkpoint(self, filename: str = "checkpoint.pt", is_best: bool = False):
        """Save model checkpoint."""
        from copy import deepcopy
        saved_config = deepcopy(self.config)
        foundation = saved_config.get("foundation", {})
        name = foundation.get("name")
        if name in ("prithvi", "terramind") and not foundation.get(name, {}).get("use_proxy", False):
            from forestmind.models.foundation.native import capture_native_runtime
            foundation["native_runtime"] = capture_native_runtime()
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_score": self.best_val_score,
            "config": saved_config,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "training_target_counts": {key: value.tolist() for key, value in self.training_target_counts.items()},
        }
        if self.training_target_counts:
            enabled = {
                "health_state": self.criterion.w_health * self.criterion.health_loss.ce_weight > 0,
                "health_score": self.criterion.w_health * self.criterion.health_loss.mse_weight > 0,
                "drivers": self.criterion.w_attribution * self.criterion.attribution_loss.bce_weight > 0,
                "driver_strengths": self.criterion.w_attribution * self.criterion.attribution_loss.strength_weight > 0,
                "trajectory": self.criterion.w_trajectory > 0 and (self.criterion.trajectory_loss.mse_weight > 0 or self.criterion.trajectory_loss.huber_weight > 0),
            }
            state["supervised_targets"] = {key: ((value > 0) & enabled[key]).tolist() for key, value in self.training_target_counts.items()}
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state, self.checkpoint_dir / filename)
        if is_best:
            torch.save(state, self.checkpoint_dir / "best_model.pt")
            logger.info(f"  ★ New best model saved (score: {self.best_val_score:.4f})")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.current_epoch = state.get("epoch", -1) + 1
        self.best_val_score = state.get("best_val_score", 0.0)
        self.train_history = state.get("train_history", [])
        self.val_history = state.get("val_history", [])
        self.training_target_counts = {key: torch.as_tensor(value) for key, value in state.get("training_target_counts", {}).items()}
        if self.scheduler and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self) -> dict:
        """Run full training loop.
        
        Returns:
            Training summary dictionary.
        """
        logger.info("=" * 60)
        logger.info("  Starting ForestMind Training")
        logger.info(f"  Epochs:     {self.epochs}")
        logger.info(f"  Device:     {self.device}")
        logger.info(f"  Train size: {len(self.train_loader.dataset)}")
        logger.info(f"  Val size:   {len(self.val_loader.dataset)}")
        logger.info(f"  Batch size: {self.train_loader.batch_size}")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Warmup
            self._warmup_lr(epoch)

            # Train
            train_losses = self.train_epoch()
            self.train_history.append(train_losses)

            # Validate
            val_losses, val_score = self.validate()
            self.val_history.append({"losses": val_losses, "score": val_score})

            # Scheduler step
            if self.scheduler is not None and epoch >= self.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_score)
                else:
                    self.scheduler.step()

            # Checkpointing
            is_best = val_score > self.best_val_score
            if is_best:
                self.best_val_score = val_score

            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(f"epoch_{epoch + 1:03d}.pt", is_best=is_best)

            # Logging
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} "
                f"[{epoch_time:.1f}s] "
                f"train_loss={train_losses['total']:.4f} "
                f"val_loss={val_losses['total']:.4f} "
                f"val_score={val_score:.4f} "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                f"{' ★' if is_best else ''}"
            )

        total_time = time.time() - start_time
        logger.info(f"\nTraining complete in {total_time / 60:.1f} minutes")
        logger.info(f"Best validation score: {self.best_val_score:.4f}")

        # Final validation with full metrics
        val_losses, val_score = self.validate()
        metrics_result = self.metrics.compute()
        logger.info("\n" + self.metrics.summary_str(metrics_result))

        return {
            "best_val_score": self.best_val_score,
            "total_time": total_time,
            "final_metrics": metrics_result,
        }
