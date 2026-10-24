"""
EO-specific data augmentations and transforms for ForestMind.

These transforms respect the physical properties of satellite imagery:
- Spectral band correlations are preserved
- SAR speckle statistics are maintained
- Temporal ordering is preserved
- Spatial augmentations are applied consistently across time steps
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class TemporalJitter:
    """Randomly perturb observation timestamps to simulate irregular revisit intervals.
    
    Preserves sequence length T for batch stacking while testing the model's
    continuous temporal positional encoding.
    """

    def __init__(self, max_offset_days: float = 7.0, drop_prob: float = 0.0):
        self.max_offset_days = max_offset_days
        self.drop_prob = drop_prob

    def __call__(self, sample: dict) -> dict:
        if "synthetic" in sample and not bool(sample["synthetic"]):
            return sample  # Real acquisition dates also anchor aligned context and targets.
        T = sample["optical"].shape[0]
        if T > 2 and "timestamps" in sample:
            jitter = np.random.uniform(-self.max_offset_days, self.max_offset_days, size=T)
            jitter[0] = 0.0
            jitter[-1] = 0.0
            sample["timestamps"] = np.maximum.accumulate(sample["timestamps"] + jitter)
        return sample


class SpectralJitter:
    """Add physically-motivated spectral noise.
    
    Models atmospheric effects and sensor noise as band-correlated noise.
    """

    def __init__(self, noise_std: float = 0.01, atmospheric_shift: float = 0.005):
        self.noise_std = noise_std
        self.atmospheric_shift = atmospheric_shift

    def __call__(self, sample: dict) -> dict:
        optical = sample["optical"].copy()

        # Band-independent noise (sensor noise)
        noise = np.random.normal(0, self.noise_std, optical.shape).astype(np.float32)
        optical += noise

        # Band-correlated atmospheric shift (shifts all bands together)
        shift = np.random.normal(0, self.atmospheric_shift)
        optical += shift

        sample["optical"] = np.clip(optical, 0.0, 1.0)
        if "optical_mask" in sample:
            sample["optical"][~sample["optical_mask"]] = 0
        return sample


class SpatialFlipRotate:
    """Random spatial flips and 90° rotations, applied consistently across time."""

    def __call__(self, sample: dict) -> dict:
        # Random horizontal flip
        if np.random.random() > 0.5:
            sample["optical"] = np.flip(sample["optical"], axis=-1).copy()
            sample["sar"] = np.flip(sample["sar"], axis=-1).copy()
            if "quality_mask" in sample:
                sample["quality_mask"] = np.flip(sample["quality_mask"], axis=-1).copy()
            if "sar_quality_mask" in sample:
                sample["sar_quality_mask"] = np.flip(sample["sar_quality_mask"], axis=-1).copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            sample["optical"] = np.flip(sample["optical"], axis=-2).copy()
            sample["sar"] = np.flip(sample["sar"], axis=-2).copy()
            if "quality_mask" in sample:
                sample["quality_mask"] = np.flip(sample["quality_mask"], axis=-2).copy()
            if "sar_quality_mask" in sample:
                sample["sar_quality_mask"] = np.flip(sample["sar_quality_mask"], axis=-2).copy()

        # Random 90° rotation (0, 90, 180, 270)
        k = np.random.randint(0, 4)
        if k > 0:
            sample["optical"] = np.rot90(sample["optical"], k=k, axes=(-2, -1)).copy()
            sample["sar"] = np.rot90(sample["sar"], k=k, axes=(-2, -1)).copy()
            if "quality_mask" in sample:
                sample["quality_mask"] = np.rot90(sample["quality_mask"], k=k, axes=(-2, -1)).copy()
            if "sar_quality_mask" in sample:
                sample["sar_quality_mask"] = np.rot90(sample["sar_quality_mask"], k=k, axes=(-2, -1)).copy()

        return sample


class RandomCrop:
    """Random spatial crop from a larger patch, consistent across time steps."""

    def __init__(self, crop_size: int = 96):
        self.crop_size = crop_size

    def __call__(self, sample: dict) -> dict:
        _, _, H, W = sample["optical"].shape
        if H <= self.crop_size or W <= self.crop_size:
            return sample

        y = np.random.randint(0, H - self.crop_size + 1)
        x = np.random.randint(0, W - self.crop_size + 1)

        sample["optical"] = sample["optical"][:, :, y:y+self.crop_size, x:x+self.crop_size]
        sample["sar"] = sample["sar"][:, :, y:y+self.crop_size, x:x+self.crop_size]
        if "quality_mask" in sample:
            sample["quality_mask"] = sample["quality_mask"][:, y:y+self.crop_size, x:x+self.crop_size]
        if "sar_quality_mask" in sample:
            sample["sar_quality_mask"] = sample["sar_quality_mask"][:, y:y+self.crop_size, x:x+self.crop_size]
        return sample


class Normalize:
    """Normalize spectral bands to zero mean, unit variance.
    
    Uses pre-computed statistics from the synthetic data distribution.
    """

    def __init__(
        self,
        optical_mean: np.ndarray | None = None,
        optical_std: np.ndarray | None = None,
        sar_mean: np.ndarray | None = None,
        sar_std: np.ndarray | None = None,
        climate_mean=None, climate_std=None,
        activity_mean=None, activity_std=None,
    ):
        # Default statistics (approximated from spectral profiles)
        self.optical_mean = optical_mean if optical_mean is not None else np.array(
            [0.05, 0.06, 0.08, 0.08, 0.10, 0.15, 0.18, 0.20, 0.19, 0.15, 0.15, 0.10],
            dtype=np.float32,
        )
        self.optical_std = optical_std if optical_std is not None else np.array(
            [0.03, 0.03, 0.04, 0.05, 0.05, 0.08, 0.09, 0.10, 0.09, 0.07, 0.07, 0.07],
            dtype=np.float32,
        )
        self.sar_mean = sar_mean if sar_mean is not None else np.array(
            [0.40, 0.28], dtype=np.float32
        )
        self.sar_std = sar_std if sar_std is not None else np.array(
            [0.18, 0.15], dtype=np.float32
        )
        for name, count in (("optical_mean", 12), ("optical_std", 12),
                            ("sar_mean", 2), ("sar_std", 2)):
            values = np.asarray(getattr(self, name), dtype=np.float32)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain {count} finite band statistics")
            if name.endswith("std") and (values <= 0).any():
                raise ValueError(f"{name} values must be positive")
            setattr(self, name, values)
        self.context_stats = {}
        for name, mean, std in (("climate", climate_mean, climate_std), ("activity", activity_mean, activity_std)):
            if (mean is None) != (std is None):
                raise ValueError(f"Both {name} mean and std are required")
            if mean is not None:
                mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
                if (mean.ndim != 1 or not 1 <= len(mean) <= 128 or mean.shape != std.shape
                        or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
                    raise ValueError(f"{name} normalization needs matching finite means and positive standard deviations")
            self.context_stats[name] = (mean, std)

    def __call__(self, sample: dict) -> dict:
        # Normalize optical: (T, C, H, W), stats broadcast over (T, 1, H, W)
        sample["optical"] = (
            (sample["optical"] - self.optical_mean[None, :, None, None])
            / (self.optical_std[None, :, None, None] + 1e-8)
        )
        sample["sar"] = (
            (sample["sar"] - self.sar_mean[None, :, None, None])
            / (self.sar_std[None, :, None, None] + 1e-8)
        )
        for modality in ("optical", "sar"):
            if f"{modality}_mask" in sample:
                sample[modality][~sample[f"{modality}_mask"]] = 0
        if "quality_mask" in sample:
            sample["optical"] = np.where(sample["quality_mask"][:, None], sample["optical"], 0)
        if "sar_quality_mask" in sample:
            sample["sar"] = np.where(sample["sar_quality_mask"][:, None], sample["sar"], 0)
        for name in ("climate", "activity"):
            key = f"{name}_context"
            if key not in sample:
                continue
            mean, std = self.context_stats[name]
            if mean is None:
                if f"{name}_mask" in sample:
                    raise ValueError(f"Measured {name} requires training/checkpoint normalization statistics")
                continue
            values = np.asarray(sample[key])
            mask = sample.get(f"{name}_mask")
            if (values.ndim != 2 or values.shape[1] != len(mean) or mask is None
                    or mask.shape != values.shape or mask.dtype.kind != "b"):
                raise ValueError(f"{name} values/mask do not match normalization features")
            if not np.isfinite(values[mask]).all():
                raise ValueError(f"Observed {name} must be finite")
            sample[key] = np.where(mask, (np.where(mask, values, mean) - mean) / std, 0).astype(np.float32)
        return sample


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""

    def __call__(self, sample: dict) -> dict:
        result = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                if value.dtype.kind in "bifu":
                    tensor = torch.from_numpy(np.array(value, copy=True, order="C"))
                    result[key] = tensor.float() if tensor.is_floating_point() else tensor
                else:
                    result[key] = value.tolist()
            elif key == "health_state":
                result[key] = torch.tensor(value, dtype=torch.long)
            elif isinstance(value, (float, np.floating)):
                result[key] = torch.tensor(value, dtype=torch.float32)
            else:
                result[key] = value
        return result


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def get_train_transforms(crop_size: int = 96, normalization: dict | None = None) -> Compose:
    """Standard training transforms with augmentation."""
    return Compose([
        TemporalJitter(drop_prob=0.1),
        SpectralJitter(noise_std=0.01),
        SpatialFlipRotate(),
        # RandomCrop(crop_size=crop_size),  # Uncomment for larger patches
        Normalize(**(normalization or {})),
        ToTensor(),
    ])


def get_eval_transforms(normalization: dict | None = None) -> Compose:
    """Evaluation transforms (no augmentation)."""
    return Compose([
        Normalize(**(normalization or {})),
        ToTensor(),
    ])
