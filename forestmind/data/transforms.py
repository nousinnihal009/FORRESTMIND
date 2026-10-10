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
        return sample


class SpatialFlipRotate:
    """Random spatial flips and 90° rotations, applied consistently across time."""

    def __call__(self, sample: dict) -> dict:
        # Random horizontal flip
        if np.random.random() > 0.5:
            sample["optical"] = np.flip(sample["optical"], axis=-1).copy()
            sample["sar"] = np.flip(sample["sar"], axis=-1).copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            sample["optical"] = np.flip(sample["optical"], axis=-2).copy()
            sample["sar"] = np.flip(sample["sar"], axis=-2).copy()

        # Random 90° rotation (0, 90, 180, 270)
        k = np.random.randint(0, 4)
        if k > 0:
            sample["optical"] = np.rot90(sample["optical"], k=k, axes=(-2, -1)).copy()
            sample["sar"] = np.rot90(sample["sar"], k=k, axes=(-2, -1)).copy()

        return sample


class RandomCrop:
    """Random spatial crop from a larger patch, consistent across time steps."""

    def __init__(self, crop_size: int = 96):
        self.crop_size = crop_size

    def __call__(self, sample: dict) -> dict:
        _, _, H, W = sample["optical"].shape
        if H <= self.crop_size or W <= self.crop_size:
            return sample

        y = np.random.randint(0, H - self.crop_size)
        x = np.random.randint(0, W - self.crop_size)

        sample["optical"] = sample["optical"][:, :, y:y+self.crop_size, x:x+self.crop_size]
        sample["sar"] = sample["sar"][:, :, y:y+self.crop_size, x:x+self.crop_size]
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
        return sample


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""

    def __call__(self, sample: dict) -> dict:
        return {
            "optical": torch.from_numpy(sample["optical"].copy()),
            "sar": torch.from_numpy(sample["sar"].copy()),
            "timestamps": torch.from_numpy(sample["timestamps"].copy()),
            "health_state": torch.tensor(sample["health_state"], dtype=torch.long),
            "health_score": torch.tensor(sample["health_score"], dtype=torch.float32),
            "drivers": torch.from_numpy(sample["drivers"].copy()),
            "driver_strengths": torch.from_numpy(sample["driver_strengths"].copy()),
            "trajectory": torch.from_numpy(sample["trajectory"].copy()),
            "health_history": torch.from_numpy(sample["health_history"].copy()),
        }


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def get_train_transforms(crop_size: int = 96) -> Compose:
    """Standard training transforms with augmentation."""
    return Compose([
        TemporalJitter(drop_prob=0.1),
        SpectralJitter(noise_std=0.01),
        SpatialFlipRotate(),
        # RandomCrop(crop_size=crop_size),  # Uncomment for larger patches
        Normalize(),
        ToTensor(),
    ])


def get_eval_transforms() -> Compose:
    """Evaluation transforms (no augmentation)."""
    return Compose([
        Normalize(),
        ToTensor(),
    ])
