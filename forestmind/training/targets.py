"""Explicit supervision masks: an unknown label never becomes a negative label."""
from __future__ import annotations

import torch

TARGET_KEYS = ("health_state", "health_score", "drivers", "driver_strengths", "trajectory")
MASK_KEYS = tuple(f"{key}_mask" for key in TARGET_KEYS)


def supervision_masks(targets: dict) -> dict:
    """Validate supplied masks and known labels; legacy complete labels remain supported."""
    masks = {}
    for name in TARGET_KEYS:
        value = targets[name]
        mask = targets.get(f"{name}_mask")
        if mask is None:
            mask = torch.ones_like(value, dtype=torch.bool)
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != value.shape:
            raise ValueError(f"{name}_mask must be boolean and match its target shape")
        known = value[mask]
        if not torch.isfinite(known).all():
            raise ValueError(f"Non-finite observed labels in targets.{name}")
        if name == "health_state" and ((known < 0) | (known > 4) | (known != known.round())).any():
            raise ValueError("health_state labels must be integer class indices in [0, 4]")
        if name == "health_score" and ((known < 0) | (known > 100)).any():
            raise ValueError("health_score labels must lie in [0, 100]")
        if name == "drivers" and not ((known == 0) | (known == 1)).all():
            raise ValueError("Driver labels must be binary")
        if name == "driver_strengths" and ((known < 0) | (known > 1)).any():
            raise ValueError("Driver strengths must lie in [0, 1]")
        masks[name] = mask
    batch = targets["health_state"].shape[0]
    observed = torch.stack([mask.reshape(batch, -1).any(1) for mask in masks.values()]).any(0)
    if not observed.all():
        raise ValueError("Each supervised example requires at least one observed target")
    if "label_valid" in targets and not torch.as_tensor(targets["label_valid"]).all():
        raise ValueError("Supervised learning requires observed labels, not unlabeled acquisitions")
    return masks


def masked_mean(values, mask):
    """Retain a zero gradient for a task with no observed labels."""
    return values[mask].mean() if mask.any() else values.reshape(-1)[:0].sum()
