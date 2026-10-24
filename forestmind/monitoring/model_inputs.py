"""Bind a registered model to measured, dated modalities with explicit coverage."""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np

from forestmind.data.sequence import PreparedSequence


def _required_modalities(contract):
    if not isinstance(contract, dict):
        return None, ["Model input contract must be an object."]
    required = contract.get("required_modalities", ["optical"])
    supported = ("optical", "sar", "climate", "activity")
    if (not isinstance(required, list) or not required or
            any(not isinstance(name, str) or name not in supported for name in required) or
            len(set(required)) != len(required)):
        return None, ["required_modalities must uniquely name supported optical, sar, climate or activity inputs."]
    if "optical" not in required:
        return None, ["This optical-anchored sequence requires optical model inputs; radar-only sampling needs its own acquisition timeline."]
    return required, []


def _bool_mask(arrays, key, shape):
    mask = arrays.get(key)
    return mask if isinstance(mask, np.ndarray) and mask.dtype.kind == "b" and mask.shape == shape else None


def _alignment_reasons(prepared, contract):
    alignment = prepared.provenance.get("alignment")
    if not isinstance(alignment, dict) or not isinstance(alignment.get("parameters"), dict):
        return ["Model requires an explicit as-of context alignment manifest."]
    reasons = []
    if contract.get("context_alignment") != alignment["parameters"]:
        reasons.append("Model must pin the same as-of context alignment parameters used for its inputs.")
    try:
        actual = hashlib.sha256(json.dumps({key: value for key, value in alignment.items() if key != "sha256"},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (ValueError, TypeError):
        actual = None
    declared = prepared.arrays.get("alignment_sha256")
    if (actual is None or alignment.get("sha256") != actual or not isinstance(declared, np.ndarray)
            or declared.shape != () or declared.dtype.kind not in "US" or str(declared.item()) != actual):
        reasons.append("Aligned input manifest or NPZ alignment hash changed.")
    return reasons


def modality_contract_reasons(contract, prepared):
    """Return concrete refusal reasons instead of inferring coverage from labels.

    Context coverage means the fraction of observed optical dates with every
    required feature present. Feature-level masks still survive model execution.
    This path is deliberately optical-anchored: dropping optical can otherwise
    move the effective forecast origin while retaining misleading timestamps.
    """
    required, reasons = _required_modalities(contract)
    if reasons:
        return reasons
    arrays = prepared.arrays
    observed = arrays.get("observation_mask")
    if (not isinstance(observed, np.ndarray) or observed.dtype.kind != "b" or observed.ndim != 1
            or not len(observed) or not observed.any()):
        return ["Input observation_mask must be a nonempty boolean vector with observed dates."]
    count = len(observed)
    optical = arrays.get("optical")
    if not isinstance(optical, np.ndarray) or optical.ndim != 4 or optical.shape[:2] != (count, 12):
        return ["Optical inputs must have shape (T, 12, H, W) matching observation_mask."]
    shape = optical.shape[-2:]
    optical_mask = _bool_mask(arrays, "optical_mask", (count,))
    sar_mask = _bool_mask(arrays, "sar_mask", (count,))
    quality = _bool_mask(arrays, "quality_mask", (count, *shape))
    if optical_mask is None or sar_mask is None or quality is None:
        return ["Satellite modality and quality masks must be explicit booleans on the input timeline/grid."]
    if (not np.array_equal(observed, optical_mask | sar_mask) or
            not np.array_equal(optical_mask, quality.reshape(count, -1).any(axis=1))):
        reasons.append("Satellite masks disagree with observed dates or valid optical pixels.")
    # Prepared optical sequences have optical evidence on every observed date.
    # Refuse an incidental SAR-only date instead of silently changing its origin.
    if np.any(observed & ~optical_mask):
        reasons.append("Optical-anchored model inputs require optical evidence on every observed date.")
    fractions = contract.get("min_modality_fraction", {})
    if not isinstance(fractions, dict) or any(name not in ("optical", "sar", "climate", "activity") for name in fractions):
        return reasons + ["min_modality_fraction must be an object keyed by supported modality names."]
    available = prepared.provenance.get("available_modalities", ["optical"])
    if not isinstance(available, list) or any(not isinstance(name, str) for name in available):
        return reasons + ["Available-modality provenance must be an explicit list."]
    for name in required:
        if name not in available:
            reasons.append(f"Required {name} model input is absent; contextual display alone is insufficient.")
            continue
        mask_key = {"optical": "optical_mask", "sar": "sar_mask", "climate": "climate_mask", "activity": "activity_mask"}[name]
        mask = arrays.get(mask_key)
        threshold = fractions.get(name, 1.)
        if type(threshold) not in (float, int) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            reasons.append(f"Minimum {name} coverage must lie in (0, 1].")
            continue
        if name in ("climate", "activity"):
            values = arrays.get(f"{name}_context")
            if (not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[0] != count
                    or values.shape[1] < 1 or _bool_mask(arrays, mask_key, values.shape) is None):
                reasons.append(f"Required {name} values and feature masks must share shape (T, F).")
                continue
            present = mask.all(axis=1)
        else:
            present = mask
        if name == "sar":
            radar = arrays.get("sar")
            radar_quality = _bool_mask(arrays, "sar_quality_mask", (count, *shape))
            if (not isinstance(radar, np.ndarray) or radar.shape != (count, 2, *shape)
                    or radar_quality is None or not np.array_equal(sar_mask, radar_quality.reshape(count, -1).any(1))):
                reasons.append("Required SAR pixels and separate radar quality mask must match the input grid and dates.")
                continue
        fraction = float(present[observed].mean())
        if fraction < threshold:
            reasons.append(f"{name} covers {fraction:.1%} of input dates, below the registered {threshold:.1%} minimum.")
    if any(name != "optical" for name in required):
        reasons.extend(_alignment_reasons(prepared, contract))
    if "sar" in required:
        units = arrays.get("sar_units")
        if (not isinstance(units, np.ndarray) or units.shape != () or units.dtype.kind not in "US"
                or contract.get("sar_units") != str(units.item())):
            reasons.append("Registered SAR units do not match aligned radar units.")
    if "climate" in required:
        for name in ("climate_feature_names", "climate_feature_units"):
            values = arrays.get(name)
            if (not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype.kind not in "US"
                    or contract.get(name) != values.tolist()):
                reasons.append(f"Registered {name} differs from measured climate features.")
        if contract.get("climate_vintage") != "retrospective_reanalysis":
            reasons.append("Model must explicitly acknowledge retrospective_reanalysis climate vintage; historical availability is unverified.")
    return reasons


def select_model_sequence(prepared, contract):
    """Exclude modalities a checkpoint was not registered to consume."""
    reasons = modality_contract_reasons(contract, prepared)
    if reasons:
        raise ValueError("Cannot select model inputs: " + "; ".join(reasons))
    required = list(contract.get("required_modalities", ["optical"]))
    arrays = dict(prepared.arrays)
    if "sar" not in required:
        arrays["sar"] = np.zeros_like(arrays["sar"])
        arrays["sar_mask"] = np.zeros_like(arrays["sar_mask"])
        if "sar_quality_mask" in arrays:
            arrays.pop("sar_quality_mask")
        for key in ("sar_acquisition_dates", "sar_age_days", "sar_units"):
            arrays.pop(key, None)
    for name in ("climate", "activity"):
        if name not in required:
            for key in (f"{name}_context", f"{name}_mask", f"{name}_coverage",
                        f"{name}_feature_names", f"{name}_feature_units"):
                arrays.pop(key, None)
    arrays["observation_mask"] = arrays["optical_mask"] | arrays["sar_mask"]
    provenance = {**prepared.provenance,
                  "source_available_modalities": list(prepared.provenance.get("available_modalities", ["optical"])),
                  "available_modalities": required, "model_input_modalities": required,
                  "sar_present": "sar" in required, "climate_present": "climate" in required}
    if "sar" not in required:
        provenance.pop("sar_units", None)
    if "climate" not in required:
        provenance.pop("climate_feature_names", None)
        provenance.pop("climate_feature_units", None)
    return PreparedSequence(arrays, provenance)
