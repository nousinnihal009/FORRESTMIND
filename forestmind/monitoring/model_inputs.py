"""Bind a registered model to measured, dated modalities with explicit coverage."""
from __future__ import annotations

import math
import numpy as np

from forestmind.data.sequence import PreparedSequence


def modality_contract_reasons(contract, prepared):
    required = contract.get("required_modalities", ["optical"])
    if (not isinstance(required, list) or not required or len(set(required)) != len(required)
            or any(name not in ("optical", "sar", "climate", "activity") for name in required)):
        return ["required_modalities must uniquely name supported optical, sar, climate or activity inputs."]
    reasons = []
    available = prepared.provenance.get("available_modalities", ["optical"])
    for name in required:
        if name not in available:
            reasons.append(f"Required {name} model input is absent; contextual display alone is insufficient.")
            continue
        mask_key = {"optical": "optical_mask", "sar": "sar_mask", "climate": "climate_mask", "activity": "activity_mask"}[name]
        mask = prepared.arrays.get(mask_key)
        threshold = contract.get("min_modality_fraction", {}).get(name, 1.)
        if type(threshold) not in (float, int) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            reasons.append(f"Minimum {name} coverage must lie in (0, 1].")
            continue
        present = mask if mask.ndim == 1 else mask.all(axis=1)
        observed = prepared.arrays["observation_mask"]
        fraction = float(present[observed].mean())
        if fraction < threshold:
            reasons.append(f"{name} covers {fraction:.1%} of input dates, below the registered {threshold:.1%} minimum.")
    if any(name != "optical" for name in required):
        alignment = prepared.provenance.get("alignment", {})
        if contract.get("context_alignment") != alignment.get("parameters") or not alignment:
            reasons.append("Model must pin the same as-of context alignment parameters used for its inputs.")
    if "sar" in required and contract.get("sar_units") != str(prepared.arrays.get("sar_units", "")):
        reasons.append("Registered SAR units do not match aligned radar units.")
    if "climate" in required:
        for name in ("climate_feature_names", "climate_feature_units"):
            if contract.get(name) != prepared.arrays.get(name, np.asarray([])).tolist():
                reasons.append(f"Registered {name} differs from measured climate features.")
        if contract.get("climate_vintage") != "retrospective_reanalysis":
            reasons.append("Model must explicitly acknowledge retrospective_reanalysis climate vintage; historical availability is unverified.")
    return reasons


def select_model_sequence(prepared, contract):
    """Exclude modalities a checkpoint was not registered to consume."""
    required = contract.get("required_modalities", ["optical"])
    arrays = dict(prepared.arrays)
    if "sar" not in required:
        arrays["sar"] = np.zeros_like(arrays["sar"])
        arrays["sar_mask"] = np.zeros_like(arrays["sar_mask"])
        if "sar_quality_mask" in arrays:
            arrays["sar_quality_mask"] = np.zeros_like(arrays["sar_quality_mask"])
    for name in ("climate", "activity"):
        if name not in required:
            for key in (f"{name}_context", f"{name}_mask", f"{name}_coverage"):
                arrays.pop(key, None)
    if "optical" not in required:
        arrays["optical"] = np.zeros_like(arrays["optical"])
        arrays["optical_mask"] = np.zeros_like(arrays["optical_mask"])
        arrays["quality_mask"] = np.zeros_like(arrays["quality_mask"])
    arrays["observation_mask"] = arrays["optical_mask"] | arrays["sar_mask"]
    return PreparedSequence(arrays, {**prepared.provenance, "model_input_modalities": required})
