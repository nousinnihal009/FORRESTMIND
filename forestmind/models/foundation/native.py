"""Restricted, reproducible native encoder construction and complete state loading.

The runtime lock identifies software, not training quality. Native inference never
asks TerraTorch to fetch pretrained weights: the full ForestMind state must be
present in the locally supplied checkpoint, including every downstream head.
"""

from __future__ import annotations

from copy import deepcopy
from importlib import metadata

import numpy as np
import torch


AUDITED_TERRATORCH = "1.2.13"
RUNTIME_PACKAGES = ("terratorch", "torch", "torchvision", "timm")
NATIVE_MODELS = {
    "prithvi": frozenset({"prithvi_eo_v1_100", "prithvi_eo_v2_300", "prithvi_eo_v2_600"}),
    "terramind": frozenset({"terramind_v1_tiny", "terramind_v1_small", "terramind_v1_base", "terramind_v1_large"}),
}
DEFAULT_MODELS = {"prithvi": "prithvi_eo_v2_300", "terramind": "terramind_v1_base"}


def capture_native_runtime() -> dict[str, str]:
    """Return exact installed versions; absent dependencies are an explicit error."""
    versions = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise ValueError(f"Native offline inference requires installed {package}; install a compatible foundation runtime") from exc
    return versions


def validate_native_architecture(name: str, model_name: str, *, pretrained: bool, use_proxy: bool) -> None:
    """Keep native registry dispatch away from generators, tokenizers and custom code."""
    if type(pretrained) is not bool or type(use_proxy) is not bool:
        raise ValueError("pretrained and use_proxy must be explicit booleans")
    if not use_proxy and (not isinstance(model_name, str) or model_name not in NATIVE_MODELS.get(name, ())):
        raise ValueError(f"Unsupported native {name} architecture: {model_name}; use an audited standard encoder")


def offline_native_config(config: dict, *, check_runtime: bool = True) -> dict:
    """Validate the checkpoint contract and force architecture-only construction.

Returns a copy, so the stored training configuration and its provenance remain
unchanged. The fixed registry arguments have no remote or local weight path.
"""
    foundation = config.get("foundation", {})
    name = foundation.get("name")
    if name not in NATIVE_MODELS:
        raise ValueError("Native offline inference requires Prithvi or TerraMind")
    options = foundation.get(name, {})
    if not isinstance(options, dict) or options.get("use_proxy", False) is not False:
        raise ValueError("Native offline inference requires use_proxy=false")
    allowed = {"model_name", "freeze", "use_proxy", "pretrained"}
    if name == "terramind":
        allowed.add("sar_modality")
    if set(options) - allowed:
        raise ValueError("Native offline inference does not accept custom constructor options or checkpoint paths")
    if "pretrained" in options and type(options["pretrained"]) is not bool:
        raise ValueError("Native pretrained must be an explicit boolean")
    if type(options.get("freeze", True)) is not bool:
        raise ValueError("Native freeze must be an explicit boolean")
    model_name = options.get("model_name", DEFAULT_MODELS[name])
    validate_native_architecture(name, model_name, pretrained=False, use_proxy=False)
    runtime = foundation.get("native_runtime")
    if (not isinstance(runtime, dict) or set(runtime) != set(RUNTIME_PACKAGES)
            or any(not isinstance(v, str) or not v.strip() for v in runtime.values())):
        raise ValueError("Native offline inference requires exact foundation.native_runtime versions for " + ", ".join(RUNTIME_PACKAGES))
    if runtime["terratorch"] != AUDITED_TERRATORCH:
        raise ValueError(f"Native offline inference supports audited terratorch {AUDITED_TERRATORCH} only")
    if check_runtime:
        installed = capture_native_runtime()
        mismatch = [package for package in RUNTIME_PACKAGES if installed[package] != runtime[package]]
        if mismatch:
            raise ValueError("Native offline inference runtime differs from the checkpoint: " + ", ".join(mismatch))
    data = config.get("data", {})
    if data.get("optical_units") != "surface_reflectance_0_1":
        raise ValueError("Native offline inference requires data.optical_units=surface_reflectance_0_1")
    normalization = data.get("normalization")
    for key, count in (("optical_mean", 12), ("optical_std", 12), ("sar_mean", 2), ("sar_std", 2)):
        if not isinstance(normalization, dict) or key not in normalization:
            raise ValueError("Native offline inference requires explicit per-band data.normalization statistics")
        values = np.asarray(normalization[key])
        if (values.dtype.kind not in "fiu" or values.shape != (count,)
                or not np.isfinite(values).all() or (key.endswith("std") and (values <= 0).any())):
            raise ValueError(f"Native normalization {key} must contain {count} finite, compatible statistics")
    if name == "terramind":
        modality = options.get("sar_modality", "S1GRD")
        expected_units = {"S1GRD": "sigma0_db", "S1RTC": "gamma0_db"}.get(modality) if isinstance(modality, str) else None
        if expected_units is None or data.get("sar_units") != expected_units:
            raise ValueError("TerraMind requires matching sar_modality and data.sar_units: S1RTC/gamma0_db or S1GRD/sigma0_db")
    result = deepcopy(config)
    result["foundation"].setdefault(name, {})["pretrained"] = False
    return result


def validate_native_sample(config: dict, sample: dict) -> None:
    """Validate patch geometry and unsupported modality use before model allocation."""
    foundation = config["foundation"]
    name = foundation["name"]
    model_name = foundation.get(name, {}).get("model_name", DEFAULT_MODELS[name])
    patch = 14 if model_name == "prithvi_eo_v2_600" else 16
    height, width = sample["optical"].shape[-2:]
    if height % patch or width % patch:
        raise ValueError(f"Native {model_name} needs height and width divisible by its {patch}-pixel patch size")
    if name == "prithvi":
        if np.any(sample["observation_mask"] & ~sample["optical_mask"]):
            raise ValueError("Prithvi cannot encode SAR-only timesteps; every observation needs optical imagery")
        if np.any(sample["sar_mask"]):
            raise ValueError("Prithvi is optical-only; supplied SAR cannot be silently discarded")


def validate_checkpoint_tensors(state: dict) -> None:
    """Reject incomplete tensor containers and nonfinite values before allocation."""
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model_state_dict must contain a complete nonempty tensor state")
    for key, value in state.items():
        if (not isinstance(key, str) or not isinstance(value, torch.Tensor)
                or value.layout != torch.strided or value.is_meta or value.is_quantized):
            raise ValueError("Checkpoint state must contain named dense, non-meta, unquantized tensors only")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint tensor {key} contains nonfinite values")


def load_complete_state(model: torch.nn.Module, state: dict) -> None:
    """No filtered keys, dtype coercion, or randomly retained model parameters."""
    validate_checkpoint_tensors(state)
    expected = model.state_dict()
    if set(expected) != set(state):
        missing, extra = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
        raise ValueError(f"Incomplete checkpoint state: {len(missing)} missing and {len(extra)} unexpected tensors")
    for name, value in expected.items():
        if value.shape != state[name].shape or value.dtype != state[name].dtype:
            raise ValueError(f"Checkpoint tensor {name} has an incompatible shape or dtype")
    model.load_state_dict(state, strict=True)
