"""Run a local checkpoint on explicit, validated satellite sequences.

No imagery, context, masks, labels or checkpoint downloads are fabricated here.
All input imagery must be raw (unnormalized), in the units used to train the
checkpoint. NPZ timestamps are day offsets and the final observed timestamp is
anchored to the explicitly declared observed-through date.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import ntpath
import zipfile

import numpy as np
import torch

from forestmind.data.transforms import get_eval_transforms
from forestmind.inference.schemas import DRIVER_NAMES, HEALTH_NAMES, InferenceReport, calendar_date
from forestmind.models.forestmind_model import ForestMindModel


INPUT_KEYS = ("optical", "sar", "timestamps", "observation_mask", "optical_mask",
              "sar_mask", "quality_mask", "climate_context", "activity_context")
REQUIRED_KEYS = INPUT_KEYS[:7]
NORMALIZATION_KEYS = {"optical_mean", "optical_std", "sar_mean", "sar_std"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool_mask(sample: dict, name: str, shape: tuple):
    value = sample[name]
    if value.shape != shape or value.dtype.kind != "b":
        raise ValueError(f"{name} must be an explicit boolean array with shape {shape}")
    return value


def _scalar(data: dict, name: str):
    value = data[name]
    if value.shape != ():
        raise ValueError(f"{name} metadata must be a scalar")
    return value.item()


def load_sequence(path: str | Path, config: dict, *, observed_through: str,
                  synthetic_input: bool = False) -> tuple[dict, dict]:
    """Read numeric/string NPZ without pickle and validate masks, dates and units."""
    calendar_date(observed_through)
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 128 or sum(item.file_size for item in entries) > 512 * 1024 * 1024:
            raise ValueError("Input NPZ exceeds the local 512 MiB / 128-array limit")
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("NPZ must not contain duplicate array names")
    with np.load(path, allow_pickle=False) as archive:
        # Read every field: hidden object arrays must also fail safe loading.
        data = {key: archive[key].copy() for key in archive.files}
    missing = set(REQUIRED_KEYS) - set(data)
    if missing:
        raise ValueError("Input NPZ requires explicit arrays: " + ", ".join(sorted(missing)))
    if "synthetic" in data:
        declared = _scalar(data, "synthetic")
        if type(declared) is not bool or declared != synthetic_input:
            raise ValueError("Input synthetic metadata must agree with --synthetic-input")
    if "observed_through" in data and _scalar(data, "observed_through") != observed_through:
        raise ValueError("Input observed_through metadata must match --observed-through")
    if "normalization" in data and _scalar(data, "normalization") != "raw":
        raise ValueError("Input must be raw; normalization is applied from the checkpoint exactly once")
    sample = {key: data[key] for key in INPUT_KEYS if key in data}
    optical, sar = sample["optical"], sample["sar"]
    if optical.ndim != 4 or optical.shape[1] != 12:
        raise ValueError("optical must have shape (T, 12, H, W)")
    steps, _, height, width = optical.shape
    max_steps = config.get("temporal", {}).get("max_seq_len", 48)
    if not 1 <= steps <= min(max_steps, 4096) or not min(height, width) >= 1:
        raise ValueError("Input dimensions exceed configured temporal capacity or are empty")
    if sar.shape != (steps, 2, height, width):
        raise ValueError("sar must share the optical grid with shape (T, 2, H, W)")
    observed = _bool_mask(sample, "observation_mask", (steps,))
    optical_mask = _bool_mask(sample, "optical_mask", (steps,))
    sar_mask = _bool_mask(sample, "sar_mask", (steps,))
    quality = _bool_mask(sample, "quality_mask", (steps, height, width))
    if not np.array_equal(observed, optical_mask | sar_mask) or not observed.any():
        raise ValueError("observation_mask must equal the nonempty union of modality masks")
    if not np.array_equal(optical_mask, quality.reshape(steps, -1).any(axis=1)):
        raise ValueError("quality_mask must contain valid pixels exactly at optical observations")
    for name in ("optical", "sar", "timestamps", "climate_context", "activity_context"):
        if name in sample:
            value = sample[name]
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite numeric values; store masked pixels as zeros")
    if np.any(optical[~optical_mask]) or np.any(sar[~sar_mask]) or np.any(optical[~quality[:, None].repeat(12, axis=1)]):
        raise ValueError("Missing modality and invalid optical pixels must be explicitly zero filled")
    timestamps = sample["timestamps"]
    if timestamps.shape != (steps,) or (timestamps < 0).any() or (np.diff(timestamps) < 0).any():
        raise ValueError("timestamps must be nonnegative, ordered day offsets of shape (T,)")
    actual = timestamps[observed].astype(np.float64)
    if actual[0] != 0 or (np.diff(actual) <= 0).any() or timestamps[-1] != actual[-1]:
        raise ValueError("Observed timestamps must begin at zero and strictly increase; padding cannot advance the observation date")
    if actual[-1] > 365250:
        raise ValueError("Input sequence cannot span more than 1000 years")
    basis = "day_offsets_anchored_to_declared_date"
    start = (date.fromisoformat(observed_through) - timedelta(days=float(actual[-1]))).isoformat()
    if "acquisition_dates" in data:
        dates = data["acquisition_dates"]
        if dates.shape != (steps,) or dates.dtype.kind not in "US":
            raise ValueError("acquisition_dates must contain one ISO date or timestamp per timestep")
        parsed = []
        for value in dates.astype(str):
            try:
                item = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("Invalid acquisition_dates ISO value") from exc
            if "T" in value and item.utcoffset() is None:
                raise ValueError("Acquisition timestamps require an explicit timezone")
            parsed.append(item.replace(tzinfo=timezone.utc) if item.utcoffset() is None else item.astimezone(timezone.utc))
        effective = [item for item, mask in zip(parsed, observed) if mask]
        offsets = [(item - effective[0]).total_seconds() / 86400 for item in parsed]
        if not np.allclose(timestamps, offsets, atol=1e-4, rtol=0):
            raise ValueError("timestamps must match the actual acquisition_dates day offsets")
        if effective[-1].date().isoformat() != observed_through:
            raise ValueError("observed_through must match the latest actual acquisition date")
        start, basis = effective[0].date().isoformat(), "acquisition_dates"
    for name in ("climate_context", "activity_context"):
        if name in sample and sample[name].shape != (steps, config.get("foundation", {}).get("embedding_dim", 256)):
            raise ValueError(f"{name} must contain one checkpoint-sized context vector per timestep")
    info = {"name": path.name, "sha256": sha256_file(path), "synthetic": synthetic_input,
            "timesteps": steps, "observed_timesteps": int(observed.sum()),
            "optical_timesteps": int(optical_mask.sum()), "sar_timesteps": int(sar_mask.sum()),
            "valid_optical_fraction": float(quality.mean()), "observed_start": start,
            "timestamp_basis": basis}
    if "source_manifest_sha256" in data:
        info["source_manifest_sha256"] = _scalar(data, "source_manifest_sha256")
    return sample, info


def predict_checkpoint(checkpoint_path: str | Path, input_path: str | Path, *,
                       region_id: str, observed_through: str, synthetic_input: bool = False,
                       device: str = "cpu") -> dict:
    """Infer once using safe local checkpoint loading and strict model/report contracts."""
    calendar_date(observed_through)
    if type(synthetic_input) is not bool:
        raise ValueError("synthetic_input must be an explicit boolean")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda or mps")
    checkpoint_path, input_path = Path(checkpoint_path), Path(input_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict and its training config")
    config = checkpoint["config"]
    foundation = config.get("foundation", {})
    name = foundation.get("name", "custom_vit")
    proxy = foundation.get(name, {}).get("use_proxy", False)
    if name not in {"custom_vit", "prithvi", "terramind"}:
        raise ValueError("Unsupported checkpoint backbone")
    if type(proxy) is not bool:
        raise ValueError("Checkpoint foundation use_proxy must be an explicit boolean")
    if name != "custom_vit" and not proxy:
        raise ValueError("Native pretrained encoders are not supported by offline inference yet; their current builders may download weights. A verified native runtime, complete local trained checkpoint and strict offline loader are required. See docs/NATIVE_MODEL_AUDIT.md. Explicit proxy checkpoints remain development baselines only.")
    heads = config.get("heads", {})
    if (heads.get("health", {}).get("num_classes", 5) != 5 or
            not heads.get("health", {}).get("continuous_score", True) or
            heads.get("attribution", {}).get("num_drivers", 8) != 8):
        raise ValueError("Inference requires the five canonical health classes, continuous health score and eight canonical drivers")
    horizons = heads.get("trajectory", {}).get("horizons", [30, 180, 365, 1095, 1825])
    if (not isinstance(horizons, list) or not 1 <= len(horizons) <= 64 or
            any(type(day) is not int or day <= 0 for day in horizons) or horizons != sorted(set(horizons))):
        raise ValueError("Checkpoint horizons must be positive, strictly increasing integer days")
    data_config = config.get("data", {})
    normalization = data_config.get("normalization")
    complete_stats = isinstance(normalization, dict) and NORMALIZATION_KEYS.issubset(normalization)
    if not synthetic_input and not complete_stats:
        raise ValueError("Real inputs require explicit checkpoint data.normalization optical_mean/std and sar_mean/std matching input units")
    sample, info = load_sequence(input_path, config, observed_through=observed_through, synthetic_input=synthetic_input)
    info["normalization"] = "checkpoint_statistics" if complete_stats else "synthetic_defaults"
    transformed = get_eval_transforms(normalization=normalization)(sample)
    inputs = {key: transformed[key].unsqueeze(0).to(device) for key in INPUT_KEYS if key in transformed}
    model = ForestMindModel(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    with torch.inference_mode():
        outputs = model(**inputs)
    health, drivers, trajectory = (outputs[name] for name in ("health", "attribution", "trajectory"))
    if health["probs"].shape != (1, 5) or drivers["probs"].shape != (1, 8) or trajectory["score_100"].shape != (1, len(horizons)):
        raise ValueError("Checkpoint output dimensions do not match the report contract")
    source = data_config.get("source", "unknown")
    if not isinstance(source, str) or not source:
        source = "unknown"
    trained_on_synthetic = True if source == "synthetic" else data_config.get("synthetic")
    if trained_on_synthetic is not None and type(trained_on_synthetic) is not bool:
        raise ValueError("Checkpoint data.synthetic must be boolean when supplied")
    caveats = [
        "Model outputs and standard deviations are uncalibrated; they are not validated confidence intervals, causal attributions, tipping probabilities or tipping dates.",
        "Checkpoint metadata, region association, acquisition timing and input provenance are user supplied and not independently verified.",
        "A hash identifies file content; it does not establish model quality or ecological validity.",
    ]
    if synthetic_input:
        caveats.append("SYNTHETIC INPUT: this run is a software demonstration, not a real ecological assessment.")
    if trained_on_synthetic:
        caveats.append("SYNTHETIC TRAINING: this checkpoint has not established predictive performance on real forests.")
    elif trained_on_synthetic is None:
        caveats.append("The checkpoint does not declare whether its training observations were synthetic.")
    if info["timestamp_basis"] != "acquisition_dates":
        caveats.append("Input day offsets are anchored to the user-declared observed-through date; absolute acquisition dates were not supplied.")
    if not info["sar_timesteps"]:
        caveats.append("No SAR observations were supplied; the SAR stream is explicitly masked as unavailable.")
    report = {
        "schema_version": 1, "kind": "forestmind_inference", "region_id": region_id,
        "observed_through": observed_through, "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"name": checkpoint_path.name, "sha256": checkpoint_hash,
                       "epoch": checkpoint.get("epoch"), "backbone": f"{name}_proxy" if proxy and name != "custom_vit" else name,
                       "training_source": ntpath.basename(source.rstrip("/\\")) or "unknown",
                       "trained_on_synthetic": trained_on_synthetic},
        "input": info,
        "predictions": {
            "health": {"state": HEALTH_NAMES[int(health["state"][0])], "score": float(health["score"][0]),
                       "class_scores": [{"name": label, "score": float(health["probs"][0, index])} for index, label in enumerate(HEALTH_NAMES)]},
            "drivers": [{"name": label, "score": float(drivers["probs"][0, index]), "strength": float(drivers["strength"][0, index])}
                        for index, label in enumerate(DRIVER_NAMES)],
            "trajectory": [{"horizon_days": horizon, "score": float(trajectory["score_100"][0, index]),
                            "stddev": float(trajectory["uncertainty"][0, index]) * 100 if "uncertainty" in trajectory else None}
                           for index, horizon in enumerate(horizons)],
        }, "calibration": "uncalibrated", "caveats": caveats,
    }
    if sha256_file(checkpoint_path) != checkpoint_hash or sha256_file(input_path) != info["sha256"]:
        raise ValueError("A source file changed during inference; rerun with stable input files")
    return InferenceReport.model_validate(report).model_dump(mode="json", exclude_none=False)


def report_destination(output: str | Path, *, protected_paths=()) -> Path:
    """Resolve output while protecting source files, including existing hardlinks."""
    output = Path(output).resolve()
    for path in protected_paths:
        protected = Path(path).resolve()
        if output == protected or (output.exists() and protected.exists() and output.samefile(protected)):
            raise ValueError("Inference report output must not overwrite the checkpoint or input sequence")
    return output


def write_inference_report(report: dict, output: str | Path, *, protected_paths=()) -> Path:
    """Save strict JSON without overwriting checkpoint or sequence input paths."""
    output = report_destination(output, protected_paths=protected_paths)
    validated = InferenceReport.model_validate(report).model_dump(mode="json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validated, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output
