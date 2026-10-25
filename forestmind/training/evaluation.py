"""Reproducible checkpoint evaluation with auditable inputs and raw predictions.

Evaluation does not fit calibration, tune thresholds or establish ecological
validity. Validation and test examples keep explicit split identities.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import torch
import yaml

from forestmind import __version__
from forestmind.data.dataset import create_dataloaders
from forestmind.data.synthetic import DRIVER_CLASSES, HEALTH_STATES
from forestmind.models.forestmind_model import ForestMindModel
from forestmind.training.losses import ForestMindLoss
from forestmind.training.metrics import ForestMindMetrics
from forestmind.training.trainer import _get_device
from forestmind.training.targets import MASK_KEYS, supervision_masks


DEFAULT_HORIZONS = [30, 180, 365, 1095, 1825]
INPUT_KEYS = ("optical", "sar", "timestamps", "observation_mask", "optical_mask",
              "sar_mask", "quality_mask", "sar_quality_mask", "climate_context", "activity_context",
              "climate_mask", "activity_mask")
TARGET_KEYS = ("health_state", "health_score", "drivers", "driver_strengths", "trajectory")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _report_config(value):
    """Keep useful settings without exporting common credentials or signed URLs."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            sensitive = normalized in {"token", "auth", "cookie", "cookies", "key"} or any(
                marker in normalized for marker in (
                    "secret", "password", "credential", "api_key", "apikey", "authorization",
                    "access_key", "private_key", "access_token", "refresh_token", "auth_token",
                    "session_token", "hf_token", "hub_token",
                )
            )
            result[key] = "[REDACTED]" if sensitive else _report_config(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_report_config(item) for item in value]
    if isinstance(value, str) and "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https", "s3", "gs", "az"}:
            # Every query field is dropped, including arbitrary signed-URL keys.
            return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
    return value


def tensor_mapping_hash(values: dict[str, torch.Tensor]) -> str:
    """Hash exact tensor content and its names, shapes and dtypes, independent of batching."""
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        header = json.dumps({"name": name, "dtype": str(array.dtype), "shape": list(array.shape)},
                            sort_keys=True, separators=(",", ":")).encode("utf-8")
        content = array.tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _finite_tensors(values: dict, prefix: str):
    for name, value in values.items():
        if isinstance(value, dict):
            _finite_tensors(value, f"{prefix}.{name}")
        elif isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite values in {prefix}.{name}; evaluation was stopped")


def _validate_targets(targets: dict, *, samples: int, classes: int, drivers: int, horizons: int):
    shapes = {"health_state": (samples,), "health_score": (samples,),
              "drivers": (samples, drivers), "driver_strengths": (samples, drivers),
              "trajectory": (samples, horizons)}
    for name, shape in shapes.items():
        if tuple(targets[name].shape) != shape:
            raise ValueError(f"Target {name} has shape {tuple(targets[name].shape)}, expected {shape}; check dataset and configured horizons")
    masks = supervision_masks(targets)
    states = targets["health_state"][masks["health_state"]]
    if (states < 0).any() or (states >= classes).any() or (states != states.round()).any():
        raise ValueError("health_state labels must be valid integer class indices")
    if ((targets["health_score"][masks["health_score"]] < 0) | (targets["health_score"][masks["health_score"]] > 100)).any():
        raise ValueError("health_score labels must lie in [0, 100]")
    if not ((targets["drivers"][masks["drivers"]] == 0) | (targets["drivers"][masks["drivers"]] == 1)).all():
        raise ValueError("Driver labels must be binary")
    for name in ("driver_strengths", "trajectory"):
        if ((targets[name][masks[name]] < 0) | (targets[name][masks[name]] > 1)).any():
            raise ValueError(f"Normalized {name} labels must lie in [0, 1]; percent trajectory labels require data.trajectory_target_scale='percent'")


def _source_manifest(dataset) -> list[dict]:
    if dataset.source == "synthetic":
        return []
    return [{"index": index, "path": str(path.resolve()), "sha256": sha256_file(path),
             "bytes": path.stat().st_size}
            for index, path in enumerate(dataset.npz_files)]


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    config_path: str | Path | None = None,
) -> dict:
    """Evaluate one explicit split and return a strict JSON-compatible report.

    Trajectory targets default to the documented unit scale [0, 1]. To evaluate
    percent-scale disk labels, declare data.trajectory_target_scale='percent';
    target scaling is never inferred independently for different batches.
    """
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be 'val' or 'test'")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain a model_state_dict")
    checkpoint_config = checkpoint.get("config")
    config = checkpoint_config
    if config_path is not None:
        with Path(config_path).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no usable configuration; provide --config")
    config_hash = _json_hash(config)
    data_config = config.get("data", {})
    trajectory_scale = data_config.get("trajectory_target_scale", "unit")
    if trajectory_scale not in {"unit", "percent"}:
        raise ValueError("data.trajectory_target_scale must be 'unit' or 'percent'")
    if data_config.get("source", "synthetic") == "synthetic" and trajectory_scale != "unit":
        raise ValueError("Generated synthetic trajectories always use data.trajectory_target_scale='unit'")
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _get_device(config)
    heads = config.get("heads", {})
    horizons = heads.get("trajectory", {}).get("horizons", DEFAULT_HORIZONS)
    if (not horizons or any(type(day) is not int or day <= 0 for day in horizons)
            or list(horizons) != sorted(set(horizons))):
        raise ValueError("Configured trajectory horizons must be positive, strictly increasing integers")
    if isinstance(checkpoint_config, dict):
        trained_horizons = checkpoint_config.get("heads", {}).get("trajectory", {}).get("horizons", DEFAULT_HORIZONS)
        if list(horizons) != list(trained_horizons):
            raise ValueError("Evaluation horizons must match the checkpoint's trained trajectory horizons; an override cannot relabel predictions")
    classes = heads.get("health", {}).get("num_classes", 5)
    driver_count = heads.get("attribution", {}).get("num_drivers", 8)
    if classes != 5:
        raise ValueError("The current evaluation health metrics require the five canonical health states")
    if not heads.get("health", {}).get("continuous_score", True):
        raise ValueError("The current multitask evaluation loss requires continuous health scores")
    loaders = create_dataloaders(config, batch_size=data_config.get("batch_size", 4),
                                 num_workers=data_config.get("num_workers", 0))
    loader = loaders[2] if split == "test" else loaders[1]
    if len(loader.dataset) == 0:
        raise ValueError(f"Requested {split} dataset has no samples")
    source_manifest = _source_manifest(loader.dataset)
    from forestmind.models.foundation.native import offline_native_config, load_complete_state, validate_native_sample
    foundation = config.get("foundation", {})
    native = foundation.get("name") in ("prithvi", "terramind") and not foundation.get(foundation.get("name"), {}).get("use_proxy", False)
    model = ForestMindModel(offline_native_config(config) if native else config)
    load_complete_state(model, checkpoint["model_state_dict"])
    model = model.to(device).eval()
    criterion = ForestMindLoss(config.get("training", {}).get("loss_weights", {}))
    metrics = ForestMindMetrics(num_drivers=driver_count, num_horizons=len(horizons))
    weighted_losses: dict[str, float] = {}
    loss_counts: dict[str, int] = {}
    examples: list[dict] = []
    batches = 0
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            if native:
                for sample_index in range(batch["optical"].shape[0]):
                    validate_native_sample(config, {name: batch[name][sample_index]
                                                   for name in ("optical", "observation_mask", "optical_mask", "sar_mask")
                                                   if name in batch})
            if "label_valid" in batch and not torch.as_tensor(batch["label_valid"]).all():
                raise ValueError("Evaluation requires observed labels; unlabeled acquisitions cannot be evaluated")
            missing = [name for name in TARGET_KEYS if name not in batch]
            if missing:
                raise ValueError(f"Evaluation targets missing: {', '.join(missing)}")
            inputs = {name: batch[name].to(device) for name in INPUT_KEYS if name in batch}
            _finite_tensors(inputs, "inputs")
            original_targets = {name: batch[name].to(device) for name in TARGET_KEYS}
            original_targets.update({name: batch[name].to(device) for name in MASK_KEYS if name in batch})
            targets = dict(original_targets)
            if trajectory_scale == "percent":
                targets["trajectory"] = targets["trajectory"] / 100
            batch_count = int(batch["optical"].shape[0])
            _validate_targets(targets, samples=batch_count, classes=classes,
                              drivers=driver_count, horizons=len(horizons))
            outputs = model(**inputs)
            _finite_tensors(outputs, "outputs")
            if outputs["trajectory"]["mean"].shape != targets["trajectory"].shape:
                raise ValueError("Predicted and target trajectory shapes do not match")
            losses = criterion(outputs, targets)
            _finite_tensors(losses, "losses")
            masks = supervision_masks(targets)
            count_keys = {"health_ce": "health_state", "health_mse": "health_score",
                          "attribution_bce": "drivers", "attribution_strength_mse": "driver_strengths",
                          "trajectory_mse": "trajectory", "trajectory_huber": "trajectory", "trajectory_hetero": "trajectory"}
            for name, target in count_keys.items():
                if name in losses:
                    count = int(masks[target].sum())
                    weighted_losses[name] = weighted_losses.get(name, 0.0) + float(losses[name].item()) * count
                    loss_counts[name] = loss_counts.get(name, 0) + count
            metrics.update(outputs, targets)
            for index in range(batch_count):
                sample_index = sample_count + index
                sample_inputs = {name: value[index] for name, value in inputs.items()}
                sample_targets = {name: value[index] for name, value in original_targets.items()}
                source_ref = (source_manifest[sample_index] if source_manifest else {
                    "generator": "SyntheticForestGenerator",
                    "seed_sequence": [loader.dataset.seed, sample_index],
                })
                predictions = {head: {name: tensor[index].detach().cpu().tolist()
                                      for name, tensor in outputs[head].items()}
                               for head in ("health", "attribution", "trajectory")}
                examples.append({
                    "sample_id": f"{split}:{sample_index}", "index": sample_index, "split": split,
                    "input_sha256": tensor_mapping_hash(sample_inputs),
                    "target_sha256": tensor_mapping_hash(sample_targets),
                    "source_ref": source_ref,
                    "timestamps": inputs["timestamps"][index].detach().cpu().tolist(),
                    "targets": {name: value[index].detach().cpu().tolist() for name, value in targets.items()},
                    "trajectory_target_original": original_targets["trajectory"][index].detach().cpu().tolist(),
                    "predictions": predictions,
                })
            sample_count += batch_count
            batches += 1
    if sample_count != len(loader.dataset):
        raise ValueError("Evaluation did not cover every sample in the requested split")
    result = asdict(metrics.compute())
    errors = np.array([example["predictions"]["trajectory"]["mean"] for example in examples]) - np.array(
        [example["targets"]["trajectory"] for example in examples])
    trajectory_mask = np.array([example["targets"].get("trajectory_mask", [True] * len(horizons)) for example in examples], dtype=bool)
    averaged_losses = {name: value / loss_counts[name] if loss_counts[name] else 0.0 for name, value in weighted_losses.items()}
    averaged_losses["health_total"] = criterion.health_loss.ce_weight * averaged_losses.get("health_ce", 0) + criterion.health_loss.mse_weight * averaged_losses.get("health_mse", 0)
    averaged_losses["attribution_total"] = criterion.attribution_loss.bce_weight * averaged_losses.get("attribution_bce", 0) + criterion.attribution_loss.strength_weight * averaged_losses.get("attribution_strength_mse", 0)
    averaged_losses["trajectory_total"] = criterion.trajectory_loss.mse_weight * averaged_losses.get("trajectory_hetero", averaged_losses.get("trajectory_mse", 0)) + criterion.trajectory_loss.huber_weight * averaged_losses.get("trajectory_huber", 0)
    averaged_losses["total"] = criterion.w_health * averaged_losses["health_total"] + criterion.w_attribution * averaged_losses["attribution_total"] + criterion.w_trajectory * averaged_losses["trajectory_total"]
    synthetic = True if loader.dataset.source == "synthetic" else data_config.get("synthetic")
    if synthetic is not None and not isinstance(synthetic, bool):
        raise ValueError("data.synthetic must be a boolean when declared for a disk dataset")
    repo_root = Path(__file__).resolve().parents[2]
    implementation_paths = sorted(path.relative_to(repo_root).as_posix()
                                  for path in (repo_root / "forestmind").rglob("*.py"))
    report = {
        "schema_version": 1, "kind": "forestmind_checkpoint_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash,
                       "epoch": checkpoint.get("epoch"),
                       "training_status": checkpoint.get("training_status", "not independently verified")},
        "config": {"effective": _report_config(config), "sha256": config_hash,
                   "snapshot_redaction": "Common credential fields and URL userinfo, queries and fragments are removed; sha256 identifies the original effective configuration before redaction",
                   "source": str(Path(config_path).resolve()) if config_path else "checkpoint",
                   "file_sha256": sha256_file(config_path) if config_path else None,
                   "checkpoint_config_sha256": _json_hash(checkpoint_config) if checkpoint_config is not None else None},
        "dataset": {
            "split": split, "source": str(loader.dataset.source), "synthetic": synthetic,
            "synthetic_provenance": "generated" if loader.dataset.source == "synthetic" else "user_declared" if synthetic is not None else "unknown",
            "count": sample_count, "seed": loader.dataset.seed,
            "source_manifest": source_manifest,
            "source_manifest_sha256": _json_hash(source_manifest),
            "input_manifest_sha256": _json_hash([item["input_sha256"] for item in examples]),
            "target_manifest_sha256": _json_hash([item["target_sha256"] for item in examples]),
            "tensor_hash_basis": "Per-sample transformed model inputs / original target names, dtypes, shapes and contiguous bytes in dataset order",
            "trajectory_target_original_scale": trajectory_scale,
            "split_isolation": "Explicit synthetic split seeds or dataset split directories; independence from checkpoint training is not independently verified",
        },
        "counts": {"samples": sample_count, "batches": batches},
        "label_schema": {"health_states": HEALTH_STATES,
                         "drivers": DRIVER_CLASSES if driver_count == len(DRIVER_CLASSES) else [f"driver_{index}" for index in range(driver_count)],
                         "horizon_days": list(horizons),
                         "mapping_provenance": "Canonical ForestMind class order; external labels must follow this order"},
        "metrics": result,
        "metric_units": {"health_mae": "score points [0, 100]", "trajectory_mae": "unit score [0, 1]",
                         "trajectory_rmse": "unit score [0, 1]", "trajectory_per_horizon_mae": "unit score [0, 1]",
                         "aggregate_score": "heuristic weighted combination; not a probability"},
        "per_horizon": [{"horizon_days": int(day), "n": int(trajectory_mask[:, index].sum()),
                         "mae": float(np.abs(errors[trajectory_mask[:, index], index]).mean()) if trajectory_mask[:, index].any() else None,
                         "rmse": float(np.sqrt(np.mean(errors[trajectory_mask[:, index], index] ** 2))) if trajectory_mask[:, index].any() else None}
                        for index, day in enumerate(horizons)],
        "losses": averaged_losses,
        "loss_observed_counts": loss_counts,
        "loss_aggregation": "Observed-label weighted component means; task totals recomposed from configured weights. Unknown labels are excluded.",
        "examples": examples,
        "calibration": {
            "status": "uncalibrated", "fitted": False,
            "classification_threshold": 0.5,
            "probabilities": "Raw model softmax/sigmoid scores are retained for future calibration; not validated confidence",
            "trajectory_uncertainty": "Raw model standard deviation on the unit scale; not a calibrated coverage interval",
            "example_role": "validation_only" if split == "val" else "held_out_evaluation_only",
            "guidance": "Fit any calibrator using a separate calibration/validation split; evaluate once on untouched test examples. Never fit calibration from test outcomes.",
        },
        "runtime": {"forestmind": __version__, "python": platform.python_version(),
                    "torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
                    "seed": seed, "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "implementation_sha256": {name: sha256_file(repo_root / name) for name in implementation_paths}},
        "synthetic_warning": (
            "SYNTHETIC DATA: these metrics evaluate simulated inputs and generated labels; they do not establish real-world ecological performance."
            if synthetic else None
        ),
        "caveats": [
            "Evaluation does not calibrate model probabilities or uncertainty and does not establish causation, tipping probabilities or tipping dates.",
            "This script does not prove the requested split was never used for checkpoint selection or upstream preprocessing; audit training provenance separately.",
            "Validation metrics are for development and are not a substitute for held-out test performance.",
            "Dataset hashes identify exact content, not label quality, independence or ecological validity.",
            "Disk data synthetic status is unknown unless explicitly declared; file-based input alone does not establish real observations.",
            "Floating-point results may differ across devices, library versions and kernels; input, target and checkpoint hashes permit content verification.",
        ],
    }
    # Fail on non-finite or unsupported values before the caller publishes a report.
    json.dumps(report, allow_nan=False)
    return report


def write_evaluation_report(report: dict, output_path: str | Path) -> Path:
    """Write only strict JSON and preserve the checkpoint, config and dataset inputs."""
    output = Path(output_path).resolve()
    protected = [report["checkpoint"]["path"]]
    if report["config"]["source"] != "checkpoint":
        protected.append(report["config"]["source"])
    protected.extend(item["path"] for item in report["dataset"]["source_manifest"])
    if output in {Path(path).resolve() for path in protected}:
        raise ValueError("Evaluation output must not overwrite a checkpoint, configuration or input sample")
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    return output
