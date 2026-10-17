"""Prepare explicitly masked inference sequences from verified Sentinel-2 caches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from forestmind.monitoring.ingest import _load_acquisition, validate_bbox


OPTICAL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


@dataclass(frozen=True)
class PreparedSequence:
    arrays: dict
    provenance: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_sequence(cache_dir: str | Path, *, through: str, time_steps: int = 12,
                     bbox=None, min_valid_fraction: float = 0.4,
                     source_paths: list[str | Path] | None = None) -> PreparedSequence:
    """Select the latest eligible dated acquisitions at one AOI without padding.

    This is a sequence of acquisitions, not monthly composites. No labels or
    radar measurements are manufactured; absent SAR has an explicit false mask.
    """
    cutoff = date.fromisoformat(through)
    if cutoff.isoformat() != through:
        raise ValueError("Use a cutoff date formatted YYYY-MM-DD")
    if isinstance(time_steps, bool) or not isinstance(time_steps, int) or not 1 <= time_steps <= 600:
        raise ValueError("time_steps must be an integer between 1 and 600")
    if isinstance(min_valid_fraction, bool) or not np.isfinite(min_valid_fraction) or not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must lie in [0, 1]")
    root = Path(cache_dir)
    directory = root / "sentinel2" if (root / "sentinel2").is_dir() else root
    if not directory.is_dir():
        raise ValueError(f"Cache directory does not exist: {directory}")
    requested_bbox = validate_bbox(bbox) if bbox is not None else None
    records, skipped = [], []
    if source_paths is None:
        paths = sorted(directory.rglob("*.npz"))
    else:
        paths = [Path(path).resolve() for path in source_paths]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("Accepted source_paths must be nonempty and unique")
        if any(not path.is_relative_to(root.resolve()) or not path.is_file() or path.suffix.lower() != ".npz" for path in paths):
            raise ValueError("Accepted sources must be existing NPZ files inside the acquisition cache")
    for path in sorted(paths):
        record = _load_acquisition(path, "sentinel2", requested_bbox)
        if record is None or record["datetime"].date() > cutoff:
            continue
        if record["source_band_names"] != OPTICAL_BANDS:
            raise ValueError(f"{path.name}: inference requires all 12 Sentinel-2 bands in canonical order")
        with np.load(path, allow_pickle=False) as cache:
            if "synthetic" in cache and bool(cache["synthetic"]):
                raise ValueError("Synthetic caches cannot be prepared as observed satellite inputs")
            data = np.asarray(cache["data"], dtype=np.float32)
            mask = np.asarray(cache["quality_mask"], dtype=bool)
            mask &= np.isfinite(data).all(axis=0) & (data >= 0).all(axis=0) & (data <= 1).all(axis=0)
            valid_fraction = float(mask.mean())
        if not mask.any() or valid_fraction < min_valid_fraction:
            skipped.append({"name": path.name, "acquired_at": record["datetime"].isoformat(),
                            "valid_fraction": valid_fraction, "reason": "Insufficient valid pixel coverage"})
            continue
        record["valid_fraction"] = valid_fraction
        records.append(record)
    if not records:
        raise ValueError("No eligible Sentinel-2 acquisitions at or before the cutoff; no sequence was produced")
    if len({tuple(round(v, 8) for v in row["bbox"]) for row in records}) != 1:
        raise ValueError("Multiple AOIs found; select exactly one with --region")
    if len({tuple(row["grid_shape"]) for row in records}) != 1:
        raise ValueError("All selected acquisitions must share one spatial grid")
    records.sort(key=lambda row: (row["datetime"], row["scene_id"]))
    if len({row["datetime"] for row in records}) != len(records):
        raise ValueError("Duplicate acquisition times found; deduplicate scenes before preparing a sequence")
    if len({row["scene_id"] for row in records}) != len(records):
        raise ValueError("Duplicate scene IDs found; deduplicate the cache before preparing a sequence")
    records = records[-time_steps:]
    optical, masks, sources = [], [], []
    for row in records:
        path = row["file"]
        with np.load(path, allow_pickle=False) as cache:
            data = np.asarray(cache["data"], dtype=np.float32)
            mask = np.asarray(cache["quality_mask"], dtype=bool)
            mask &= np.isfinite(data).all(axis=0) & (data >= 0).all(axis=0) & (data <= 1).all(axis=0)
            optical.append(np.where(mask[None], data, 0))
            masks.append(mask)
        sources.append({"name": path.name, "sha256": _sha256(path), "scene_id": row["scene_id"],
                        "acquired_at": row["datetime"].isoformat(), "valid_fraction": row["valid_fraction"],
                        "provider": row["provider"], "collection": row["collection"],
                        "preprocessing_version": row["preprocessing_version"]})
    manifest_hash = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    observed_through = records[-1]["datetime"].date().isoformat()
    stack = np.stack(optical).astype(np.float32)
    count, _, height, width = stack.shape
    arrays = {
        "optical": stack, "sar": np.zeros((count, 2, height, width), dtype=np.float32),
        "timestamps": np.asarray([(row["datetime"] - records[0]["datetime"]).total_seconds() / 86400
                                  for row in records], dtype=np.float64),
        "observation_mask": np.ones(count, dtype=bool), "optical_mask": np.ones(count, dtype=bool),
        "sar_mask": np.zeros(count, dtype=bool), "quality_mask": np.stack(masks),
        "synthetic": np.asarray(False), "label_valid": np.asarray(False),
        "normalization": np.asarray("raw"), "observed_through": np.asarray(observed_through),
        "acquisition_dates": np.asarray([row["datetime"].isoformat() for row in records]),
        "bbox": np.asarray(records[0]["bbox"], dtype=np.float64),
        "source_manifest_sha256": np.asarray(manifest_hash),
    }
    provenance = {
        "schema_version": 1, "kind": "forestmind_inference_sequence", "synthetic": False,
        "requested_cutoff": through, "observed_through": observed_through,
        "requested_time_steps": time_steps, "selected_acquisitions": count,
        "minimum_valid_fraction": min_valid_fraction, "bbox": list(records[0]["bbox"]),
        "grid_shape": [height, width], "grid_crs": "EPSG:4326", "band_names": OPTICAL_BANDS,
        "optical_units": "surface_reflectance_0_1", "normalization": "raw",
        "sar_present": False, "labels_present": False, "source_manifest_sha256": manifest_hash,
        "sources": sources, "skipped_for_quality": skipped,
        "limitations": ["Input consists of individual acquisitions, not monthly composites.",
                        "No missing dates, SAR values or labels are imputed.",
                        "A verified cache contract does not establish ground-truth forest condition.",
                        "Match the checkpoint's temporal sampling, bands, grid and normalization before inference."],
    }
    return PreparedSequence(arrays, provenance)


def write_sequence(prepared: PreparedSequence, output: str | Path) -> tuple[Path, Path]:
    """Write a pickle-free NPZ and provenance sidecar using atomic replacements."""
    destination = Path(output)
    if destination.suffix.lower() != ".npz":
        raise ValueError("Sequence output must use the .npz extension")
    sidecar = destination.with_suffix(".provenance.json")
    if destination.exists() or sidecar.exists():
        raise ValueError("Output or provenance sidecar already exists; choose new paths to preserve source artifacts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **prepared.arrays)
        provenance = {**prepared.provenance, "sequence_name": destination.name, "sequence_sha256": _sha256(temporary)}
        text = json.dumps(provenance, indent=2, allow_nan=False) + "\n"
        temporary.replace(destination)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".json", mode="w", encoding="utf-8", delete=False) as stream:
            sidecar_temp = Path(stream.name)
            stream.write(text)
        try:
            sidecar_temp.replace(sidecar)
        finally:
            sidecar_temp.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, sidecar
