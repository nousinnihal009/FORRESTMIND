"""Selected-area analysis orchestration with immutable evidence manifests.

Checkpoint registration is operator-controlled local configuration. HTTP callers
cannot supply model paths, code, download URLs or pickle payloads.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import threading
from uuid import uuid4

import numpy as np

from forestmind import __version__
from forestmind.data.sequence import prepare_sequence, write_sequence
from forestmind.monitoring.analysis import METHOD, analyze_region
from forestmind.monitoring.coverage import assess_history
from forestmind.monitoring.evaluation import backtest_prepared
from forestmind.monitoring.spatial import fragmentation_metrics, geometry_bbox, recovery_diagnostics, spectral_change
from forestmind.monitoring.projections import scenario_projection

MAX_WORKFLOW_JOBS = 10000
MAX_WORKFLOW_SNAPSHOTS = 2048
MAX_REGION_SNAPSHOTS = 256
MAX_WORKFLOW_METADATA_BYTES = 256 * 1024**2
MAX_WORKFLOW_ARTIFACT_BYTES = 1024**3
MAX_WORKFLOW_RECORD_BYTES = 16 * 1024**2
WORKFLOW_TABLES = ("workflow_snapshots", "workflow_bindings", "workflow_jobs", "workflow_context", "model_runs", "scenarios")
IMPLEMENTATION_FILES = ("workflow.py", "spatial.py", "climate.py", "projections.py", "landcover.py", "radar.py", "analysis.py", "coverage.py", "evaluation.py", "../data/sequence.py", "../inference/predict.py")


class WorkflowCapacityError(ValueError):
    """Refuse new work without removing existing reproducible evidence."""


def file_fingerprint(path):
    """Cheap freshness hint; execution still verifies the full pinned SHA256."""
    try:
        stat = Path(path).stat()
        return {"bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}
    except OSError:
        return None


def implementation_hashes():
    return {name: file_hash(Path(__file__).parent / name) for name in IMPLEMENTATION_FILES}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def manifest_for(region: dict, observations: list[dict], cutoff=None, model=None) -> dict:
    cutoff = cutoff or observations[-1]["date"]
    rows = [row for row in observations if row["date"] <= cutoff]
    manifest = {"schema_version": 1, "region_id": region["id"], "geometry": region["geometry"],
                "geometry_sha256": digest(region["geometry"]), "observation_cutoff": cutoff,
                "observed_through": rows[-1]["date"] if rows else None,
                "observations_sha256": digest(rows), "observation_count": len(rows),
                "software_version": __version__, "method": METHOD,
                "model": model, "synthetic": bool(region["provenance"].get("synthetic", False))}
    manifest["id"] = digest(manifest)
    return manifest


def capability(key, name, available, reason, requires=(), *, evidence=None, status=None):
    return {"id": key, "name": name, "available": bool(available),
            "status": status or ("available" if available else "unavailable"), "reason": reason,
            "requires": list(requires), "evidence": evidence}


def model_eligibility(registry: list[dict], region: dict, prepared) -> tuple[dict | None, list[dict]]:
    """Require real held-out evidence, matching spatial/temporal/input scope and hashes."""
    checks = []
    if not registry:
        return None, [{"id": None, "reasons": ["No trusted local model is registered. Supply a real-trained compatible checkpoint and independent evaluation evidence in FORESTMIND_MODEL_REGISTRY."]}]
    for entry in registry:
        reasons = []
        if region["provenance"].get("synthetic"):
            reasons.append("Real monitoring model execution does not accept synthetic regions.")
        if entry.get("trained_on_synthetic") is not False:
            reasons.append("Registry does not declare real, nonsynthetic training.")
        if not entry.get("version") or not entry.get("license") or not entry.get("target"):
            reasons.append("Registry must document model version, license and measurable prediction target.")
        checkpoint = Path(entry.get("path", ""))
        if not checkpoint.is_file() or checkpoint.stat().st_size > 2 * 1024**3:
            reasons.append("Registered checkpoint is missing or exceeds the 2 GiB limit.")
        elif file_hash(checkpoint) != entry.get("sha256"):
            reasons.append("Checkpoint SHA256 differs from the trusted registration.")
        evidence = entry.get("evaluation", {})
        evidence_path = Path(evidence.get("path", ""))
        if (not evidence_path.is_file() or evidence_path.stat().st_size > 32 * 1024**2
                or file_hash(evidence_path) != evidence.get("sha256")):
            reasons.append("A hash-pinned independent evaluation artifact is missing or changed.")
        else:
            try:
                report = json.loads(evidence_path.read_text(encoding="utf-8"))
                if (report.get("kind") != "forestmind_checkpoint_evaluation" or
                        report.get("dataset", {}).get("synthetic") is not False or
                        report.get("dataset", {}).get("split") != "test" or
                        report.get("checkpoint", {}).get("sha256") != entry.get("sha256")):
                    reasons.append("Evaluation must be real-data test-split evidence for this exact checkpoint.")
            except (ValueError, TypeError):
                reasons.append("Registered evaluation is not valid JSON.")
        if evidence.get("independence_audited") is not True or not evidence.get("acceptance_record"):
            reasons.append("An operator-reviewed holdout/leakage audit and acceptance record are required.")
        bbox = geometry_bbox(region["geometry"])
        scope = entry.get("scope_bbox")
        if (not isinstance(scope, list) or len(scope) != 4 or
                not (scope[0] <= bbox[0] < bbox[2] <= scope[2] and scope[1] <= bbox[1] < bbox[3] <= scope[3])):
            reasons.append("Selected geometry is outside the model's documented geographic scope.")
        if prepared is None:
            reasons.append("Accepted real optical cache sequence is unavailable.")
        else:
            contract = entry.get("input", {})
            count = len(prepared.arrays["timestamps"])
            if contract.get("bands") != prepared.provenance["band_names"] or contract.get("units") != "surface_reflectance_0_1":
                reasons.append("Model band order or physical units do not match the twelve-band Sentinel-2 sequence.")
            if contract.get("grid_shape") != prepared.provenance["grid_shape"]:
                reasons.append("Model grid dimensions do not match the acquired cache.")
            height, width = prepared.provenance["grid_shape"]
            pixel_m = [math.radians(bbox[2] - bbox[0]) * 6371008.8 * math.cos(math.radians((bbox[1] + bbox[3]) / 2)) / width,
                       math.radians(bbox[3] - bbox[1]) * 6371008.8 / height]
            resolution = contract.get("pixel_size_m_range")
            if (not isinstance(resolution, list) or len(resolution) != 2 or
                    not all(isinstance(v, (int, float)) and math.isfinite(v) for v in resolution) or
                    not all(resolution[0] <= value <= resolution[1] for value in pixel_m)):
                reasons.append("Actual AOI grid spacing is outside the model's documented pixel_size_m_range.")
            if contract.get("sampling") != "individual_acquisitions":
                reasons.append("Model must explicitly support individual acquisitions; this sequence is not monthly composites.")
            if not contract.get("min_steps", 1) <= count <= contract.get("max_steps", 48):
                reasons.append(f"Sequence has {count} acquisitions outside the registered length range.")
            if contract.get("required_modalities", ["optical"]) != ["optical"]:
                reasons.append("This sequence has optical imagery only; required radar/climate/activity model inputs are absent.")
            if prepared.provenance["observed_through"] <= entry.get("training_through", "9999-12-31"):
                reasons.append("Observation cutoff must follow the registered training cutoff.")
        checks.append({"id": entry.get("id"), "version": entry.get("version"), "reasons": reasons,
                       "eligible": not reasons})
        if not reasons:
            return entry, checks
    return None, checks


class WorkflowService:
    def __init__(self, store, *, registry_path: str | None = None):
        self.store = store
        self.root = store.path.parent / (store.path.stem + "-workflow")
        self.root.mkdir(exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forestmind-analysis")
        self.lock = threading.RLock()
        self.analysis_lock = threading.Lock()
        self.registry = []
        if registry_path:
            path = Path(registry_path).resolve()
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or not isinstance(document.get("models"), list):
                raise ValueError("Model registry must contain a models array")
            self.registry = document["models"]
            for entry in self.registry:
                for target in (entry, entry.get("evaluation", {})):
                    if target.get("path"):
                        target["path"] = str((path.parent / target["path"]).resolve())
        with store.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS workflow_snapshots (
                    id TEXT PRIMARY KEY, region_id TEXT NOT NULL REFERENCES regions(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS workflow_bindings (
                    region_id TEXT PRIMARY KEY REFERENCES regions(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS workflow_jobs (id TEXT PRIMARY KEY, region_id TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS workflow_context (
                    region_id TEXT NOT NULL REFERENCES regions(id), kind TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(region_id, kind));
            """)
            for row in db.execute("SELECT id,payload FROM workflow_jobs").fetchall():
                value = json.loads(row["payload"])
                if value["status"] in {"pending", "running"}:
                    value.update(status="interrupted", error="Service restarted. Rerun selected-area analysis; completed snapshots remain intact.", completed_at=now())
                    db.execute("UPDATE workflow_jobs SET payload=? WHERE id=?", (store._json(value), row["id"]))

    def _context(self, region_id, kind):
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM workflow_context WHERE region_id=? AND kind=?", (region_id, kind)).fetchone()
        return json.loads(row["payload"]) if row else None

    def _metadata_bytes(self, db):
        return sum(db.execute(f"SELECT COALESCE(SUM(length(CAST(payload AS BLOB))),0) FROM {table}").fetchone()[0]
                   for table in WORKFLOW_TABLES)

    def _check_metadata(self, db, payload, previous=None):
        size = len(payload.encode("utf-8"))
        if size > MAX_WORKFLOW_RECORD_BYTES:
            raise WorkflowCapacityError("Analysis record exceeds the 16 MiB limit. Reduce the observation history or raster grid.")
        added = size - len((previous or "").encode("utf-8"))
        if self._metadata_bytes(db) + added > MAX_WORKFLOW_METADATA_BYTES:
            raise WorkflowCapacityError("Workflow metadata storage limit reached. Back up this workspace and use a new workspace database; existing evidence is preserved.")

    def _artifact_bytes(self):
        total = 0
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise WorkflowCapacityError("Workflow artifacts contain a symbolic link; operator review is required before writing more artifacts.")
            if path.is_file():
                total += path.stat().st_size
        return total

    def _check_artifacts(self, reserve=0):
        if self._artifact_bytes() + reserve > MAX_WORKFLOW_ARTIFACT_BYTES:
            raise WorkflowCapacityError("Workflow artifact storage limit reached (1 GiB). Back up this workspace and use a new workspace database; no saved artifacts were deleted.")

    def capacity(self):
        with self.store.connection() as db:
            counts = {name: db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in ("workflow_jobs", "workflow_snapshots")}
            metadata = self._metadata_bytes(db)
        return {"metadata_bytes": metadata, "metadata_limit_bytes": MAX_WORKFLOW_METADATA_BYTES,
                "artifact_bytes": self._artifact_bytes(), "artifact_limit_bytes": MAX_WORKFLOW_ARTIFACT_BYTES,
                "jobs": counts["workflow_jobs"], "job_limit": MAX_WORKFLOW_JOBS,
                "snapshots": counts["workflow_snapshots"], "snapshot_limit": MAX_WORKFLOW_SNAPSHOTS,
                "per_region_snapshot_limit": MAX_REGION_SNAPSHOTS, "record_limit_bytes": MAX_WORKFLOW_RECORD_BYTES,
                "retention": "Refuse new work at capacity. Existing snapshots, jobs, scenarios and artifacts are never silently deleted."}

    def _check_snapshot_capacity(self, db, region_id):
        total = db.execute("SELECT COUNT(*) FROM workflow_snapshots").fetchone()[0]
        local = db.execute("SELECT COUNT(*) FROM workflow_snapshots WHERE region_id=?", (region_id,)).fetchone()[0]
        if total >= MAX_WORKFLOW_SNAPSHOTS or local >= MAX_REGION_SNAPSHOTS:
            raise WorkflowCapacityError("Analysis snapshot limit reached (2048 per workspace, 256 per area). Back up this workspace and use a new workspace database; saved snapshots remain exportable.")

    def _registry_files(self):
        return [{"id": entry.get("id"), "checkpoint": file_fingerprint(entry.get("path", "")),
                 "evaluation": file_fingerprint(entry.get("evaluation", {}).get("path", ""))} for entry in self.registry]

    @staticmethod
    def _source_files(binding):
        return [{"source": digest(row["path"]), "expected_sha256": row["sha256"], "file": file_fingerprint(row["path"])}
                for row in (binding or {}).get("sources", [])]

    def save_context(self, region_id, kind, value):
        payload = self.store._json(value)
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT payload FROM workflow_context WHERE region_id=? AND kind=?", (region_id, kind)).fetchone()
            self._check_metadata(db, payload, prior[0] if prior else None)
            db.execute("INSERT OR REPLACE INTO workflow_context VALUES (?,?,?)", (region_id, kind, payload))

    def _binding(self, region_id):
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM workflow_bindings WHERE region_id=?", (region_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def latest(self, region_id):
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM workflow_snapshots WHERE region_id=? ORDER BY rowid DESC LIMIT 1", (region_id,)).fetchone()
        if row:
            result = json.loads(row["payload"])
            saved = self.store.region(region_id)
            result["stale"] = bool(saved and (digest(saved[1]) != result["manifest"]["observations_sha256"]
                                             or digest(saved[0]["geometry"]) != result["manifest"]["geometry_sha256"]))
            if result["manifest"].get("implementation_sha256") != implementation_hashes():
                result["stale"] = True
            if result["manifest"].get("software_version") != __version__:
                result["stale"] = True
            if result["manifest"].get("registry_sha256") != digest(self.registry):
                result["stale"] = True
            if result["manifest"].get("registry_files") != self._registry_files():
                result["stale"] = True
            if result["manifest"].get("source_files") != self._source_files(self._binding(region_id)):
                result["stale"] = True
            for kind, expected in result["manifest"].get("context_sha256", {}).items():
                context = self._context(region_id, kind)
                if context and (context.get("geometry_sha256") != result["manifest"]["geometry_sha256"] or
                                context.get("end", context.get("observed_at", "9999")) > result["manifest"]["observation_cutoff"]):
                    context = None
                if (digest(context) if context else None) != expected:
                    result["stale"] = True
            return result
        saved = self.store.region(region_id)
        if saved is None:
            raise ValueError("Region not found")
        return {"status": "not_run", "manifest": manifest_for(*saved), "capabilities": [],
                "reason": "Run selected-area analysis to create an immutable evidence snapshot on the worker.", "stale": False}

    def run(self, region_id, *, cache_dir=None, cutoff=None, job_id=None, source_paths=None,
            execute_model=True, include_climate=False, include_landcover=False, include_radar=False):
        # Separate from submit's short lock: HTTP submissions never wait on raster/network/model work.
        with self.analysis_lock:
            return self._run(region_id, cache_dir=cache_dir, cutoff=cutoff, job_id=job_id,
                             source_paths=source_paths, execute_model=execute_model,
                             include_climate=include_climate, include_landcover=include_landcover, include_radar=include_radar)

    def _run(self, region_id, *, cache_dir=None, cutoff=None, job_id=None, source_paths=None,
             execute_model=True, include_climate=False, include_landcover=False, include_radar=False):
        saved = self.store.region(region_id)
        if saved is None:
            raise ValueError("Region not found")
        region, observations = saved
        self._check_artifacts()
        with self.store.connection() as db:
            self._check_snapshot_capacity(db, region_id)
            self._check_metadata(db, "")
        cutoff = cutoff or observations[-1]["date"]
        date.fromisoformat(cutoff)
        observations = [row for row in observations if row["date"] <= cutoff]
        if not observations:
            raise ValueError("No observations at or before the requested cutoff")
        analysis = analyze_region(region, observations)
        payload = {"name": region["name"], "country": region["country"], "source": region["provenance"]["source"],
                   "synthetic": region["provenance"].get("synthetic", False), "geometry": region["geometry"], "observations": observations}
        binding = self._binding(region_id)
        if binding and binding.get("geometry_sha256") != digest(region["geometry"]):
            binding = None
        if cache_dir and source_paths:
            root = Path(cache_dir).resolve()
            accepted = [str(Path(p).resolve()) for p in source_paths]
            if any(not Path(p).is_relative_to(root) for p in accepted):
                raise ValueError("Accepted caches must be within the acquisition cache root")
            prior = binding.get("sources", []) if binding else []
            sources = {row["path"]: row for row in prior}
            for path in accepted:
                sources[path] = {"path": path, "sha256": file_hash(path)}
            binding = {"geometry_sha256": digest(region["geometry"]), "sources": list(sources.values())}
            with self.store.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                previous = db.execute("SELECT payload FROM workflow_bindings WHERE region_id=?", (region_id,)).fetchone()
                serialized = self.store._json(binding)
                self._check_metadata(db, serialized, previous[0] if previous else None)
                db.execute("INSERT OR REPLACE INTO workflow_bindings VALUES (?,?)", (region_id, serialized))
        prepared, sequence_error = None, "No accepted real optical caches are bound to this region. Collect observations through the acquisition workflow."
        if binding and not payload["synthetic"]:
            try:
                paths = [Path(row["path"]) for row in binding["sources"]]
                for row in binding["sources"]:
                    if file_hash(row["path"]) != row["sha256"]:
                        raise ValueError("Accepted acquisition changed after import; reacquire or restore its original artifact")
                # Cache roots may change on a refresh; commonpath is internal and never supplied by HTTP.
                import os
                common = Path(os.path.commonpath([str(p.parent) for p in paths]))
                prepared = prepare_sequence(common, through=cutoff, time_steps=48, bbox=geometry_bbox(region["geometry"]), source_paths=paths)
            except (ValueError, OSError) as error:
                sequence_error = str(error)
        spatial = spectral_change(prepared, region["geometry"]) if prepared else {"available": False, "reason": sequence_error}
        recovery = recovery_diagnostics(observations)
        cover = self._context(region_id, "forest_cover")
        if cover and (cover["geometry_sha256"] != digest(region["geometry"]) or cover["observed_at"] > cutoff):
            cover = None
        landcover_error = None
        if include_landcover and not payload["synthetic"]:
            try:
                from forestmind.monitoring.landcover import acquire_landcover
                cover = acquire_landcover(region["geometry"], cutoff=cutoff)
                self.save_context(region_id, "forest_cover", cover)
            except Exception as error:
                from forestmind.data.adapters.network import public_error
                landcover_error = f"Forest-cover collection failed: {public_error(error)}"
        fragmentation = fragmentation_metrics(cover, region["geometry"]) if cover else {"available": False,
            "reason": landcover_error or "No forest/nonforest raster at or before this cutoff is attached. Collect WorldCover reference or import a dated classified cover raster; NDVI thresholds are not a substitute."}
        if landcover_error:
            fragmentation["collection_error"] = landcover_error
        climate = self._context(region_id, "climate")
        if climate and (climate.get("geometry_sha256") != digest(region["geometry"]) or climate["end"] > cutoff):
            climate = None
        climate_collection_error = None
        if include_climate and not payload["synthetic"]:
            try:
                from forestmind.monitoring.climate import acquire_climate
                climate = acquire_climate(region["geometry"], start=observations[0]["date"], end=cutoff)
                climate["geometry_sha256"] = digest(region["geometry"])
                self.save_context(region_id, "climate", climate)
            except Exception as error:
                logging.getLogger(__name__).warning("climate_context_failed", extra={"region_id": region_id, "error_type": type(error).__name__})
                climate_collection_error = f"Climate provider request failed ({type(error).__name__}); retry collection. Previously saved evidence remains intact."
                if not climate:
                    climate = {"available": False, "reason": climate_collection_error}
        climate = climate or {"available": False, "reason": "Collect climate context to request actual ERA5 rainfall, temperature, soil moisture and reference evapotranspiration; no climate measurements were supplied."}
        radar = self._context(region_id, "radar")
        if radar and (radar.get("geometry_sha256") != digest(region["geometry"]) or radar.get("end", "9999") > cutoff):
            radar = None
        radar_collection_error = None
        if include_radar and not payload["synthetic"]:
            try:
                from forestmind.monitoring.radar import acquire_radar
                collected = acquire_radar(region["geometry"], start=observations[0]["date"], end=cutoff)
                if collected.get("available"):
                    radar = collected
                    self.save_context(region_id, "radar", radar)
                else:
                    radar_collection_error = collected.get("reason", "No eligible radar pair was returned; previously saved evidence remains intact.")
                    if not radar:
                        radar = collected
            except Exception as error:
                from forestmind.data.adapters.network import public_error
                radar_collection_error = f"Radar collection failed: {public_error(error)}"
                if not radar:
                    radar = {"available": False, "reason": radar_collection_error}
        radar = radar or {"available": False, "reason": "Collect radar context to request a bounded same-platform, same-orbit Sentinel-1 RTC pair. No actual radar context is attached."}
        model, eligibility = model_eligibility(self.registry, region, prepared)
        model_result = {"available": False, "eligibility": eligibility, "reason": "; ".join(r for check in eligibility for r in check["reasons"])}
        model_version = None
        if model and execute_model:
            try:
                # Safe deserialization only after local registration and hash verification.
                import torch
                checkpoint = torch.load(model["path"], map_location="cpu", weights_only=True)
                config = checkpoint.get("config", {})
                data_config = config.get("data", {})
                if data_config.get("source") == "synthetic" or data_config.get("synthetic") is not False:
                    raise ValueError("Checkpoint itself must declare data.synthetic=false and a real training source")
                foundation = config.get("foundation", {})
                if foundation.get("name") != "custom_vit":
                    raise ValueError("Native pretrained offline loading is not verified; proxy checkpoints are development-only and cannot serve real monitoring")
                sequence_id = digest({"sources": prepared.provenance["sources"], "model": model["sha256"], "cutoff": cutoff})
                input_path = self.root / f"sequence-{sequence_id}.npz"
                if not input_path.exists():
                    # Uncompressed arrays plus ZIP/JSON overhead are a conservative write reservation.
                    self._check_artifacts(sum(array.nbytes for array in prepared.arrays.values()) + 1024**2)
                    write_sequence(prepared, input_path)
                from forestmind.inference.predict import predict_checkpoint
                report = predict_checkpoint(model["path"], input_path, region_id=region_id,
                                            observed_through=prepared.provenance["observed_through"], synthetic_input=False)
                run = {"id": "model-" + uuid4().hex[:12], "imported_at": now(), "verification": "local_registered_execution",
                       "registry_id": model["id"], "report": report}
                model_version = {"id": model["id"], "version": model["version"], "sha256": model["sha256"], "target": model["target"]}
                model_result = {"available": True, "eligibility": eligibility, "run_id": run["id"], "report": report,
                                "reason": "Executed compatible registered model. Registration records operator-reviewed scope; it does not independently establish scientific validity."}
            except WorkflowCapacityError:
                raise
            except Exception as error:
                model_result["reason"] = f"Registered model execution abstained ({type(error).__name__}). Verify the local checkpoint configuration and input contract."
        elif model:
            model_result["reason"] = "A registered checkpoint is eligible. Run selected-area analysis to execute it on the worker."
        forecast_available = bool(analysis["forecast"]["points"])
        resilience_available = analysis["resilience"]["status"] != "insufficient_data"
        projections = scenario_projection(analysis)
        capabilities = [
            capability("optical", "Optical observations and history", bool(observations), "Dated NDVI, NDMI and NBR; masks and source provenance retained.", evidence={"count": len(observations)}),
            capability("radar", "Radar structural evidence", radar.get("available", False), radar.get("reason", "Actual same-platform, same-orbit RTC context; radar is not a neural input and changes can reflect moisture and geometry."), ["Two calibrated RTC acquisitions sharing platform/orbit/pass, common grid and valid masks"]),
            capability("climate", "Rainfall, temperature, water context", climate.get("available", False), climate.get("reason", "Measured-provider ERA5 reanalysis context, separate from neural model inputs."), ["Successful bounded ERA5 request"]),
            capability("condition", "Forest-condition screening", analysis["health_score"] is not None, analysis["health"]["interpretation"], ["Latest NDVI/NDMI/NBR and valid fraction >= 0.4"]),
            capability("disturbance", "Spatial spectral disturbance evidence", spatial["available"], spatial["reason"], ["Two common-grid real acquisitions with common valid pixels"]),
            capability("land_cover", "Forest-cover and clearing evidence", cover is not None, "A dated classified forest-cover raster is attached; change requires a second comparable epoch." if cover else fragmentation["reason"], ["Dated classified cover raster, separate from spectral indices"]),
            capability("drivers", "Candidate drivers and competing explanations", any(d["evidence_score"] is not None for d in analysis["drivers"]), "Evidence scores are associations. Missing rainfall, fire, land-cover, human-activity and radar evidence remain explicit; cause may be uncertain."),
            capability("forecast", "30-day and one-year indicator forecast", forecast_available, analysis["forecast"]["caveat"], ["12 consecutive usable months ending at the cutoff"]),
            capability("five_year", "Five-year hypothetical sensitivity projection", projections["available"], projections["reason"] + " No trained long-horizon ecological projection or calibrated uncertainty is established.", ["Usable one-year indicator baseline; explicit scenario assumptions"], status="hypothetical" if projections["available"] else "unavailable"),
            capability("resilience", "Autocorrelation and variance", resilience_available, " ".join(analysis["resilience"]["evidence"]), ["36 consecutive usable monthly observations"]),
            capability("recovery", "Observed recovery-time diagnostics", recovery["available"], recovery["reason"], ["24 contiguous baseline months plus 3 subsequent observations"]),
            capability("fragmentation", "Patch, edge and structural connectivity", fragmentation["available"], "Four-neighbour classified-cover topology; species-specific connectivity is unvalidated." if cover else fragmentation["reason"], ["Dated forest/nonforest raster and explicit valid mask"]),
            capability("risk_map", "Spatial future-degradation probability", False, "Current maps show observed spectral change. No independently calibrated pixel-level future degradation model/reference labels are available.", ["Spatial target labels, calibrated spatial predictor and independent geographic evaluation"]),
            capability("model", "Eligible checkpoint inference", model_result["available"], model_result["reason"], ["Trusted pinned real-trained model, independent evaluation and compatible inputs"]),
            capability("scenarios", "Intervention sensitivity comparison", forecast_available, "Existing conservation, restoration, fire-prevention and climate sliders use explicit assumed coefficients; they do not estimate causal effects." if forecast_available else analysis["forecast"]["caveat"]),
            capability("evaluation", "Historical backtesting and exports", len(observations) > 1, "Rolling-origin persistence, seasonal and damped-trend comparisons preserve skipped targets. Independent field outcomes and trained baseline comparisons remain research requirements."),
            capability("alerts", "Prioritized evidence alerts", True, "Dated heuristic condition and coverage alerts; ecological evidence is distinct from collection failures."),
            capability("tipping", "Validated ecosystem tipping-point prediction", False, "No operational tipping-point target or independent longitudinal validation exists. Recovery, variance and autocorrelation are descriptive research diagnostics.")]
        manifest = manifest_for(region, observations, cutoff, model_version)
        manifest["source_manifest_sha256"] = prepared.provenance["source_manifest_sha256"] if prepared else None
        manifest["context_sha256"] = {"climate": digest(climate) if climate.get("available") else None,
                                      "forest_cover": digest(cover) if cover else None,
                                      "radar": digest(radar) if radar.get("available") else None}
        manifest["registry_sha256"] = digest(self.registry)
        manifest["registry_files"] = self._registry_files()
        manifest["source_files"] = self._source_files(binding)
        manifest["model_execution_status_sha256"] = digest({"available": model_result["available"], "reason": model_result["reason"]})
        manifest["diagnostics_sha256"] = digest({"sequence": sequence_error if prepared is None else None,
                                                  "climate": climate_collection_error or (climate.get("reason") if not climate.get("available") else None),
                                                  "radar": radar_collection_error or (radar.get("reason") if not radar.get("available") else None),
                                                  "landcover": landcover_error})
        manifest["implementation_sha256"] = implementation_hashes()
        manifest["id"] = digest({key: value for key, value in manifest.items() if key != "id"})
        result = {"schema_version": 1, "kind": "forestmind_selected_area_analysis", "created_at": now(),
                  "manifest": manifest, "stale": False, "job_id": job_id, "capabilities": capabilities,
                  "observations": observations, "condition": analysis["health"], "drivers": analysis["drivers"],
                  "forecast": analysis["forecast"], "resilience": analysis["resilience"], "recovery": recovery,
                  "spatial_change": spatial, "fragmentation": fragmentation, "forest_cover": cover,
                  "climate": climate, "radar": radar, "model": model_result,
                  "collection_errors": {"climate": climate_collection_error, "radar": radar_collection_error, "forest_cover": landcover_error},
                  "alerts": analysis["alerts"], "coverage": assess_history(payload), "evaluation": backtest_prepared(payload),
                  "scenarios": {**projections, "endpoint": f"/api/v1/regions/{region_id}/workflow/scenarios"},
                  "sequence": prepared.provenance if prepared else {"available": False, "reason": sequence_error}}
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT payload FROM workflow_snapshots WHERE id=?", (manifest["id"],)).fetchone()
            if existing:
                return json.loads(existing["payload"])
            self._check_snapshot_capacity(db, region_id)
            if model_result["available"]:
                run["manifest"] = manifest
                serialized_run = self.store._json(run)
                self._check_metadata(db, serialized_run)
                db.execute("INSERT INTO model_runs (id,region_id,payload) VALUES (?,?,?)", (run["id"], region_id, serialized_run))
            serialized = self.store._json(result)
            self._check_metadata(db, serialized)
            db.execute("INSERT INTO workflow_snapshots VALUES (?,?,?)", (manifest["id"], region_id, serialized))
        return result

    def job(self, job_id):
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM workflow_jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def _save_job(self, job):
        with self.store.connection() as db:
            db.execute("INSERT OR REPLACE INTO workflow_jobs VALUES (?,?,?)", (job["id"], job["region_id"], self.store._json(job)))

    def submit(self, region_id, include_climate=False, include_landcover=False, include_radar=False):
        with self.lock:
            with self.store.connection() as db:
                jobs = [json.loads(r["payload"]) for r in db.execute("SELECT payload FROM workflow_jobs")]
            active = [job for job in jobs if job["status"] in {"pending", "running"}]
            for job in active:
                if job["region_id"] == region_id:
                    if (job.get("include_climate", False) != include_climate or job.get("include_landcover", False) != include_landcover
                            or job.get("include_radar", False) != include_radar):
                        raise ValueError("An analysis with different collection settings is already active for this area; wait for it to finish.")
                    return job
            if len(active) >= 4:
                raise ValueError("Analysis queue is full (one worker, at most four queued/running jobs). Retry after a job finishes.")
            job = {"id": "workflow-" + uuid4().hex[:12], "region_id": region_id, "status": "pending",
                   "created_at": now(), "include_climate": include_climate, "include_landcover": include_landcover,
                   "include_radar": include_radar, "error": None}
            self._save_job(job)
            self.executor.submit(self._execute, job.copy())
            return job

    def _execute(self, job):
        job.update(status="running", started_at=now())
        self._save_job(job)
        try:
            result = self.run(job["region_id"], job_id=job["id"], include_climate=job["include_climate"],
                              include_landcover=job.get("include_landcover", False), include_radar=job.get("include_radar", False))
            job.update(status="completed", manifest_id=result["manifest"]["id"], completed_at=now())
        except Exception:
            logging.getLogger(__name__).exception("selected_area_analysis_failed", extra={"job_id": job["id"]})
            job.update(status="failed", error="Selected-area analysis failed. See local logs; prior snapshots remain available.", completed_at=now())
        self._save_job(job)
