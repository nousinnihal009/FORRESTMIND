"""Frozen, bounded monthly optical acquisition plans with resumable progress.

This chooses at most one observed, quality-passing scene in each UTC month. It
does not fill missing months, produce labels, or infer forest condition. Catalog
ranking applies only to the bounded subset returned by the selected provider.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from time import sleep
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from forestmind.data.adapters.base import BoundingBox, PREPROCESSING_VERSION, SceneMetadata
from forestmind.data.adapters.network import public_error
from forestmind.data.adapters.sentinel2 import Sentinel2Adapter, S2_BANDS
from forestmind.monitoring.ingest import _load_acquisition, validate_bbox

SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 2
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_MONTHS = 120
# Indexers and virus scanners can briefly deny replacement after a file closes.
# Six attempts wait at most 310 ms; unrelated and persistent errors still fail.
ATOMIC_REPLACE_RETRY_DELAYS = (.01, .02, .04, .08, .16)
WINDOWS_REPLACE_RETRY_ERRORS = frozenset({5, 32, 33})
PROVIDERS = {"earth-search": "element84-earth-search",
             "planetary-computer": "microsoft-planetary-computer"}
COLLECTIONS = {"earth-search": "sentinel-2-c1-l2a", "planetary-computer": "sentinel-2-l2a"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _date(value: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Dates must use YYYY-MM-DD")
    return date.fromisoformat(value)


def month_windows(start: str, end: str) -> list[dict]:
    """Inclusive UTC date endpoints, clipped to each calendar month."""
    first, last = _date(start), _date(end)
    if first > last:
        raise ValueError("start must be on or before end")
    count = (last.year - first.year) * 12 + last.month - first.month + 1
    if count > MAX_MONTHS:
        raise ValueError(f"At most {MAX_MONTHS} calendar months may be planned")
    windows = []
    current = first.replace(day=1)
    while current <= last:
        month_end = current.replace(day=monthrange(current.year, current.month)[1])
        lo, hi = max(first, current), min(last, month_end)
        windows.append({"month": current.strftime("%Y-%m"),
                        "start": datetime.combine(lo, time.min, timezone.utc).isoformat(),
                        "end": datetime.combine(hi, time.max, timezone.utc).isoformat()})
        if hi == last:
            break
        current = month_end + timedelta(days=1)
    return windows


@dataclass(frozen=True)
class HistoryRequest:
    provider: str
    bbox: tuple[float, float, float, float]
    start: str
    end: str
    patch_size: int = 64
    max_cloud_cover: float = 30.0
    min_valid_fraction: float = 0.6
    candidates_per_month: int = 3
    search_limit: int = 30
    max_attempts: int = 12
    max_downloads: int = 6
    timeout_seconds: float = 30.0

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError("provider must be earth-search or planetary-computer")
        object.__setattr__(self, "bbox", validate_bbox(self.bbox))
        month_windows(self.start, self.end)
        for key, low, high in (("patch_size", 1, 2048), ("candidates_per_month", 1, 20),
                               ("search_limit", 1, 100), ("max_attempts", 1, 10000),
                               ("max_downloads", 0, 1000)):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be an integer between {low} and {high}")
        if self.search_limit < self.candidates_per_month:
            raise ValueError("search_limit must be at least candidates_per_month")
        for key, low, high, exclusive in (("max_cloud_cover", 0., 100., False),
                                           ("min_valid_fraction", 0., 1., True),
                                           ("timeout_seconds", 0., 300., True)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            if value > high or (value <= low if exclusive else value < low):
                raise ValueError(f"{key} is outside its supported bounds")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["bbox"] = list(self.bbox)
        return result


def _sanitize(value):
    """Strip credentials/query signatures before any provider metadata is saved."""
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()
                if not re.search(r"authorization|password|secret|token|credential|signature|api.?key", key, re.I)}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        def unsigned(match):
            parsed = urlsplit(match.group(0))
            host = parsed.hostname or ""
            if parsed.port:
                host += f":{parsed.port}"
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        return re.sub(r"https?://[^\s\"'<>]+", unsigned, value, flags=re.I)
    return value


def _safe_scene(scene: SceneMetadata, request: HistoryRequest, window: dict) -> dict:
    if not isinstance(scene.scene_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", scene.scene_id):
        raise ValueError("Unsafe scene id in catalog metadata")
    acquired = scene.datetime
    if acquired.tzinfo is None or acquired.utcoffset() is None:
        raise ValueError("Scene acquisition must include a timezone")
    acquired = acquired.astimezone(timezone.utc)
    if not datetime.fromisoformat(window["start"]) <= acquired <= datetime.fromisoformat(window["end"]):
        raise ValueError("Scene acquisition falls outside its requested month window")
    cloud = scene.cloud_cover
    if isinstance(cloud, bool) or not isinstance(cloud, (int, float)) or not math.isfinite(cloud):
        raise ValueError("Scene must declare finite cloud cover")
    if not 0 <= cloud <= request.max_cloud_cover:
        raise ValueError("Scene cloud cover exceeds the request")
    item = _sanitize(scene.stac_item)
    if not isinstance(item, dict) or item.get("id") != scene.scene_id:
        raise ValueError("Scene requires matching STAC item metadata")
    if item.get("collection") != COLLECTIONS[request.provider]:
        raise ValueError("Scene collection differs from the selected provider")
    if scene.extra.get("provider") != PROVIDERS[request.provider]:
        raise ValueError("Scene provider differs from the request")
    item_time = datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00"))
    if item_time.tzinfo is None or item_time != acquired:
        raise ValueError("STAC and scene acquisition timestamps disagree")
    bbox = validate_bbox([scene.bbox.west, scene.bbox.south, scene.bbox.east, scene.bbox.north])
    result = {"scene_id": scene.scene_id, "datetime": acquired.isoformat(), "bbox": list(bbox),
              "cloud_cover": cloud, "stac_item": item,
              "extra": {"provider": PROVIDERS[request.provider]}}
    result["candidate_sha256"] = _digest(result)
    return result


def _scene(candidate: dict) -> SceneMetadata:
    return SceneMetadata(scene_id=candidate["scene_id"], sensor="sentinel2",
                         datetime=datetime.fromisoformat(candidate["datetime"]),
                         bbox=BoundingBox(*candidate["bbox"]), cloud_cover=candidate["cloud_cover"],
                         stac_item=candidate["stac_item"], extra=candidate["extra"].copy())


def _json_path(path: str | Path) -> Path:
    result = Path(path).resolve()
    if result.suffix.lower() != ".json":
        raise ValueError("Plan and report paths must end in .json")
    return result


def _atomic_json(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError("History JSON exceeds the 32 MiB input/output limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        # The writer is closed before every attempt. Keep using the same
        # complete temporary file and atomic replacement; never delete or
        # truncate the destination to work around a Windows sharing lock.
        for attempt in range(len(ATOMIC_REPLACE_RETRY_DELAYS) + 1):
            try:
                temporary.replace(path)
                break
            except OSError as error:
                if (getattr(error, "winerror", None) not in WINDOWS_REPLACE_RETRY_ERRORS
                        or attempt == len(ATOMIC_REPLACE_RETRY_DELAYS)):
                    raise
                sleep(ATOMIC_REPLACE_RETRY_DELAYS[attempt])
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("History JSON exceeds the 32 MiB input limit")
    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _adapter(request: HistoryRequest, cache_dir: Path, factory=None):
    return (factory or Sentinel2Adapter)(cache_dir=cache_dir, patch_size=request.patch_size,
                                        timeout_seconds=request.timeout_seconds, provider=request.provider)


def create_history_plan(request: HistoryRequest, output: str | Path, *, adapter_factory=None) -> dict:
    """Query each month once and persist a reproducible, unsigned candidate plan."""
    path = _json_path(output)
    if path.exists():
        raise FileExistsError("Plan already exists; choose a new --plan path")
    plan = {"kind": "forestmind_history_plan", "schema_version": SCHEMA_VERSION,
            "created_at": _now(), "preprocessing_version": PREPROCESSING_VERSION,
            "request": request.to_dict(), "status": "planning", "months": [],
            "selection": "Lowest scene cloud cover, then acquisition UTC and scene id, within the bounded catalog subset; AOI quality is checked during execution.",
            "limitations": ["No missing month is filled.", "Scene cloud cover is not AOI coverage.",
                            "This plan contains unsigned metadata, not independently verified ground truth."]}
    # The catalog adapter creates its cache directory even for searches. Keep it
    # isolated and temporary; planning never writes imagery into the output tree.
    with tempfile.TemporaryDirectory(prefix="forestmind-plan-") as scratch:
        adapter = _adapter(request, Path(scratch), adapter_factory)
        for window in month_windows(request.start, request.end):
            row = {**window, "status": "searched", "candidates": [], "discarded": []}
            try:
                scenes = adapter.search(BoundingBox(*request.bbox), window["start"], window["end"],
                                        max_cloud_cover=request.max_cloud_cover, max_results=request.search_limit)
                if len(scenes) > request.search_limit:
                    raise ValueError("Catalog returned more scenes than the bounded search limit")
                unique = {}
                for scene in scenes:
                    try:
                        candidate = _safe_scene(scene, request, window)
                        if candidate["scene_id"] in unique:
                            raise ValueError("Duplicate scene id in catalog results")
                        unique[candidate["scene_id"]] = candidate
                    except (ValueError, TypeError, KeyError) as error:
                        row["discarded"].append({"reason": _sanitize(public_error(error))})
                ordered = sorted(unique.values(), key=lambda item: (item["cloud_cover"], item["datetime"], item["scene_id"]))
                row["returned_count"] = len(scenes)
                row["eligible_count"] = len(ordered)
                row["candidates"] = ordered[:request.candidates_per_month]
            except Exception as error:
                row.update(status="search_failed", reason=_sanitize(public_error(error)))
            plan["months"].append(row)
            plan["plan_sha256"] = _digest({key: value for key, value in plan.items() if key != "plan_sha256"})
            _atomic_json(path, plan)
    plan["status"] = "planned"
    plan["plan_sha256"] = _digest({key: value for key, value in plan.items() if key != "plan_sha256"})
    _atomic_json(path, plan)
    return plan


def load_history_plan(path: str | Path) -> tuple[dict, HistoryRequest]:
    plan = _read_json(_json_path(path))
    if plan.get("kind") != "forestmind_history_plan" or plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported history plan")
    if plan.get("plan_sha256") != _digest({key: value for key, value in plan.items() if key != "plan_sha256"}):
        raise ValueError("History plan hash mismatch; use the frozen original plan")
    if plan.get("status") != "planned" or plan.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("History plan is unfinished or uses incompatible preprocessing")
    request = HistoryRequest(**plan["request"])
    windows = month_windows(request.start, request.end)
    if len(plan["months"]) != len(windows):
        raise ValueError("History plan has missing or extra month windows")
    seen = set()
    for month, window in zip(plan["months"], windows):
        if any(month.get(key) != value for key, value in window.items()):
            raise ValueError("History plan month boundaries disagree with the request")
        if month.get("status") not in {"searched", "search_failed"}:
            raise ValueError("Invalid month search status")
        candidates = month["candidates"]
        if len(candidates) > request.candidates_per_month:
            raise ValueError("Plan exceeds its candidate budget")
        for candidate in candidates:
            if candidate != _safe_scene(_scene(candidate), request, window):
                raise ValueError("Candidate metadata is not canonical, unsigned, or hash-valid")
            if candidate["scene_id"] in seen:
                raise ValueError("History plan contains duplicate scenes")
            seen.add(candidate["scene_id"])
        if candidates != sorted(candidates, key=lambda item: (item["cloud_cover"], item["datetime"], item["scene_id"])):
            raise ValueError("Candidate ordering does not match the frozen selection rule")
    return plan, request


def _inspect_cache(path: Path, candidate: dict, request: HistoryRequest) -> dict:
    record = _load_acquisition(path, "sentinel2", request.bbox)
    if record is None:
        raise ValueError("Cached sensor or AOI differs from the plan")
    if (record["scene_id"] != candidate["scene_id"] or record["datetime"].isoformat() != candidate["datetime"]
            or record["provider"] != PROVIDERS[request.provider]
            or record["collection"] != COLLECTIONS[request.provider]):
        raise ValueError("Cached scene identity differs from the frozen plan")
    if record["source_band_names"] != S2_BANDS or record["grid_shape"] != [request.patch_size] * 2:
        raise ValueError("Cached band order or grid shape differs from the plan")
    with np.load(path, allow_pickle=False) as cache:
        # Adapter provenance hashes use the base adapter's JSON serialization.
        expected_metadata = hashlib.sha256(json.dumps(candidate["stac_item"], sort_keys=True, default=str).encode()).hexdigest()
        if str(cache["source_metadata_sha256"]) != expected_metadata:
            raise ValueError("Cache source metadata hash differs from the frozen unsigned STAC item")
        data, mask = cache["data"], cache["quality_mask"].astype(bool)
        if not np.isfinite(data[:, mask]).all() or (data[:, mask] < 0).any() or (data[:, mask] > 1).any():
            raise ValueError("Cache has invalid reflectance in one or more unmasked optical bands")
    fraction = record["valid_samples"] / record["total_samples"]
    if fraction < request.min_valid_fraction:
        raise ValueError(f"AOI valid fraction {fraction:.6f} is below {request.min_valid_fraction:.6f}")
    return {"scene_id": candidate["scene_id"], "candidate_sha256": candidate["candidate_sha256"],
            "path": str(path.resolve()), "sha256": _file_hash(path),
            "datetime": record["datetime"].isoformat(), "valid_fraction": fraction,
            "valid_samples": record["valid_samples"], "total_samples": record["total_samples"]}


def _report_fields(value, required: set[str], optional: set[str] = frozenset()) -> None:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError("Invalid progress report structure")


def _report_time(value) -> None:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("Progress report timestamps must be UTC") from None


def _report_integer(value, minimum=0, maximum=None) -> None:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError("Invalid progress report counter or run index")


def _validate_report(report: dict, plan: dict, request: HistoryRequest, root: Path) -> None:
    """Check an intact local checkpoint; its digest is not source authentication."""
    if report.get("report_sha256") != _digest({k: v for k, v in report.items() if k != "report_sha256"}):
        raise ValueError("Progress report hash mismatch; restore the intact original report")
    _report_fields(report, {"kind", "schema_version", "plan_sha256", "cache_dir", "created_at",
                            "updated_at", "months", "runs", "status", "accepted_cache_paths", "report_sha256"})
    if (report["kind"] != "forestmind_history_report" or report["schema_version"] != REPORT_SCHEMA_VERSION
            or report["plan_sha256"] != plan["plan_sha256"] or not isinstance(report["cache_dir"], str)
            or Path(report["cache_dir"]).resolve() != root):
        raise ValueError("Progress report does not match the frozen plan and cache directory")
    if report["status"] not in {"running", "complete", "incomplete"}:
        raise ValueError("Invalid progress report status")
    _report_time(report["created_at"])
    _report_time(report["updated_at"])
    runs, months = report["runs"], report["months"]
    if not isinstance(runs, list) or not runs or not isinstance(months, list) or len(months) != len(plan["months"]):
        raise ValueError("Invalid progress report runs or months")
    for index, run in enumerate(runs):
        _report_fields(run, {"run_index", "started_at", "attempt_limit", "download_limit", "attempts", "downloads", "status"},
                       {"finished_at", "stop_reason"})
        for key in ("run_index", "attempt_limit", "download_limit", "attempts", "downloads"):
            _report_integer(run[key])
        if (run["run_index"] != index or run["attempt_limit"] != request.max_attempts
                or run["download_limit"] != request.max_downloads or run["attempts"] > run["attempt_limit"]
                or run["downloads"] > min(run["download_limit"], run["attempts"])):
            raise ValueError("Progress report run counters exceed or differ from frozen limits")
        _report_time(run["started_at"])
        if run["status"] == "running":
            if index != len(runs) - 1 or report["status"] != "running" or "finished_at" in run or "stop_reason" in run:
                raise ValueError("Invalid running progress report state")
        elif run["status"] in {"complete", "incomplete", "interrupted"}:
            if "finished_at" not in run or "stop_reason" not in run:
                raise ValueError("Finished progress report run lacks completion metadata")
            _report_time(run["finished_at"])
            reasons = {"complete": {"complete"}, "incomplete": {"budget_exhausted", "candidates_exhausted"},
                       "interrupted": {"interrupted"}}
            if run["stop_reason"] not in reasons[run["status"]]:
                raise ValueError("Progress report run status and stop reason disagree")
        else:
            raise ValueError("Invalid progress report run status")
    if runs[-1]["status"] != report["status"]:
        raise ValueError("Progress report status differs from its latest run")
    counted = [[0, 0] for _ in runs]
    running_attempts = 0
    paths = []
    for planned, row in zip(plan["months"], months):
        _report_fields(row, {"month", "accepted", "attempts", "status"}, {"reason"})
        if row["month"] != planned["month"] or not isinstance(row["attempts"], list):
            raise ValueError("Progress report month identity or attempts mismatch")
        expected_status = "accepted" if row["accepted"] is not None else "pending" if report["status"] == "running" else "missing"
        if row["status"] != expected_status or ("reason" in row and not isinstance(row["reason"], str)):
            raise ValueError("Invalid progress report month status")
        candidates = {item["scene_id"]: item for item in planned["candidates"]}
        active = None
        previous_run = -1
        for attempt in row["attempts"]:
            common = {"scene_id", "candidate_sha256", "run_index", "stage", "status"}
            if not isinstance(attempt, dict) or not common <= attempt.keys():
                raise ValueError("Invalid progress attempt structure")
            index = attempt["run_index"]
            _report_integer(index, maximum=len(runs) - 1)
            if index < previous_run:
                raise ValueError("Progress attempts are not in run order")
            previous_run = index
            candidate = candidates.get(attempt["scene_id"])
            if candidate is None or attempt["candidate_sha256"] != candidate["candidate_sha256"]:
                raise ValueError("Progress attempt does not belong to its frozen candidate plan")
            if attempt["stage"] == "resume_validation":
                _report_fields(attempt, common | {"finished_at", "reason"})
                if (attempt["status"] not in {"invalidated", "rejected"} or active is None
                        or active["scene_id"] != attempt["scene_id"] or index <= active["run_index"]):
                    raise ValueError("Invalid resume-validation audit status or accepted scene")
                active = None
            elif attempt["stage"] == "acquisition":
                _report_fields(attempt, common | {"started_at", "download_attempted"},
                               {"finished_at", "reason", "valid_fraction", "cache_sha256"})
                if active is not None or type(attempt["download_attempted"]) is not bool:
                    raise ValueError("Invalid acquisition after acceptance or download flag")
                _report_time(attempt["started_at"])
                counted[index][0] += 1
                counted[index][1] += int(attempt["download_attempted"])
                status = attempt["status"]
                if status == "running":
                    running_attempts += 1
                    if runs[index]["status"] != "running" or attempt is not row["attempts"][-1] or "finished_at" in attempt:
                        raise ValueError("Invalid running acquisition state")
                elif status not in {"accepted", "rejected", "failed", "interrupted"}:
                    raise ValueError("Invalid progress acquisition status")
                elif "finished_at" not in attempt:
                    raise ValueError("Finished acquisition lacks a timestamp")
                if status == "accepted":
                    if "valid_fraction" not in attempt or "cache_sha256" not in attempt or "reason" in attempt:
                        raise ValueError("Accepted acquisition lacks its result")
                    fraction = attempt["valid_fraction"]
                    if (isinstance(fraction, bool) or not isinstance(fraction, (float, int))
                            or not request.min_valid_fraction <= fraction <= 1
                            or not isinstance(attempt["cache_sha256"], str)
                            or not re.fullmatch(r"[0-9a-f]{64}", attempt["cache_sha256"])):
                        raise ValueError("Invalid accepted acquisition result")
                    active = attempt
                elif ("valid_fraction" in attempt or "cache_sha256" in attempt
                      or (status != "running" and "reason" not in attempt) or (status == "running" and "reason" in attempt)):
                    raise ValueError("Invalid acquisition result fields")
            else:
                raise ValueError("Invalid progress attempt stage")
            if "finished_at" in attempt:
                _report_time(attempt["finished_at"])
            if "reason" in attempt and not isinstance(attempt["reason"], str):
                raise ValueError("Invalid progress attempt reason")
        accepted = row["accepted"]
        if (accepted is None) != (active is None):
            raise ValueError("Accepted cache differs from progress attempt history")
        if accepted is not None:
            _report_fields(accepted, {"scene_id", "candidate_sha256", "path", "sha256", "datetime", "valid_fraction", "valid_samples", "total_samples"})
            candidate = candidates.get(accepted["scene_id"])
            fraction = accepted["valid_fraction"]
            _report_integer(accepted["total_samples"], minimum=1)
            _report_integer(accepted["valid_samples"], minimum=1, maximum=accepted["total_samples"])
            if (candidate is None or accepted["scene_id"] != active["scene_id"]
                    or accepted["candidate_sha256"] != candidate["candidate_sha256"] or accepted["datetime"] != candidate["datetime"]
                    or not isinstance(accepted["path"], str) or not Path(accepted["path"]).is_absolute()
                    or not isinstance(accepted["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", accepted["sha256"])
                    or isinstance(fraction, bool) or not isinstance(fraction, (float, int))
                    or not request.min_valid_fraction <= fraction <= 1 or fraction != accepted["valid_samples"] / accepted["total_samples"]
                    or active["cache_sha256"] != accepted["sha256"] or active["valid_fraction"] != fraction):
                raise ValueError("Invalid accepted cache metadata or acquisition link")
            paths.append(accepted["path"])
    if running_attempts > 1 or counted != [[run["attempts"], run["downloads"]] for run in runs]:
        raise ValueError("Progress report run counters disagree with acquisition attempts")
    if (paths != report["accepted_cache_paths"]
            or (report["status"] != "running" and (report["status"] == "complete") != (len(paths) == len(months)))):
        raise ValueError("Progress report completion or accepted cache list disagrees with months")


def execute_history_plan(plan_path: str | Path, report_path: str | Path, cache_dir: str | Path,
                         *, resume: bool = False, adapter_factory=None) -> dict:
    """Execute bounded attempts, checkpointing before and after every acquisition.

    Attempt/download limits apply per invocation. A download is a candidate read
    attempt, even if the provider fails. Resume continues untried candidates and
    interrupted attempts; rejected or failed candidates are not silently retried.
    """
    plan_path, report_path = _json_path(plan_path), _json_path(report_path)
    if plan_path == report_path or (report_path.exists() and plan_path.samefile(report_path)):
        raise ValueError("Report must not overwrite its plan")
    plan, request = load_history_plan(plan_path)
    root = Path(cache_dir).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("cache_dir must be a directory")
    if report_path.exists() and not resume:
        raise FileExistsError("Report already exists; use --resume or choose a new --report")
    if resume and not report_path.is_file():
        raise ValueError("--resume requires an existing progress report")
    if resume:
        report = _read_json(report_path)
        try:
            _validate_report(report, plan, request, root)
        except (KeyError, TypeError, OverflowError):
            raise ValueError("Invalid progress report structure") from None
    else:
        report = {"kind": "forestmind_history_report", "schema_version": REPORT_SCHEMA_VERSION,
                  "plan_sha256": plan["plan_sha256"], "cache_dir": str(root), "created_at": _now(),
                  "months": [{"month": row["month"], "accepted": None, "attempts": []} for row in plan["months"]],
                  "runs": []}
    adapter = _adapter(request, root, adapter_factory)
    if report["runs"] and report["runs"][-1]["status"] == "running":
        report["runs"][-1].update(status="interrupted", finished_at=_now(), stop_reason="interrupted")
    run = {"run_index": len(report["runs"]), "started_at": _now(), "attempt_limit": request.max_attempts,
           "download_limit": request.max_downloads, "attempts": 0, "downloads": 0, "status": "running"}
    report["runs"].append(run)
    report["status"] = "running"

    def save():
        report["updated_at"] = _now()
        report["accepted_cache_paths"] = [row["accepted"]["path"] for row in report["months"] if row["accepted"]]
        for row in report["months"]:
            row["status"] = "accepted" if row["accepted"] else "pending" if report["status"] == "running" else "missing"
            if row["status"] != "missing":
                row.pop("reason", None)
        report["report_sha256"] = _digest({key: value for key, value in report.items() if key != "report_sha256"})
        _atomic_json(report_path, report)

    # Revalidate every prior success before spending any new download budget.
    for planned, row in zip(plan["months"], report["months"]):
        candidates = {item["scene_id"]: item for item in planned["candidates"]}
        for attempt in row["attempts"]:
            if attempt.get("scene_id") not in candidates or attempt.get("candidate_sha256") != candidates[attempt["scene_id"]]["candidate_sha256"]:
                raise ValueError("Progress attempt does not belong to its frozen candidate plan")
            if attempt.get("status") == "running":
                attempt.update(status="interrupted", finished_at=_now(), reason="Previous process stopped before the attempt completed")
        accepted = row["accepted"]
        if not accepted:
            continue
        candidate = candidates.get(accepted.get("scene_id"))
        if candidate is None:
            raise ValueError("Accepted scene does not belong to its frozen candidate plan")
        expected = adapter.cache_path_for_scene(_scene(candidate), BoundingBox(*request.bbox)).resolve()
        if not expected.is_relative_to(root / "sentinel2"):
            raise ValueError("Resolved cache path escapes the selected Sentinel-2 cache directory")
        if Path(accepted["path"]).resolve() != expected:
            raise ValueError("Accepted cache path differs from the provider's deterministic cache path")
        try:
            if _file_hash(expected) != accepted["sha256"]:
                raise ValueError("Accepted cache hash changed; refusing to reuse the altered file")
            inspected = _inspect_cache(expected, candidate, request)
            if inspected != accepted:
                raise ValueError("Accepted cache contract differs from the recorded result")
        except (OSError, ValueError, KeyError) as error:
            row["accepted"] = None
            row["attempts"].append({"scene_id": candidate["scene_id"], "candidate_sha256": candidate["candidate_sha256"],
                                    "status": "invalidated" if not expected.exists() else "rejected",
                                    "run_index": run["run_index"], "stage": "resume_validation",
                                    "finished_at": _now(), "reason": _sanitize(public_error(error))})
    save()
    budget_hit = False
    for planned, row in zip(plan["months"], report["months"]):
        if row["accepted"]:
            continue
        blocked = {attempt["scene_id"] for attempt in row["attempts"] if attempt["status"] in {"rejected", "failed"}}
        for candidate in planned["candidates"]:
            if candidate["scene_id"] in blocked:
                continue
            scene = _scene(candidate)
            expected = adapter.cache_path_for_scene(scene, BoundingBox(*request.bbox)).resolve()
            if not expected.is_relative_to(root / "sentinel2"):
                raise ValueError("Resolved cache path escapes the selected Sentinel-2 cache directory")
            needs_download = not expected.exists()
            if run["attempts"] >= request.max_attempts:
                budget_hit = True
                break
            # A zero/used network budget still permits other months' cached files.
            if needs_download and run["downloads"] >= request.max_downloads:
                budget_hit = True
                continue
            attempt = {"scene_id": candidate["scene_id"], "candidate_sha256": candidate["candidate_sha256"],
                       "run_index": run["run_index"], "stage": "acquisition",
                       "started_at": _now(), "status": "running", "download_attempted": needs_download}
            row["attempts"].append(attempt)
            run["attempts"] += 1
            run["downloads"] += int(needs_download)
            save()
            try:
                if needs_download:
                    paths = adapter.download([scene], BoundingBox(*request.bbox))
                    if len(paths) != 1 or Path(paths[0]).resolve() != expected:
                        failures = getattr(adapter, "download_failures", [])
                        reason = failures[0].get("reason", "No cache returned") if failures else "No cache returned"
                        raise OSError(reason)
                accepted = _inspect_cache(expected, candidate, request)
                row["accepted"] = accepted
                attempt.update(status="accepted", valid_fraction=accepted["valid_fraction"], cache_sha256=accepted["sha256"])
            except Exception as error:
                attempt.update(status="rejected" if expected.exists() else "failed", reason=_sanitize(public_error(error)))
            finally:
                if attempt["status"] != "running":
                    attempt["finished_at"] = _now()
                save()
            if row["accepted"]:
                break
    for planned, row in zip(plan["months"], report["months"]):
        row["status"] = "accepted" if row["accepted"] else "missing"
        if not row["accepted"]:
            row["reason"] = ("Catalog search failed" if planned["status"] == "search_failed" else
                             "No eligible catalog candidates" if not planned["candidates"] else
                             "No quality-passing acquisition within the available candidates and budgets")
        else:
            row.pop("reason", None)
    report["status"] = "complete" if all(row["accepted"] for row in report["months"]) else "incomplete"
    run["status"] = report["status"]
    run["finished_at"] = _now()
    run["stop_reason"] = "complete" if report["status"] == "complete" else "budget_exhausted" if budget_hit else "candidates_exhausted"
    save()
    return report
