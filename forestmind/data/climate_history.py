"""Bounded, resumable public ERA5 history acquisition and frozen diagnostics.

Only a new directory is initialized. Accepted source bytes are never replaced;
each resume rechecks them before making any request or deriving a new artifact.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import sleep
from urllib.error import HTTPError, URLError

from forestmind.monitoring.climate import acquire_climate
from forestmind.monitoring.climatology import (
    _context, analyze_climate, artifact_hash, freeze_climatology,
    read_json_record, write_new_artifact,
)

MAX_REQUESTS = 16
MAX_ATTEMPTS_PER_WINDOW = 2
REQUEST_INTERVAL_SECONDS = 35
MAX_SOURCE_BYTES = 16 * 1024**2


def _sealed(value):
    value["artifact_sha256"] = artifact_hash(value)
    return value


def _read(path):
    value, metadata = read_json_record(path)
    if value.get("artifact_sha256") != artifact_hash(value):
        raise ValueError(f"Artifact hash mismatch: {Path(path).name}")
    return value, metadata


def _date(value):
    if not isinstance(value, str):
        raise ValueError("Dates must be explicit YYYY-MM-DD strings")
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD")
    return result


def _windows(first, last, role):
    result = []
    while first <= last:
        end = min(last, date(first.year + 9, 12, 31))
        result.append({"role": role, "start": first.isoformat(), "end": end.isoformat(),
                       "file": f"{role}-{first.isoformat()}-{end.isoformat()}.json"})
        first = date(end.year + 1, 1, 1)
    return result


def history_plan(bbox, *, baseline_start, baseline_end, observed_start, observed_end):
    """Specify the reference period before looking at later measured departures."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox requires west, south, east and north")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox):
        raise ValueError("bbox requires finite coordinates, not booleans")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must be an ordered WGS84 rectangle")
    first, last, start, end = map(_date, (baseline_start, baseline_end, observed_start, observed_end))
    if (first.year < 1940 or first.month != 1 or first.day != 1 or last.month != 12 or last.day != 31
            or not 30 <= last.year - first.year + 1 <= 60):
        raise ValueError("Reference requires 30–60 complete calendar years from 1940 onward")
    if (start <= last or start > end or start.month != 1 or start.day != 1
            or end.month != 12 or end.day != 31 or end.year - start.year >= 10):
        raise ValueError("Observations require 1–10 complete calendar years strictly after the reference")
    if end >= datetime.now(timezone.utc).date():
        raise ValueError("Only completed historical years can be requested")
    modules = [Path(__file__), Path(__file__).parents[1] / "monitoring" / "climate.py",
               Path(__file__).parents[1] / "monitoring" / "climatology.py"]
    return _sealed({"schema_version": 1, "kind": "forestmind_climate_history_plan",
                    "bbox": list(bbox), "baseline_start": baseline_start, "baseline_end": baseline_end,
                    "observed_start": observed_start, "observed_end": observed_end,
                    "provider": "Open-Meteo", "dataset": "ERA5", "model_input": False,
                    "windows": _windows(first, last, "reference") + _windows(start, end, "observed"),
                    "maximum_attempts_per_window": MAX_ATTEMPTS_PER_WINDOW,
                    "maximum_source_bytes": MAX_SOURCE_BYTES,
                    "request_interval_seconds": REQUEST_INTERVAL_SECONDS,
                    "implementation": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in modules}})


def _write_progress(value, path):
    _sealed(value)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                     prefix=".climate-progress-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _matching_source(context, window, bbox):
    _, _, identity, _, _ = _context(context)
    expected = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    if (context["start"] != window["start"] or context["end"] != window["end"]
            or identity["location_requested"] != expected):
        raise ValueError("Provider context differs from the frozen window or requested location")
    return identity


def _persist_product(value, path):
    if path.exists():
        previous, _ = _read(path)
        if previous != value:
            raise ValueError(f"Existing derived artifact differs: {path.name}")
    else:
        write_new_artifact(value, path)


def build_climate_history(bbox, output_directory, *, baseline_start="1991-01-01",
                         baseline_end="2020-12-31", observed_start="2021-01-01",
                         observed_end="2025-12-31", max_requests=4, timeout=30, resume=False):
    """Execute a bounded history build; failures and pending windows stay explicit.

    All attempts count, including failed ones. A transient request can be retried
    at most once across every resume. HTTP 429 stops the invocation for a later
    explicit resume; permanent HTTP/input failures are never retried.
    """
    if type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS:
        raise ValueError("max_requests must be an integer in [1, 16]")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 60:
        raise ValueError("timeout must be a finite value in [1, 60] seconds")
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    plan = history_plan(bbox, baseline_start=baseline_start, baseline_end=baseline_end,
                        observed_start=observed_start, observed_end=observed_end)
    root = Path(output_directory).resolve()
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
        write_new_artifact(plan, root / "plan.json")
    else:
        stored, _ = _read(root / "plan.json")
        if stored != plan:
            raise ValueError("Resume inputs or implementation differ from the frozen plan")
    lock = root / ".build.lock"
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    try:
        return _execute(plan, root, max_requests, timeout)
    finally:
        lock.unlink(missing_ok=True)


def _execute(plan, root, budget, timeout):
    progress_path = root / "progress.json"
    rows = [{**window, "attempts": 0, "status": "pending"} for window in plan["windows"]]
    if progress_path.exists():
        report, _ = _read(progress_path)
        if report.get("plan_sha256") != plan["artifact_sha256"] or len(report.get("windows", [])) != len(rows):
            raise ValueError("Progress does not match the frozen plan")
        rows = report["windows"]
        for row, window in zip(rows, plan["windows"]):
            if (any(row.get(key) != value for key, value in window.items())
                    or type(row.get("attempts")) is not int or not 0 <= row["attempts"] <= MAX_ATTEMPTS_PER_WINDOW
                    or row.get("status") not in {"pending", "attempting", "retryable", "failed", "accepted"}):
                raise ValueError("Progress window or attempt budget is invalid")
    report = {"schema_version": 1, "kind": "forestmind_climate_history_progress",
              "plan_sha256": plan["artifact_sha256"], "windows": rows, "status": "incomplete",
              "scientifically_validated": False, "model_input": False}
    sources, identity, total_bytes = {}, None, 0
    # Audit every cached file before making even one provider request.
    for row in rows:
        path = root / row["file"]
        if path.is_symlink():
            raise ValueError("Cached source evidence must be a regular local file, not a symlink")
        if path.exists():
            context, metadata = _read(path)
            if row.get("file_sha256") not in (None, metadata["file_sha256"]):
                raise ValueError("Cached source bytes differ from the accepted source hash")
            current = _matching_source(context, row, plan["bbox"])
            if identity is not None and current != identity:
                raise ValueError("Cached contexts use different source/grid identities")
            identity = current
            total_bytes += metadata["bytes"]
            row.update(status="accepted", file_sha256=metadata["file_sha256"], bytes=metadata["bytes"])
            sources[row["file"]] = context
        elif row["status"] == "accepted":
            raise ValueError("An accepted source file is missing; refusing to silently redownload it")
    if total_bytes > len(rows) * MAX_SOURCE_BYTES:
        raise ValueError("Cached source evidence exceeds its fixed byte budget")
    _write_progress(report, progress_path)
    bbox = plan["bbox"]
    west, south, east, north = bbox
    geometry = {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}
    requests = 0
    stopped = False
    for row in rows:
        if row["status"] == "attempting" and row["attempts"] >= MAX_ATTEMPTS_PER_WINDOW:
            row.update(status="failed", error={"type": "InterruptedAttempt", "http_status": None,
                       "reason": "Interrupted request exhausted the persisted attempt budget"})
        while row["status"] not in {"accepted", "failed"} and row["attempts"] < MAX_ATTEMPTS_PER_WINDOW:
            if requests >= budget or stopped:
                break
            if requests:
                sleep(REQUEST_INTERVAL_SECONDS)
            row["attempts"] += 1
            row["status"] = "attempting"
            _write_progress(report, progress_path)
            requests += 1
            try:
                context = acquire_climate(geometry, start=row["start"], end=row["end"], timeout=timeout)
                current = _matching_source(context, row, bbox)
                if identity is not None and current != identity:
                    raise ValueError("Provider contexts use different source/grid identities")
                _sealed(context)
                if len(json.dumps(context, ensure_ascii=False, allow_nan=False).encode()) > MAX_SOURCE_BYTES:
                    raise ValueError("Provider context exceeds its source byte budget")
            except (OSError, ValueError, TypeError, KeyError) as error:
                status = error.code if isinstance(error, HTTPError) else None
                retryable = ((status in {408, 429, 500, 502, 503, 504})
                             or (status is None and isinstance(error, (TimeoutError, URLError, ConnectionError))))
                row.update(status="retryable" if retryable and row["attempts"] < MAX_ATTEMPTS_PER_WINDOW else "failed",
                           error={"type": type(error).__name__, "http_status": status,
                                  "reason": "Provider request failed" if isinstance(error, OSError) else str(error)[:300]})
                # Avoid repeated denial/rate-limit traffic to the same service.
                stopped = status in {401, 403, 429}
                _write_progress(report, progress_path)
                continue
            # Storage failures propagate; they must not consume another network attempt.
            path = root / row["file"]
            write_new_artifact(context, path)
            _, metadata = _read(path)
            identity = current
            sources[row["file"]] = context
            row.update(status="accepted", file_sha256=metadata["file_sha256"], bytes=metadata["bytes"])
            row.pop("error", None)
            _write_progress(report, progress_path)
    if all(row["status"] == "accepted" for row in rows):
        reference = freeze_climatology([sources[r["file"]] for r in rows if r["role"] == "reference"],
                                       baseline_start=plan["baseline_start"], baseline_end=plan["baseline_end"])
        departures = analyze_climate([sources[r["file"]] for r in rows if r["role"] == "observed"], reference,
                                     cutoff=plan["observed_end"])
        _persist_product(reference, root / "reference.json")
        _persist_product(departures, root / "departures.json")
        report.update(status="complete", reference_sha256=reference["artifact_sha256"],
                      departures_sha256=departures["artifact_sha256"],
                      available_reference_feature_months=reference["available_feature_months"],
                      observed_months=len(departures["months"]),
                      available_observed_feature_months=sum(v["available"] for m in departures["months"] for v in m["variables"].values()))
    report["requests_this_invocation"] = requests
    report["accepted_windows"] = sum(row["status"] == "accepted" for row in rows)
    _write_progress(report, progress_path)
    return report
