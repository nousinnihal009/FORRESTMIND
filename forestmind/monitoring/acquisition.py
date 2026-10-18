"""Durable, bounded local acquisition worker backed by the application's SQLite DB.

Only a single local worker is supported. Catalog and raster work runs in its own
thread, never in request handling. Frozen plans, progress and accepted-only
provenance survive restart; existing monthly observations are never overwritten.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import logging
import math
from pathlib import Path
import threading
from uuid import uuid4

from forestmind.api.schemas import Geometry, ImportRegion, geometry_metadata
from forestmind.data.adapters.base import BoundingBox
from forestmind.data.adapters.network import public_error, restricted_acquisition
from forestmind.data.adapters.sentinel2 import Sentinel2Adapter
from forestmind.data.history import (HistoryRequest, _atomic_json, _digest, _read_json,
    _sanitize, _scene, create_history_plan, execute_history_plan, load_history_plan, month_windows)
from forestmind.monitoring.analysis import METHOD, analyze_region
from forestmind.monitoring.ingest import prepare_cached_observations

LOG = logging.getLogger(__name__)
ACTIVE = {"queued", "running", "retry_wait"}
TERMINAL = {"completed", "partial", "failed", "cancelled"}
LIMITS = {"max_area_km2": 25, "max_months": 60, "max_queued_jobs": 12,
          "max_storage_bytes": 2 * 1024 ** 3, "max_downloads": 24,
          "max_attempts": 72, "max_patch_size": 256, "max_auto_retries": 2,
          "concurrency": 1}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def geometry_for_bbox(bbox):
    w, s, e, n = bbox
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def validate_request(values: dict) -> dict:
    """Apply the smaller interactive-service limits on top of CLI validation."""
    name, country = values.get("name", ""), values.get("country", "Unspecified")
    if not isinstance(name, str) or not 2 <= len(name.strip()) <= 120:
        raise ValueError("Name must contain 2–120 characters")
    if not isinstance(country, str) or not 2 <= len(country.strip()) <= 100:
        raise ValueError("Country must contain 2–100 characters")
    fields = set(HistoryRequest.__dataclass_fields__)
    request = HistoryRequest(**{key: value for key, value in values.items() if key in fields})
    if request.provider != "earth-search":
        raise ValueError("Interactive acquisition currently supports explicit Earth Search Sentinel-2 only")
    if date.fromisoformat(request.start) < date(2015, 6, 23) or date.fromisoformat(request.end) > datetime.now(timezone.utc).date():
        raise ValueError("Choose dates from 2015-06-23 through today (UTC); provider availability varies")
    if len(month_windows(request.start, request.end)) > LIMITS["max_months"]:
        raise ValueError("Choose at most 60 calendar months per job")
    if not 32 <= request.patch_size <= LIMITS["max_patch_size"]:
        raise ValueError("patch_size must be between 32 and 256")
    if not 1 <= request.max_downloads <= LIMITS["max_downloads"] or request.max_attempts > LIMITS["max_attempts"]:
        raise ValueError("Each job permits 1–24 download attempts and at most 72 candidate attempts")
    if request.candidates_per_month > 5 or request.search_limit > 50 or request.timeout_seconds > 60:
        raise ValueError("Limit candidates to 5, catalog results to 50 and each request timeout to 60 seconds")
    w, s, e, n = request.bbox
    # Spherical rectangle area, independent of winding and stable near the poles.
    area = 6371.0088 ** 2 * math.radians(e - w) * (math.sin(math.radians(n)) - math.sin(math.radians(s)))
    if abs(s) > 85 or abs(n) > 85:
        raise ValueError("Choose an AOI between 85°S and 85°N; polar acquisition is unsupported")
    if e - w > 1 or n - s > 1 or not .0001 <= area <= LIMITS["max_area_km2"]:
        raise ValueError("Select a compact rectangle between 0.0001 and 25 km² (maximum side span 1°)")
    Geometry.model_validate(geometry_for_bbox(request.bbox))
    return {**request.to_dict(), "name": name.strip(), "country": country.strip(), "area_km2": round(area, 6)}


class AcquisitionConflict(ValueError):
    pass


class AcquisitionStopped(BaseException):
    """Control flow must escape history's per-scene Exception handlers."""


class AcquisitionService:
    def __init__(self, store, *, root=None, adapter_factory=None, on_complete=None):
        self.store = store
        self.root = Path(root or store.path.parent / (store.path.stem + "-acquisition")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.adapter_factory = adapter_factory or Sentinel2Adapter
        self.on_complete = on_complete
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.thread = None
        self._lease = None
        self.worker_status = "not_started"
        with store.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS acquisition_jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, region_id TEXT,
                    fingerprint TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS acquisition_job_status ON acquisition_jobs(status);
                CREATE UNIQUE INDEX IF NOT EXISTS acquisition_active_fingerprint
                    ON acquisition_jobs(fingerprint) WHERE status IN ('queued','running','retry_wait');
                CREATE UNIQUE INDEX IF NOT EXISTS acquisition_active_region
                    ON acquisition_jobs(region_id) WHERE region_id IS NOT NULL AND status IN ('queued','running','retry_wait');
                CREATE TABLE IF NOT EXISTS acquisition_areas (
                    region_id TEXT PRIMARY KEY REFERENCES regions(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS acquisition_snapshots (
                    job_id TEXT PRIMARY KEY, region_id TEXT NOT NULL REFERENCES regions(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS acquisition_schedules (
                    region_id TEXT PRIMARY KEY REFERENCES regions(id), payload TEXT NOT NULL);
            """)

    def _save(self, db, job):
        job["updated_at"] = utcnow()
        db.execute("UPDATE acquisition_jobs SET status=?,region_id=?,payload=? WHERE id=?",
                   (job["status"], job["region_id"], self.store._json(job), job["id"]))

    def job(self, job_id, *, progress=True):
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM acquisition_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        job = json.loads(row[0])
        if progress:
            plan_path = self.root / job_id / "plan.json"
            if plan_path.exists() and job["phase"] in {"planning", "recovered", "retrying_provider"}:
                try:
                    job["progress"]["searched_months"] = len(_read_json(plan_path)["months"])
                except (OSError, ValueError, KeyError):
                    pass
            report_path = self.root / job_id / "progress.json"
            if report_path.exists():
                try:
                    report = _read_json(report_path)
                    job["progress"].update(self._progress(report))
                except (OSError, ValueError, KeyError):
                    pass
            job["progress"]["candidate_downloads"] = job["progress"]["downloads"]
            job["progress"]["downloads"] = job["download_attempts"]
        return job

    def jobs(self, region_id=None):
        with self.store.connection() as db:
            rows = db.execute("SELECT id FROM acquisition_jobs " + ("WHERE region_id=? " if region_id else "") +
                              "ORDER BY rowid DESC LIMIT 100", (region_id,) if region_id else ()).fetchall()
        return [self.job(row[0]) for row in rows]

    def submit(self, values, *, region_id=None):
        request = validate_request(values)
        if region_id and self.store.region(region_id) is None:
            raise KeyError("Region not found")
        fingerprint = _digest({"request": request, "region_id": region_id})
        created = utcnow()
        job = {"id": "acq-" + uuid4().hex[:16], "status": "queued", "phase": "queued",
               "region_id": region_id, "request": request, "created_at": created, "updated_at": created,
               "started_at": None, "completed_at": None, "error": None, "cancel_requested": False,
               "recovery_count": 0, "retry_count": 0, "download_attempts": 0, "download_budget": request["max_downloads"],
               "resume_count": 0, "progress": {"requested_months": len(month_windows(request["start"], request["end"])),
                    "accepted_months": 0, "attempts": 0, "downloads": 0, "missing_months": []},
               "execution": "Single local worker; cancellation takes effect between bounded provider requests"}
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            reusable = "('queued','running','retry_wait','completed','partial')" if region_id is None else "('queued','running','retry_wait')"
            duplicate = db.execute("SELECT payload FROM acquisition_jobs WHERE fingerprint=? AND status IN " + reusable + " ORDER BY rowid DESC LIMIT 1",
                                   (fingerprint,)).fetchone()
            if duplicate:
                return {**json.loads(duplicate[0]), "deduplicated": True}
            if region_id and db.execute("SELECT 1 FROM acquisition_jobs WHERE region_id=? AND status IN ('queued','running','retry_wait')", (region_id,)).fetchone():
                raise AcquisitionConflict("This area already has an active acquisition job")
            if db.execute("SELECT COUNT(*) FROM acquisition_jobs WHERE status IN ('queued','running','retry_wait')").fetchone()[0] >= LIMITS["max_queued_jobs"]:
                raise AcquisitionConflict("Local queue is full (12 active jobs); wait or cancel a job")
            self._check_storage(request["patch_size"])
            db.execute("INSERT INTO acquisition_jobs VALUES (?,?,?,?,?)",
                       (job["id"], job["status"], region_id, fingerprint, self.store._json(job)))
        self.wake_event.set()
        return job

    def cancel(self, job_id):
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM acquisition_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("Acquisition job not found")
            job = json.loads(row[0])
            if job["status"] in ACTIVE:
                job["cancel_requested"] = True
                if job["status"] != "running":
                    job.update(status="cancelled", phase="cancelled", completed_at=utcnow())
                self._save(db, job)
        self.wake_event.set()
        return self.job(job_id)

    def resume(self, job_id):
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM acquisition_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("Acquisition job not found")
            job = json.loads(row[0])
            if job["status"] in ACTIVE:
                return job
            if job["status"] == "completed":
                raise AcquisitionConflict("Completed jobs cannot resume; refresh the saved area instead")
            fingerprint = db.execute("SELECT fingerprint FROM acquisition_jobs WHERE id=?", (job_id,)).fetchone()[0]
            if db.execute("SELECT 1 FROM acquisition_jobs WHERE fingerprint=? AND id<>? AND status IN ('queued','running','retry_wait')", (fingerprint, job_id)).fetchone():
                raise AcquisitionConflict("An identical request already has an active acquisition job")
            if job["region_id"] and db.execute("SELECT 1 FROM acquisition_jobs WHERE region_id=? AND status IN ('queued','running','retry_wait')", (job["region_id"],)).fetchone():
                raise AcquisitionConflict("This area already has an active acquisition job")
            if db.execute("SELECT COUNT(*) FROM acquisition_jobs WHERE status IN ('queued','running','retry_wait')").fetchone()[0] >= LIMITS["max_queued_jobs"]:
                raise AcquisitionConflict("Local queue is full")
            # An explicit resume authorizes one more original-size network budget.
            job.update(status="queued", phase="queued", completed_at=None, error=None, cancel_requested=False,
                       resume_count=job["resume_count"] + 1, retry_count=0,
                       download_budget=job["download_attempts"] + job["request"]["max_downloads"])
            self._save(db, job)
        self.wake_event.set()
        return self.job(job_id)

    def recover(self):
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT payload FROM acquisition_jobs WHERE status='running'").fetchall():
                job = json.loads(row[0])
                job.update(status="cancelled" if job["cancel_requested"] else "queued", phase="recovered",
                           recovery_count=job["recovery_count"] + 1,
                           error="Service restarted; frozen plan and completed cache files will be checked before resuming.")
                self._save(db, job)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        # OS lock is released automatically on a crash; prevents two processes
        # from treating the same running job as recoverable simultaneously.
        lock_path = self.root / "worker.lock"
        lease = lock_path.open("a+b")
        lease.seek(0)
        # Windows denies reads of a locked byte; stat is safe before attempting
        # the nonblocking lock and does not touch the existing owner's byte.
        if lock_path.stat().st_size == 0:
            lease.write(b"0")
            lease.flush()
        lease.seek(0)
        try:
            import os
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lease.close()
            self.worker_status = "another_local_worker"
            raise RuntimeError("Another acquisition worker owns this database; run one service process") from None
        self._lease = lease
        self.stop_event.clear()
        self.recover()
        self.thread = threading.Thread(target=self._loop, name="forestmind-acquisition", daemon=True)
        self.worker_status = "running"
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.wake_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                try:
                    self.tick_schedules()
                    if self.run_once():
                        continue
                except Exception:
                    LOG.exception("Acquisition worker cycle failed")
                self.wake_event.wait(2)
                self.wake_event.clear()
        finally:
            if self._lease:
                self._lease.close()
                self._lease = None
            self.worker_status = "stopped"

    def _check(self, job_id):
        job = self.job(job_id, progress=False)
        if self.stop_event.is_set() or job is None or job["cancel_requested"]:
            raise AcquisitionStopped()
        return job

    def _check_storage(self, patch_size=64):
        used = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
        reserve = patch_size ** 2 * 12 * 8 + 1024 ** 2
        if used + reserve > LIMITS["max_storage_bytes"]:
            raise AcquisitionConflict("Acquisition storage limit reached (2 GiB); archive local artifacts before retrying")
        return used

    def _reserve_download(self, job_id):
        self._check(job_id)
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            job = json.loads(db.execute("SELECT payload FROM acquisition_jobs WHERE id=?", (job_id,)).fetchone()[0])
            if job["download_attempts"] >= job["download_budget"]:
                raise AcquisitionStopped()
            self._check_storage(job["request"]["patch_size"])
            job["download_attempts"] += 1
            self._save(db, job)

    def _guarded_factory(self, job_id, frozen_months=None, skip_months=None):
        service = self
        frozen_months = frozen_months or {}
        skip_months = skip_months or set()
        def factory(**kwargs):
            adapter = service.adapter_factory(**kwargs)
            class Guard:
                def __getattr__(self, key):
                    return getattr(adapter, key)

                def cache_path_for_scene(self, *args, **kw):
                    service._check(job_id)
                    return adapter.cache_path_for_scene(*args, **kw)

                def search(self, bbox, start, end, **kw):
                    service._check(job_id)
                    if start[:7] in frozen_months:
                        # Reconstruct precisely the completed frozen candidate set.
                        return [_scene(item) for item in frozen_months[start[:7]]["candidates"]]
                    if start[:7] in skip_months:
                        return []
                    return self._retry(lambda: adapter.search(bbox, start, end, **kw), download=False)

                def download(self, *args, **kw):
                    return self._retry(lambda: adapter.download(*args, **kw), download=True)

                def _retry(self, operation, *, download):
                    for attempt in range(LIMITS["max_auto_retries"] + 1):
                        service._check(job_id)
                        if download:
                            service._reserve_download(job_id)
                        try:
                            result = operation()
                            if download and not result:
                                failures = getattr(adapter, "download_failures", [])
                                reason = failures[0].get("reason") if failures else None
                                raise OSError(reason or "Provider returned no usable cache")
                            return result
                        except (OSError, TimeoutError, ConnectionError) as error:
                            if attempt == LIMITS["max_auto_retries"]:
                                raise
                            with service.store.connection() as db:
                                job = json.loads(db.execute("SELECT payload FROM acquisition_jobs WHERE id=?", (job_id,)).fetchone()[0])
                                job["retry_count"] += 1
                                job["phase"] = "retrying_provider"
                                job["error"] = _sanitize(public_error(error))
                                service._save(db, job)
                            # Interruptible exponential backoff; retries consume the
                            # same network budget and never reset it automatically.
                            if service.stop_event.wait(min(2 ** attempt, 4)):
                                raise AcquisitionStopped()
            return Guard()
        return factory

    def _plan(self, job, directory, factory):
        path = directory / "plan.json"
        request = HistoryRequest(**{key: job["request"][key] for key in HistoryRequest.__dataclass_fields__})
        if path.exists():
            raw = _read_json(path)
            if raw.get("status") == "planned":
                return load_history_plan(path)[0]
            if raw.get("plan_sha256") != _digest({k: v for k, v in raw.items() if k != "plan_sha256"}) or raw.get("request") != request.to_dict():
                raise ValueError("Interrupted plan integrity check failed; create a fresh acquisition job")
            prior = {row["month"]: row for row in raw["months"]}
            # A fresh temporary plan replays completed search results locally and
            # requests only previously unfinished windows; then atomically adopts it.
            temporary = directory / "planning-resume.json"
            temporary.unlink(missing_ok=True)
            saved = self.store.region(job["region_id"]) if job["region_id"] else None
            skip = {row["date"][:7] for row in saved[1]} if saved else set()
            plan = create_history_plan(request, temporary, adapter_factory=self._guarded_factory(job["id"], prior, skip))
            plan["created_at"] = raw["created_at"]
            plan["months"] = [prior.get(row["month"], row) for row in plan["months"]]
            plan["plan_sha256"] = _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
            _atomic_json(path, plan)
            temporary.unlink(missing_ok=True)
        else:
            plan = create_history_plan(request, path, adapter_factory=factory)
        # Avoid downloading months already saved. All original candidates remain
        # in the immutable plan; guard's local cache/report handles later resumes.
        return plan

    @staticmethod
    def _progress(report):
        return {"requested_months": len(report["months"]),
                "accepted_months": sum(bool(row["accepted"]) for row in report["months"]),
                "attempts": sum(row["attempts"] for row in report["runs"]),
                "downloads": sum(row["downloads"] for row in report["runs"]),
                "missing_months": [row["month"] for row in report["months"] if not row["accepted"]],
                "months": [{"month": row["month"], "status": row["status"], "reason": row.get("reason")}
                           for row in report["months"]]}

    def run_once(self):
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM acquisition_jobs WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                return False
            job = json.loads(row[0])
            job.update(status="running", phase="planning", started_at=job["started_at"] or utcnow(), error=None)
            self._save(db, job)
        try:
            with restricted_acquisition(check=lambda: self._check(job["id"])):
                self._execute(job)
        except AcquisitionStopped:
            fresh = self.job(job["id"], progress=False)
            if fresh["cancel_requested"]:
                self._finish(fresh, "cancelled", "Cancelled between provider requests; saved observations are unchanged")
            elif self.stop_event.is_set():
                with self.store.connection() as db:
                    fresh.update(status="queued", phase="interrupted")
                    self._save(db, fresh)
            else:
                # Budget exhausted during a bounded provider retry: retain any
                # accepted months and make the need for an explicit resume clear.
                try:
                    with restricted_acquisition(check=lambda: self._check(job["id"])):
                        self._complete_from_report(fresh, self.root / job["id"], budget_exhausted=True)
                except AcquisitionStopped:
                    self._finish(self.job(job["id"], progress=False), "cancelled", "Cancelled before saving results")
                except Exception as error:
                    self._finish(self.job(job["id"], progress=False), "failed", _sanitize(public_error(error)))
        except Exception as error:
            LOG.warning("Acquisition %s failed: %s", job["id"], _sanitize(public_error(error)))
            self._finish(self.job(job["id"], progress=False), "failed", _sanitize(public_error(error)))
        return True

    def _execute(self, job):
        directory = self.root / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        self._check(job["id"])
        saved = self.store.region(job["region_id"]) if job["region_id"] else None
        skip = {row["date"][:7] for row in saved[1]} if saved else set()
        factory = self._guarded_factory(job["id"], skip_months=skip)
        plan = self._plan(job, directory, factory)
        # Refresh uses saved months to drop unnecessary candidate searches from a
        # new plan before any execution. Never alter a plan that has a report.
        report_path = directory / "progress.json"
        if job["region_id"] and not report_path.exists():
            saved = self.store.region(job["region_id"])
            months = {row["date"][:7] for row in saved[1]} if saved else set()
            selected = [row for row in plan["months"] if row["month"] not in months]
            already_saved = [row["month"] for row in plan["months"] if row["month"] in months]
            with self.store.connection() as db:
                fresh = json.loads(db.execute("SELECT payload FROM acquisition_jobs WHERE id=?", (job["id"],)).fetchone()[0])
                fresh["progress"]["already_saved_months"] = already_saved
                self._save(db, fresh)
            # Existing month windows still appear as explicit skipped gaps in the
            # acquisition report; the coverage API reports the combined history.
            for row in plan["months"]:
                if row["month"] in months:
                    row["candidates"] = []
                    row["discarded"].append({"reason": "Month already saved; no duplicate acquisition requested"})
            plan["plan_sha256"] = _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
            _atomic_json(directory / "plan.json", plan)
            if not selected:
                self._finish(self.job(job["id"], progress=False), "completed", None)
                self._record_success(job["region_id"])
                return
        with self.store.connection() as db:
            fresh = json.loads(db.execute("SELECT payload FROM acquisition_jobs WHERE id=?", (job["id"],)).fetchone()[0])
            fresh["phase"] = "acquiring"
            self._save(db, fresh)
        execute_history_plan(directory / "plan.json", report_path, directory / "cache",
                             resume=report_path.exists(), adapter_factory=factory)
        self._check(job["id"])
        self._complete_from_report(self.job(job["id"], progress=False), directory)

    def _complete_from_report(self, job, directory, *, budget_exhausted=False):
        report_path = directory / "progress.json"
        if not report_path.exists():
            self._finish(job, "partial", "Download budget exhausted; explicitly resume to authorize another bounded batch")
            return
        report = _read_json(report_path)
        job["progress"].update(self._progress(report))
        accepted = report["accepted_cache_paths"]
        if not accepted:
            self._finish(job, "partial" if budget_exhausted else "failed",
                         "No usable observations accepted. Inspect month reasons; resume untried candidates or create a new plan with different dates/quality settings.")
            return
        prepared = prepare_cached_observations(directory / "cache", sensor="sentinel2",
                    name=job["request"]["name"], country=job["request"]["country"],
                    bbox=job["request"]["bbox"], source_paths=accepted)
        prepared.provenance["history_selection"] = {"plan_sha256": report["plan_sha256"],
            "report_sha256": report["report_sha256"], "accepted_cache_paths": accepted,
            "missing_months": job["progress"]["missing_months"], "request": job["request"],
            "sampling": "At most one quality-passing actual acquisition per UTC month; no gap filling"}
        snapshot = {"job_id": job["id"], "created_at": utcnow(), "payload": prepared.payload,
                    "provenance": prepared.provenance, "accepted_cache_paths": accepted,
                    "plan": str(directory / "plan.json"), "report": str(report_path)}
        # Immutable files keep an earlier committed snapshot reproducible even
        # when cancellation interrupts an explicitly resumed batch before commit.
        snapshot_path = directory / ("provenance-" + _digest(snapshot)[:20] + ".json")
        snapshot["artifact_path"] = str(snapshot_path)
        _atomic_json(snapshot_path, snapshot)
        self._check(job["id"])
        self._merge(job, prepared.payload, snapshot, report, budget_exhausted)
        if self.on_complete:
            try:
                area = self.area(job["region_id"])
                source_paths = sorted({path for item in area["snapshots"] for path in item["accepted_cache_paths"]})
                self.on_complete(job["region_id"], cache_dir=str(self.root), source_paths=source_paths,
                                 cutoff=self.store.region(job["region_id"])[1][-1]["date"], job_id=job["id"])
            except Exception as error:
                # Acquisition remains a valid persisted observation result even if
                # an optional analysis dependency is unavailable.
                with self.store.connection() as db:
                    fresh = json.loads(db.execute("SELECT payload FROM acquisition_jobs WHERE id=?", (job["id"],)).fetchone()[0])
                    fresh["analysis_error"] = _sanitize(public_error(error))
                    self._save(db, fresh)

    def _merge(self, job, payload, snapshot, report, budget_exhausted):
        validated = ImportRegion.model_validate(payload)
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            fresh = json.loads(db.execute("SELECT payload FROM acquisition_jobs WHERE id=?", (job["id"],)).fetchone()[0])
            if fresh["cancel_requested"]:
                raise AcquisitionStopped()
            region_id = fresh["region_id"] or "area-" + uuid4().hex[:12]
            saved = db.execute("SELECT metadata,observations FROM regions WHERE id=?", (region_id,)).fetchone()
            if saved:
                region, observations = json.loads(saved[0]), json.loads(saved[1])
                if region["geometry"] != validated.geometry.model_dump():
                    raise ValueError("Saved area geometry differs from the acquisition plan")
            else:
                center, area = geometry_metadata(validated.geometry)
                region = {"id": region_id, "name": validated.name, "country": validated.country,
                    "center": center, "area_km2": area, "geometry": validated.geometry.model_dump(),
                    "provenance": {"mode": "satellite_acquisition", "synthetic": False, "source": validated.source,
                        "notes": validated.notes, "method": METHOD, "model": "No trained model used", "verified": False,
                        "geometry_note": "Confirmed WGS84 rectangle; not a verified forest boundary",
                        "validation": "Acquisition processing and quality checks; no independent field verification"}}
                observations = []
            by_month = {row["date"][:7]: row for row in observations}
            added = 0
            for observation in payload["observations"]:
                if observation["date"][:7] not in by_month:
                    by_month[observation["date"][:7]] = observation
                    added += 1
            merged = sorted(by_month.values(), key=lambda row: row["date"])
            if len(merged) > 600:
                raise ValueError("Saved area has reached the 600-month observation limit")
            region["provenance"].update(last_successful_update=utcnow(), acquisition_job=job["id"],
                provenance_sha256=_digest(snapshot), provider="element84-earth-search",
                collection="sentinel-2-c1-l2a", observation_cutoff=merged[-1]["date"])
            result = analyze_region(region, merged)
            db.execute("INSERT INTO regions VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET metadata=excluded.metadata,observations=excluded.observations",
                       (region_id, self.store._json(region), self.store._json(merged)))
            self.store._save_analysis(db, result)
            db.execute("INSERT OR REPLACE INTO acquisition_snapshots VALUES (?,?,?)", (job["id"], region_id, self.store._json(snapshot)))
            area_record = {"region_id": region_id, "request": job["request"], "last_successful_update": utcnow(),
                           "last_job_id": job["id"], "latest_observation": merged[-1]["date"]}
            db.execute("INSERT OR REPLACE INTO acquisition_areas VALUES (?,?)", (region_id, self.store._json(area_record)))
            missing = [window["month"] for window in month_windows(job["request"]["start"], job["request"]["end"])
                       if window["month"] not in by_month]
            fresh.update(region_id=region_id, status="partial" if missing else "completed", phase="saved",
                         completed_at=utcnow(), progress=job["progress"], added_months=added,
                         error=("Some months remain unavailable; inspect coverage and resume remaining candidates or adjust a new plan." if missing else None))
            fresh["progress"]["remaining_area_months"] = missing
            self._save(db, fresh)
            self._schedule_outcome(db, region_id, None)
            job.update(fresh)

    def _finish(self, job, status, error):
        with self.store.connection() as db:
            # Preserve counters and cancellation set by concurrent HTTP actions.
            fresh = json.loads(db.execute("SELECT payload FROM acquisition_jobs WHERE id=?", (job["id"],)).fetchone()[0])
            fresh.update(status=status, phase=status, error=error, completed_at=utcnow(), progress=job["progress"])
            self._save(db, fresh)
            if fresh["region_id"]:
                self._schedule_outcome(db, fresh["region_id"], error if status in {"failed", "partial"} else None)
        LOG.info("Acquisition completed: %s", json.dumps({"job_id": job["id"], "status": status, "region_id": job["region_id"]}))

    def _schedule_outcome(self, db, region_id, error):
        row = db.execute("SELECT payload FROM acquisition_schedules WHERE region_id=?", (region_id,)).fetchone()
        if row:
            schedule = json.loads(row[0])
            schedule["last_error"] = error
            if error is None:
                schedule["last_successful_update"] = utcnow()
            db.execute("UPDATE acquisition_schedules SET payload=? WHERE region_id=?", (self.store._json(schedule), region_id))

    def area(self, region_id):
        if self.store.region(region_id) is None:
            raise KeyError("Region not found")
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM acquisition_areas WHERE region_id=?", (region_id,)).fetchone()
            snapshots = [json.loads(item[0]) for item in db.execute("SELECT payload FROM acquisition_snapshots WHERE region_id=? ORDER BY rowid", (region_id,))]
        return {"configuration": json.loads(row[0]) if row else None, "schedule": self.schedule(region_id),
                "jobs": self.jobs(region_id), "snapshots": snapshots,
                "local_schedule_notice": "Scheduled updates run only while this local service is running"}

    def refresh(self, region_id, values=None):
        area = self.area(region_id)
        if not area["configuration"]:
            raise AcquisitionConflict("Only areas created by satellite acquisition can refresh automatically")
        request = area["configuration"]["request"].copy()
        values = values or {}
        request["end"] = values.get("end") or datetime.now(timezone.utc).date().isoformat()
        if values.get("start"):
            request["start"] = values["start"]
        else:
            end = date.fromisoformat(request["end"])
            month_index = end.year * 12 + end.month - LIMITS["max_months"]
            earliest = date(month_index // 12, month_index % 12 + 1, 1).isoformat()
            request["start"] = max(request["start"], earliest)
        if values.get("max_downloads") is not None:
            request["max_downloads"] = values["max_downloads"]
        return self.submit(request, region_id=region_id)

    def schedule(self, region_id):
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM acquisition_schedules WHERE region_id=?", (region_id,)).fetchone()
        return json.loads(row[0]) if row else {"enabled": False, "interval_days": 30, "next_run_at": None,
                                               "last_error": None, "requires_running_service": True}

    def set_schedule(self, region_id, enabled, interval_days):
        if type(enabled) is not bool or type(interval_days) is not int or not 1 <= interval_days <= 365:
            raise ValueError("Use enabled=true/false and an interval of 1–365 days")
        if self.area(region_id)["configuration"] is None:
            raise AcquisitionConflict("Acquire a real area before enabling scheduled updates")
        schedule = {"enabled": enabled, "interval_days": interval_days,
                    "next_run_at": (datetime.now(timezone.utc) + timedelta(days=interval_days)).isoformat() if enabled else None,
                    "last_error": None, "requires_running_service": True}
        with self.store.connection() as db:
            db.execute("INSERT OR REPLACE INTO acquisition_schedules VALUES (?,?)", (region_id, self.store._json(schedule)))
        return schedule

    def _record_success(self, region_id):
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM acquisition_areas WHERE region_id=?", (region_id,)).fetchone()
            if row:
                area = json.loads(row[0])
                area["last_successful_update"] = utcnow()
                db.execute("UPDATE acquisition_areas SET payload=? WHERE region_id=?", (self.store._json(area), region_id))

    def tick_schedules(self):
        now = datetime.now(timezone.utc)
        with self.store.connection() as db:
            rows = db.execute("SELECT region_id,payload FROM acquisition_schedules").fetchall()
        for row in rows:
            schedule = json.loads(row[1])
            if not schedule["enabled"] or datetime.fromisoformat(schedule["next_run_at"]) > now:
                continue
            try:
                job = self.refresh(row[0])
                schedule.update(last_job_id=job["id"], last_error=None)
            except (ValueError, KeyError) as error:
                schedule["last_error"] = str(error)
            schedule["next_run_at"] = (now + timedelta(days=schedule["interval_days"])).isoformat()
            with self.store.connection() as db:
                db.execute("UPDATE acquisition_schedules SET payload=? WHERE region_id=?", (self.store._json(schedule), row[0]))

    def capabilities(self):
        return {"provider": "earth-search", "collection": "sentinel-2-c1-l2a", "sensor": "Sentinel-2 optical",
                "limits": LIMITS, "geometry": "WGS84 rectangle; no antimeridian crossing or polar AOIs",
                "worker_status": self.worker_status,
                "attribution": "Copernicus Sentinel data, served by Element 84 Earth Search",
                "source_url": "https://earth-search.aws.element84.com/v1",
                "storage_bytes": sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file()),
                "scheduling": "Local only; service must remain running", "minimum_date": "2015-06-23",
                "cancellation": "Cooperative between bounded provider requests; completed observations are retained",
                "sampling": "At most one quality-passing acquisition per month; no fabricated gaps or auxiliary modalities",
                "resolution": "Output grid equals AOI extent / patch_size; resampling adds no native detail"}
