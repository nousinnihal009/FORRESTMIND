"""FastAPI application serving the local ForestMind monitoring workspace."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import logging
import os
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from forestmind import __version__
from forestmind.api.schemas import AnalysisRequest, ImportRegion, InferenceReport, ScenarioRequest, geometry_metadata
from forestmind.api.store import Store
from forestmind.monitoring.analysis import DISCLAIMER, METHOD, analyze_region, sensitivity_scenario
from forestmind.monitoring.coverage import assess_history
from forestmind.monitoring.demo import demo_regions
from forestmind.api.places import install_places
from forestmind.api.security import LocalOnlyMiddleware
from forestmind.monitoring.alerts import install_alerts
from forestmind.api.operations import install_operations


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summary(result: dict) -> dict:
    fields = ("id", "name", "country", "area_km2", "center", "geometry", "status", "health_score",
              "health_change", "confidence", "quality", "latest_date", "alert_count", "provenance")
    return {**{field: result[field] for field in fields}, "coverage": result["health"]["coverage"]}


def create_app(db_path: str | Path | None = None, *, seed_demo: bool = True) -> FastAPI:
    """Create an isolated app; db_path supports temporary databases in tests."""
    database = db_path or os.environ.get("FORESTMIND_DB", "data/forestmind.db")
    store = Store(database)
    store.recover_jobs(now())
    if seed_demo:
        for region, observations in demo_regions():
            if store.region(region["id"]) is None:
                store.put_region(region, observations, analyze_region(region, observations))
    app = FastAPI(title="ForestMind local monitoring", version=__version__, description=(
        "Local, single-user observation screening and explicitly synthetic demonstrations. "
        "There is no authentication. Bind to localhost. " + DISCLAIMER
    ))
    app.state.store = store
    app.add_middleware(LocalOnlyMiddleware)
    install_places(app, store)
    install_alerts(app, store)
    install_operations(app, store)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, error: RequestValidationError):
        # Omit reflected input: nonstandard JSON NaN/Infinity must return a usable
        # validation error rather than breaking the strict JSON error serializer.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]})

    @app.middleware("http")
    async def local_response_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.url.path.startswith(("/static/", "/assets/")):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def require_analysis(region_id: str) -> dict:
        result = store.analysis(region_id)
        if result is None:
            raise HTTPException(404, "Region not found")
        return result

    @app.get("/api/v1/status")
    def status():
        return {"status": "ok", "service": "ForestMind", "method": METHOD,
                "authentication": "none-local-only", "timestamp": now()}

    @app.get("/api/v1/overview")
    def overview():
        regions = [_summary(require_analysis(region["id"])) for region in store.list_regions()]
        scores = [region["health_score"] for region in regions if region["health_score"] is not None]
        modes = {region["provenance"]["mode"] for region in regions}
        mode = next(iter(modes)) if len(modes) == 1 else "mixed" if modes else "empty"
        return {
            "as_of": max((region["latest_date"] for region in regions), default=None),
            "mode": mode, "regions": regions,
            "metrics": {"regions": len(regions), "monitored_area_km2": round(sum(region["area_km2"] for region in regions), 2),
                        "active_alerts": sum(region["alert_count"] for region in regions),
                        "mean_health": round(sum(scores) / len(scores), 1) if scores else None},
            "disclaimer": DISCLAIMER,
        }

    @app.get("/api/v1/regions")
    def list_regions():
        return {"regions": [_summary(require_analysis(region["id"])) for region in store.list_regions()]}

    @app.post("/api/v1/regions/import", status_code=201)
    def import_region(payload: ImportRegion):
        center, area = geometry_metadata(payload.geometry)
        region_id = "import-" + uuid4().hex[:12]
        region = {
            "id": region_id, "name": payload.name, "country": payload.country,
            "center": center, "area_km2": area, "geometry": payload.geometry.model_dump(),
            "provenance": {
                "mode": "synthetic_import" if payload.synthetic else "imported_observations",
                "synthetic": payload.synthetic, "source": payload.source,
                "notes": payload.notes, "method": METHOD, "model": "No trained model used",
                "verified": False, "imported_at": now(),
                "geometry_note": "User-supplied polygon; area is a local planar approximation.",
                "validation": "Schema and range validated only; observations and source are not independently verified.",
            },
        }
        observations = [row.model_dump() for row in payload.observations]
        result = analyze_region(region, observations)
        store.put_region(region, observations, result)
        return result

    @app.get("/api/v1/import-example")
    def import_example():
        region, observations = demo_regions()[0]
        return {"name": "Example imported AOI", "country": "Brazil",
                "synthetic": True,
                "source": "SCHEMA EXAMPLE ONLY: these example observations are synthetic.",
                "notes": "Replace every observation and geometry with your prepared measurements before importing.",
                "geometry": region["geometry"], "observations": observations}

    @app.get("/api/v1/regions/{region_id}")
    def get_region(region_id: str):
        return require_analysis(region_id)

    @app.get("/api/v1/regions/{region_id}/health")
    def get_health(region_id: str):
        result = require_analysis(region_id)
        return {**result["health"], "data_quality": result["data_quality"], "provenance": result["provenance"]}

    @app.get("/api/v1/regions/{region_id}/forecast")
    def get_forecast(region_id: str):
        result = require_analysis(region_id)
        return {**result["forecast"], "provenance": result["provenance"]}

    @app.get("/api/v1/regions/{region_id}/drivers")
    def get_drivers(region_id: str):
        result = require_analysis(region_id)
        return {"drivers": result["drivers"], "provenance": result["provenance"]}

    @app.get("/api/v1/regions/{region_id}/alerts")
    def get_region_alerts(region_id: str):
        return {"alerts": require_analysis(region_id)["alerts"]}

    @app.get("/api/v1/regions/{region_id}/evaluation")
    def evaluate_region(region_id: str):
        """Backtest the same fixed indicator using only history before each cutoff."""
        from forestmind.monitoring.evaluation import backtest_prepared

        saved = store.region(region_id)
        if saved is None:
            raise HTTPException(404, "Region not found")
        region, observations = saved
        return backtest_prepared({
            "name": region["name"], "country": region["country"],
            "source": region["provenance"]["source"],
            "synthetic": region["provenance"].get("synthetic", False),
            "geometry": region["geometry"], "observations": observations,
        })

    @app.get("/api/v1/regions/{region_id}/coverage")
    def history_coverage(region_id: str, start: str | None = Query(None), end: str | None = Query(None)):
        saved = store.region(region_id)
        if saved is None:
            raise HTTPException(404, "Region not found")
        region, observations = saved
        try:
            return assess_history({
                "name": region["name"], "country": region["country"],
                "source": region["provenance"]["source"],
                "synthetic": region["provenance"].get("synthetic", False),
                "geometry": region["geometry"], "observations": observations,
            }, start=start, end=end)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/v1/alerts")
    def get_alerts():
        return {"alerts": store.list_alerts()}

    @app.post("/api/v1/regions/{region_id}/model-runs", status_code=201)
    def import_model_run(region_id: str, payload: InferenceReport):
        require_analysis(region_id)
        if payload.region_id != region_id:
            raise HTTPException(422, "Report region_id must match the selected monitoring region")
        run = {"id": "model-" + uuid4().hex[:12], "imported_at": now(),
               "verification": "user_provided_unverified", "report": payload.model_dump(mode="json")}
        store.save_model_run(region_id, run)
        return run

    @app.get("/api/v1/regions/{region_id}/model-runs")
    def get_model_runs(region_id: str):
        require_analysis(region_id)
        return {"runs": store.list_model_runs(region_id)}

    @app.get("/api/v1/regions/{region_id}/model-runs/{run_id}")
    def get_model_run(region_id: str, run_id: str):
        require_analysis(region_id)
        run = store.model_run(region_id, run_id)
        if run is None:
            raise HTTPException(404, "Model run not found for this region")
        return run

    def execute_analysis(job: dict, saved: tuple[dict, list[dict]]):
        job.update({"status": "running", "started_at": now()})
        store.save_job(job)
        try:
            result = analyze_region(*saved)
            job.update({"status": "completed", "completed_at": now(), "result": result})
            store.save_analysis(result, job)
        except Exception:
            logging.getLogger(__name__).exception("Analysis job %s failed", job["id"])
            job.update({"status": "failed", "completed_at": now(),
                        "error": "Analysis failed. See local server logs for details."})
            store.save_job(job)

    @app.post("/api/v1/analysis", status_code=202)
    def run_analysis(payload: AnalysisRequest, tasks: BackgroundTasks):
        saved = store.region(payload.region_id)
        if saved is None:
            raise HTTPException(404, "Region not found")
        created = now()
        job = {"id": "job-" + uuid4().hex[:12], "status": "pending", "region_id": payload.region_id,
               "created_at": created, "started_at": None, "completed_at": None, "result": None,
               "error": None, "execution": "Local background deterministic baseline analysis"}
        store.save_job(job)
        tasks.add_task(execute_analysis, job.copy(), saved)
        return job

    @app.get("/api/v1/analysis/{job_id}")
    def get_job(job_id: str):
        job = store.job(job_id)
        if job is None:
            raise HTTPException(404, "Analysis job not found")
        return job

    @app.post("/api/v1/scenarios", status_code=201)
    def scenario(payload: ScenarioRequest):
        result = sensitivity_scenario(require_analysis(payload.region_id), payload.model_dump(exclude={"region_id"}))
        result.update({"id": "scenario-" + uuid4().hex[:12], "created_at": now()})
        store.save_scenario(result)
        return result

    @app.get("/api/v1/scenarios/{scenario_id}")
    def get_scenario(scenario_id: str):
        result = store.scenario(scenario_id)
        if result is None:
            raise HTTPException(404, "Scenario not found")
        return result

    @app.get("/api/v1/regions/{region_id}/export")
    def export(region_id: str, format: str = Query("json", pattern="^(json|csv)$")):
        result = require_analysis(region_id)
        filename = f"forestmind-{region_id}.{format}"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        if format == "csv":
            output = io.StringIO(newline="")
            rows = [{**row, "provenance_mode": result["provenance"]["mode"],
                     "source": result["provenance"]["source"], "method": METHOD,
                     "score_interpretation": "Heuristic uncalibrated indicator score"} for row in result["observations"]]
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            # Spreadsheet formula injection is possible in user-supplied source strings.
            for row in rows:
                writer.writerow({key: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")) else value for key, value in row.items()})
            return Response(content=output.getvalue(), media_type="text/csv", headers=headers)
        return JSONResponse(result, headers=headers)

    web = Path(__file__).resolve().parents[1] / "web"
    if web.exists():
        app.mount("/static", StaticFiles(directory=web), name="static")
        app.mount("/assets", StaticFiles(directory=web), name="assets")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(web / "index.html", headers={"Cache-Control": "no-cache"})

    return app


# Factory entrypoint avoids creating a database merely by importing this module.
