"""Bounded selected-area analysis endpoints; model paths are never HTTP inputs."""
from __future__ import annotations

from datetime import date
import json
import os
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import Field, model_validator

from forestmind.api.schemas import Geometry, StrictModel
from forestmind.monitoring.workflow import WorkflowService, digest


class WorkflowRequest(StrictModel):
    include_climate: bool = Field(default=False, strict=True)


Pixel = Annotated[bool, Field(strict=True)]
RasterRow = Annotated[list[Pixel], Field(min_length=1, max_length=128)]


class ForestCoverRequest(StrictModel):
    """Classified cover and nodata masks, never raw arbitrary files or URLs."""
    geometry: Geometry
    observed_at: str
    source: str = Field(min_length=3, max_length=500)
    classification: str = Field(min_length=5, max_length=500)
    license: str = Field(min_length=2, max_length=200)
    data_kind: Literal["classified_forest_cover"]
    forest: list[RasterRow] = Field(min_length=1, max_length=128)
    valid_mask: list[RasterRow] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_grid(self):
        parsed = date.fromisoformat(self.observed_at)
        if parsed.isoformat() != self.observed_at:
            raise ValueError("observed_at must use YYYY-MM-DD")
        if len({len(row) for row in self.forest}) != 1:
            raise ValueError("Forest-cover rows must share one width")
        if len(self.forest) != len(self.valid_mask) or any(len(a) != len(b) for a, b in zip(self.forest, self.valid_mask)):
            raise ValueError("Forest cover and valid mask must share the same nonempty grid")
        if not any(any(row) for row in self.valid_mask):
            raise ValueError("Forest-cover raster has no valid classified pixels")
        return self


def install_workflow(app, store):
    service = WorkflowService(store, registry_path=os.environ.get("FORESTMIND_MODEL_REGISTRY"))
    app.state.workflow = service
    app.add_event_handler("shutdown", lambda: service.executor.shutdown(wait=False, cancel_futures=True))
    router = APIRouter(prefix="/api/v1", tags=["selected-area-analysis"])

    def require_region(region_id):
        saved = store.region(region_id)
        if saved is None:
            raise HTTPException(404, "Region not found")
        return saved

    @router.get("/regions/{region_id}/workflow")
    def workflow(region_id: str):
        require_region(region_id)
        return service.latest(region_id)

    @router.get("/regions/{region_id}/capabilities")
    def capabilities(region_id: str):
        require_region(region_id)
        result = service.latest(region_id)
        return {"manifest": result["manifest"], "stale": result["stale"], "capabilities": result["capabilities"]}

    @router.post("/regions/{region_id}/workflow", status_code=202)
    def submit(region_id: str, payload: WorkflowRequest | None = None):
        require_region(region_id)
        try:
            return service.submit(region_id, include_climate=bool(payload and payload.include_climate))
        except ValueError as error:
            raise HTTPException(409, str(error)) from None

    @router.get("/workflow-jobs/{job_id}")
    def job(job_id: str):
        value = service.job(job_id)
        if value is None:
            raise HTTPException(404, "Selected-area analysis job not found")
        return value

    @router.get("/regions/{region_id}/workflow/export")
    def export(region_id: str, manifest_id: str | None = Query(None, pattern=r"^[a-f0-9]{64}$")):
        require_region(region_id)
        if manifest_id:
            with store.connection() as db:
                row = db.execute("SELECT payload FROM workflow_snapshots WHERE region_id=? AND id=?", (region_id, manifest_id)).fetchone()
            if not row:
                raise HTTPException(404, "Analysis snapshot not found for the selected region")
            result = json.loads(row["payload"])
        else:
            result = service.latest(region_id)
        return JSONResponse(result, headers={"Content-Disposition": f'attachment; filename="forestmind-analysis-{result["manifest"]["id"][:16]}.json"'})

    @router.post("/regions/{region_id}/forest-cover", status_code=201)
    def attach_forest_cover(region_id: str, payload: ForestCoverRequest):
        region, _ = require_region(region_id)
        geometry = payload.geometry.model_dump()
        if digest(geometry) != digest(region["geometry"]):
            raise HTTPException(422, "Forest-cover geometry must exactly match the selected monitoring area")
        value = payload.model_dump()
        value["geometry_sha256"] = digest(geometry)
        value["sha256"] = digest(value)
        service.save_context(region_id, "forest_cover", value)
        job = service.submit(region_id)
        return {"source_sha256": value["sha256"], "job": job,
                "interpretation": "User-supplied forest-cover classification; provenance is retained but not independently verified."}

    app.include_router(router)
    return service
