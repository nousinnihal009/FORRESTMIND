"""HTTP contracts for the persistent local acquisition worker."""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field

from forestmind.api.schemas import Number, StrictModel
from forestmind.monitoring.acquisition import AcquisitionConflict, AcquisitionService


class AcquisitionRequest(StrictModel):
    name: str = Field(min_length=2, max_length=120)
    country: str = Field(default="Unspecified", min_length=2, max_length=100)
    bbox: list[Number] = Field(min_length=4, max_length=4)
    provider: Literal["earth-search"]
    start: str
    end: str
    patch_size: int = Field(default=64, strict=True, ge=32, le=256)
    max_cloud_cover: Annotated[Number, Field(ge=0, le=100)] = 30.
    min_valid_fraction: Annotated[Number, Field(gt=0, le=1)] = .6
    candidates_per_month: int = Field(default=3, strict=True, ge=1, le=5)
    search_limit: int = Field(default=30, strict=True, ge=1, le=50)
    max_attempts: int = Field(default=12, strict=True, ge=1, le=72)
    max_downloads: int = Field(default=6, strict=True, ge=1, le=24)
    timeout_seconds: Annotated[Number, Field(gt=0, le=60)] = 30.


class RefreshRequest(StrictModel):
    start: str | None = None
    end: str | None = None
    max_downloads: int | None = Field(default=None, strict=True, ge=1, le=24)


class ScheduleRequest(StrictModel):
    enabled: bool = Field(strict=True)
    interval_days: int = Field(default=30, strict=True, ge=1, le=365)


def install_acquisition(app, store):
    def analyze(region_id, **kwargs):
        workflow = getattr(app.state, "workflow", None)
        if workflow is not None:
            workflow.run(region_id, **kwargs)
    service = AcquisitionService(store, on_complete=analyze)
    app.state.acquisition = service
    app.add_event_handler("startup", service.start)
    app.add_event_handler("shutdown", service.stop)
    router = APIRouter(prefix="/api/v1", tags=["acquisition"])

    def call(operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except KeyError as error:
            raise HTTPException(404, str(error).strip("'")) from None
        except AcquisitionConflict as error:
            raise HTTPException(409, str(error)) from None
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from None

    @router.get("/acquisition/capabilities")
    def capabilities():
        return call(service.capabilities)

    @router.get("/acquisition/jobs")
    def jobs():
        return {"jobs": service.jobs()}

    @router.post("/acquisition/jobs", status_code=202)
    def submit(payload: AcquisitionRequest):
        return call(service.submit, payload.model_dump())

    @router.get("/acquisition/jobs/{job_id}")
    def job(job_id: str):
        result = service.job(job_id)
        if result is None:
            raise HTTPException(404, "Acquisition job not found")
        return result

    @router.post("/acquisition/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return call(service.cancel, job_id)

    @router.post("/acquisition/jobs/{job_id}/resume", status_code=202)
    def resume(job_id: str):
        return call(service.resume, job_id)

    @router.get("/regions/{region_id}/acquisition")
    def area(region_id: str):
        return call(service.area, region_id)

    @router.post("/regions/{region_id}/refresh", status_code=202)
    def refresh(region_id: str, payload: RefreshRequest | None = None):
        return call(service.refresh, region_id, payload.model_dump(exclude_none=True) if payload else {})

    @router.get("/regions/{region_id}/schedule")
    def schedule(region_id: str):
        if store.region(region_id) is None:
            raise HTTPException(404, "Region not found")
        return service.schedule(region_id)

    @router.put("/regions/{region_id}/schedule")
    def set_schedule(region_id: str, payload: ScheduleRequest):
        return call(service.set_schedule, region_id, payload.enabled, payload.interval_days)

    app.include_router(router)
    return service
