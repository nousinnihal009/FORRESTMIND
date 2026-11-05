"""Saved worldwide study-area plans; planning never queues acquisition."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from forestmind.api.schemas import Number, StrictModel
from forestmind.monitoring.landscapes import LandscapeService, MAX_PAGE


class LandscapeRequest(StrictModel):
    name: str = Field(min_length=2, max_length=120)
    country: str = Field(default="Unspecified", min_length=2, max_length=100)
    bbox: list[Number] = Field(min_length=4, max_length=4)
    tile_area_km2: Annotated[Number, Field(ge=1, le=25)] = 25.


def install_landscapes(app, store):
    service = LandscapeService(store)
    app.state.landscapes = service
    router = APIRouter(prefix="/api/v1/landscapes", tags=["landscapes"])

    def call(operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except KeyError as error:
            raise HTTPException(404, str(error).strip("'")) from None
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from None

    @router.post("", status_code=201)
    def create(payload: LandscapeRequest):
        return {"landscape": call(service.create, payload.model_dump())}

    @router.get("")
    def plans():
        return {"landscapes": service.list()}

    @router.get("/{landscape_id}")
    def page(landscape_id: str, offset: int = Query(default=0, ge=0),
             limit: int = Query(default=50, ge=1, le=MAX_PAGE)):
        return call(service.page, landscape_id, offset, limit)

    app.include_router(router)
    return service
