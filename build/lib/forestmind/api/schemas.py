"""Strict input contracts for prepared observations and sensitivity scenarios."""

from __future__ import annotations

from datetime import date
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Index = Annotated[Number, Field(ge=-1, le=1)]
Fraction = Annotated[Number, Field(ge=0, le=1)]
Percent = Annotated[Number, Field(ge=0, le=100)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Geometry(StrictModel):
    type: Literal["Polygon"]
    coordinates: list[list[list[Number]]]

    @field_validator("coordinates")
    @classmethod
    def validate_rings(cls, rings):
        if not rings or len(rings) > 32:
            raise ValueError("Polygon must contain between 1 and 32 rings")
        for ring in rings:
            if not 4 <= len(ring) <= 10000 or ring[0] != ring[-1]:
                raise ValueError("Each ring needs 4–10000 positions and must be closed")
            if any(len(point) != 2 or not -180 <= point[0] <= 180 or not -90 <= point[1] <= 90 for point in ring):
                raise ValueError("Coordinates must be [longitude, latitude] in geographic bounds")
            if max(point[0] for point in ring) - min(point[0] for point in ring) >= 180:
                raise ValueError("Polygons crossing the antimeridian are not supported by this local importer")
            area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:])) / 2
            if abs(area) < 1e-8:
                raise ValueError("Polygon ring must have nonzero area")
            # Reject self intersections without requiring the optional geospatial stack.
            def orientation(a, b, c):
                return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            def on_segment(a, b, c):
                return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
            def crosses(a, b, c, d):
                o1, o2, o3, o4 = orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b)
                if o1 * o2 < 0 and o3 * o4 < 0:
                    return True
                return any((abs(o) < 1e-12 and on_segment(p, q, r)) for o, p, q, r in (
                    (o1, a, b, c), (o2, a, b, d), (o3, c, d, a), (o4, c, d, b)))
            # Bound quadratic validation work and payload size with a compact AOI limit.
            if len(ring) > 512:
                raise ValueError("Simplify AOI rings to at most 512 positions")
            segments = list(zip(ring, ring[1:]))
            for i, (a, b) in enumerate(segments):
                if a == b:
                    raise ValueError("Polygon ring has a zero-length segment")
                for j in range(i + 2, len(segments)):
                    if i == 0 and j == len(segments) - 1:
                        continue
                    if crosses(a, b, *segments[j]):
                        raise ValueError("Polygon ring must not intersect itself")
        if len(rings) > 1:
            raise ValueError("Only simple polygons without interior holes are currently supported")
        return rings


class Observation(StrictModel):
    date: str
    ndvi: Index | None = None
    ndmi: Index | None = None
    nbr: Index | None = None
    precipitation_anomaly: Annotated[Number, Field(ge=-20, le=20)] | None = None
    temperature_anomaly: Annotated[Number, Field(ge=-30, le=30)] | None = None
    sar_change: Annotated[Number, Field(ge=-60, le=60)] | None = None
    forest_cover: Fraction | None = None
    fire_anomaly: Annotated[Number, Field(ge=-20, le=20)] | None = None
    valid_fraction: Fraction

    @field_validator("date")
    @classmethod
    def valid_date(cls, value):
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            raise ValueError("Use a calendar date formatted YYYY-MM-DD") from None
        if parsed.isoformat() != value:
            raise ValueError("Use a calendar date formatted YYYY-MM-DD")
        return value


class ImportRegion(StrictModel):
    name: str = Field(min_length=2, max_length=120)
    country: str = Field(default="Unspecified", min_length=2, max_length=100)
    source: str = Field(default="User-provided prepared observations", min_length=2, max_length=500)
    notes: str = Field(default="", max_length=2000)
    synthetic: bool = Field(default=False, strict=True)
    geometry: Geometry
    observations: list[Observation] = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def validate_time(self):
        dates = [item.date for item in self.observations]
        if dates != sorted(set(dates)):
            raise ValueError("Observations must have unique dates in strictly increasing order")
        months = [day[:7] for day in dates]
        if len(set(months)) != len(months):
            raise ValueError("Provide at most one prepared observation per calendar month")
        return self


class AnalysisRequest(StrictModel):
    region_id: str = Field(min_length=1, max_length=80)


class ScenarioRequest(AnalysisRequest):
    fire_reduction: Percent = 0
    pressure_reduction: Percent = 0
    restoration: Percent = 0
    climate_stress: Percent = 0


def geometry_metadata(geometry: Geometry) -> tuple[list[float], float]:
    ring = geometry.coordinates[0]
    center = [sum(point[index] for point in ring[:-1]) / (len(ring) - 1) for index in (0, 1)]
    # Local equal-latitude planar approximation, disclosed in imported provenance.
    scale_x = 111.32 * math.cos(math.radians(center[1]))
    scale_y = 111.32
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:]))) / 2
    return [round(item, 5) for item in center], round(area * scale_x * scale_y, 2)
