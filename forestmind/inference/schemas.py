"""Portable inference contracts; intentionally independent of Torch and the API."""

from __future__ import annotations

from datetime import date, datetime
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


HEALTH_NAMES = ("Healthy", "Stressed", "Degraded", "Severely Degraded", "Cleared")
DRIVER_NAMES = ("Agriculture", "Cattle Pasture", "Mining", "Logging", "Road Construction",
                "Wildfire", "Drought", "Urban Expansion")
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Fraction = Annotated[Number, Field(ge=0, le=1)]
Score = Annotated[Number, Field(ge=0, le=100)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


def calendar_date(value: str) -> str:
    try:
        if date.fromisoformat(value).isoformat() == value:
            return value
    except ValueError:
        pass
    raise ValueError("Use a calendar date formatted YYYY-MM-DD")


def basename(value: str) -> str:
    if value in {".", ".."} or any(char in value for char in ("/", "\\", ":", "\x00")):
        raise ValueError("Only a file or dataset basename is allowed; do not include a path")
    return value


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CheckpointInfo(ReportModel):
    name: str = Field(min_length=1, max_length=255)
    sha256: Digest
    epoch: Annotated[int, Field(strict=True, ge=0)] | None
    backbone: Literal["custom_vit", "prithvi_proxy", "terramind_proxy", "prithvi", "terramind"]
    training_source: str = Field(min_length=1, max_length=255)
    trained_on_synthetic: Annotated[bool, Field(strict=True)] | None

    _basename = field_validator("name", "training_source")(basename)

    @model_validator(mode="after")
    def source_consistency(self):
        if self.training_source == "synthetic" and self.trained_on_synthetic is not True:
            raise ValueError("Synthetic training source must retain synthetic provenance")
        if self.training_source == "unknown" and self.trained_on_synthetic is not None:
            raise ValueError("Unknown training provenance must retain null synthetic status")
        return self


class InputInfo(ReportModel):
    name: str = Field(min_length=1, max_length=255)
    sha256: Digest
    synthetic: Annotated[bool, Field(strict=True)]
    timesteps: Annotated[int, Field(strict=True, ge=1, le=4096)]
    observed_timesteps: Annotated[int, Field(strict=True, ge=1, le=4096)] | None = None
    optical_timesteps: Annotated[int, Field(strict=True, ge=0, le=4096)] | None = None
    sar_timesteps: Annotated[int, Field(strict=True, ge=0, le=4096)] | None = None
    valid_optical_fraction: Fraction | None = None
    observed_start: str | None = None
    timestamp_basis: Literal["acquisition_dates", "day_offsets_anchored_to_declared_date"] | None = None
    normalization: Literal["checkpoint_statistics", "synthetic_defaults"] | None = None
    source_manifest_sha256: Digest | None = None

    _basename = field_validator("name")(basename)

    @field_validator("observed_start")
    @classmethod
    def start_date(cls, value):
        return calendar_date(value) if value is not None else None

    @model_validator(mode="after")
    def counts(self):
        for count in (self.observed_timesteps, self.optical_timesteps, self.sar_timesteps):
            if count is not None and count > self.timesteps:
                raise ValueError("Observation counts cannot exceed sequence timesteps")
        if not self.synthetic and self.normalization == "synthetic_defaults":
            raise ValueError("Real inputs require explicit checkpoint normalization")
        return self


class NamedScore(ReportModel):
    name: str
    score: Fraction


class HealthPrediction(ReportModel):
    state: Literal["Healthy", "Stressed", "Degraded", "Severely Degraded", "Cleared"]
    score: Score
    class_scores: list[NamedScore] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def canonical_classes(self):
        if tuple(item.name for item in self.class_scores) != HEALTH_NAMES:
            raise ValueError("Class scores must use all five canonical health names in model order")
        if not math.isclose(sum(item.score for item in self.class_scores), 1, abs_tol=1e-5):
            raise ValueError("Health class scores must sum to one")
        if self.state != max(self.class_scores, key=lambda item: item.score).name:
            raise ValueError("Health state must match the greatest class score")
        return self


class DriverPrediction(NamedScore):
    strength: Fraction


class TrajectoryPrediction(ReportModel):
    horizon_days: PositiveInt
    score: Score
    stddev: Annotated[Number, Field(ge=0, le=1000)] | None


class Predictions(ReportModel):
    health: HealthPrediction
    drivers: list[DriverPrediction] = Field(min_length=8, max_length=8)
    trajectory: list[TrajectoryPrediction] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def canonical_dimensions(self):
        if tuple(item.name for item in self.drivers) != DRIVER_NAMES:
            raise ValueError("Driver scores must use all eight canonical driver names in model order")
        horizons = [item.horizon_days for item in self.trajectory]
        if horizons != sorted(set(horizons)):
            raise ValueError("Trajectory horizons must be unique and strictly increasing")
        return self


class InferenceReport(ReportModel):
    schema_version: Literal[1]
    kind: Literal["forestmind_inference"]
    region_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    observed_through: str
    created_at: str
    checkpoint: CheckpointInfo
    input: InputInfo
    predictions: Predictions
    calibration: Literal["uncalibrated"]
    caveats: list[Annotated[str, Field(min_length=1, max_length=2000)]] = Field(min_length=1, max_length=32)

    _calendar_date = field_validator("observed_through")(calendar_date)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1")
        return value

    @field_validator("created_at")
    @classmethod
    def aware_timestamp(cls, value):
        try:
            parsed = datetime.fromisoformat(value)
            if "T" in value and parsed.utcoffset() is not None:
                return value
        except ValueError:
            pass
        raise ValueError("created_at must be an ISO timestamp with an explicit timezone")

    @model_validator(mode="after")
    def timing(self):
        if self.input.observed_start and self.input.observed_start > self.observed_through:
            raise ValueError("Input start date cannot follow observed_through")
        if self.observed_through > datetime.fromisoformat(self.created_at).date().isoformat():
            raise ValueError("Observed data cannot follow report creation")
        return self
