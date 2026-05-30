"""The published forecast document — the only thing thin clients read (L0)."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class ForecastStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lead_hour: int = Field(ge=1, le=48)
    valid_time: AwareDatetime
    temp_c: float
    pop: float = Field(ge=0.0, le=1.0)


class ForecastDocument(BaseModel):
    """Derived predictions only — never raw observations (ADR-0009)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    deployment_id: str
    issue_time: AwareDatetime
    last_updated: AwareDatetime
    status: Literal["ok", "stale", "degraded"]
    model_versions: dict[Literal["temp", "pop"], str]
    attribution: list[str] = Field(min_length=1)
    series: list[ForecastStep]
