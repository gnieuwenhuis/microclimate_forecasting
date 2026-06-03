"""DeploymentConfig and nested local models (L1, ADR-0006)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StationRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    station_id: str
    connector_key: str
    lat: float
    lon: float
    elevation_m: float | None = None


class NwpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str
    live_connector: str
    historical_connector: str
    sampling: str


class FeatureGroupSwitches(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nwp: bool
    observations: bool


class LabelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precip_occurrence_threshold_mm: float = Field(ge=0.0)


class SeedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    start: str


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: SeedConfig
    holdout_months: int = Field(ge=1)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_json: str


class DeploymentConfig(BaseModel):
    """One fully-specified deployment. Everything is namespaced by deployment_id."""

    model_config = ConfigDict(extra="forbid")

    deployment_id: str
    target: StationRef
    neighbors: list[StationRef]
    enabled_sources: list[str]
    nwp: NwpConfig
    horizon_hours: int = Field(default=48, ge=1, le=48)  # HRDPS lead-time ceiling (ADR-0007)
    min_horizon_hours: int = Field(
        default=12, ge=1
    )  # min available leads to still publish; else retry
    lag_hours: int = Field(ge=0)
    feature_groups: FeatureGroupSwitches
    label: LabelConfig
    training: TrainingConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _min_horizon_within_horizon(self) -> DeploymentConfig:
        if self.min_horizon_hours > self.horizon_hours:
            raise ValueError(
                f"min_horizon_hours ({self.min_horizon_hours}) must be <= "
                f"horizon_hours ({self.horizon_hours})"
            )
        return self
