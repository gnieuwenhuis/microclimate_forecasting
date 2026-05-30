"""The single canonical model-input object (L0)."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import AwareDatetime, BaseModel, ConfigDict


class FeatureSnapshot(BaseModel):
    """Inputs for one prediction at issue_time. Built only by features.build_snapshot."""

    model_config = ConfigDict(extra="forbid")

    deployment_id: str
    issue_time: AwareDatetime
    nwp_features: Mapping[str, float]
    observation_features: Mapping[str, float]
    observation_masks: Mapping[str, bool]
    static_features: Mapping[str, float]
    temporal_features: Mapping[str, float]
    lead_hours: tuple[int, ...]
    schema_version: str
