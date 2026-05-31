"""The single canonical model-input object (L0)."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import AwareDatetime, BaseModel, ConfigDict

# Single source of truth for the snapshot feature contract version. Bumped when the
# set/meaning of feature keys changes. The training store stamps the same value.
SNAPSHOT_SCHEMA_VERSION = "1.0.0"


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
