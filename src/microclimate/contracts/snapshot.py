"""The single canonical model-input object (L0)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import AwareDatetime, BaseModel, ConfigDict, field_validator

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

    @field_validator("nwp_features", "observation_features", "static_features", mode="before")
    @classmethod
    def _null_to_nan(cls, value: Any) -> Any:
        """Round-trip missing values as NaN.

        Absent observation features (e.g. ECCC never reports cloud/solar) and an unknown
        static elevation are legitimately NaN. Pydantic's default JSON serialization writes
        NaN as ``null``; on read it returns ``None``, which the float-Mapping fields reject.
        Coerce ``None`` back to NaN so serialize->deserialize is lossless (the masks, not the
        sentinel value, carry presence).
        """
        if isinstance(value, Mapping):
            m = cast("Mapping[str, object]", value)
            return {k: (float("nan") if v is None else v) for k, v in m.items()}
        return value
