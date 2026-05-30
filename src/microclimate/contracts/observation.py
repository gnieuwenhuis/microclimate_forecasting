"""Standardized observation frame + single-record model (L0)."""

from __future__ import annotations

import pandera.pandas as pa
from pydantic import AwareDatetime, BaseModel, ConfigDict

# Every observation source must emit exactly these columns. Each measurement is paired
# with a `<field>_present` mask so a down feed degrades to imputed+masked, never crashes.
OBSERVATION_FRAME = pa.DataFrameSchema(
    {
        "station_id": pa.Column(str),
        "timestamp": pa.Column("datetime64[ns, UTC]"),
        "temp_c": pa.Column(float, nullable=True),
        "temp_c_present": pa.Column(bool),
        "precip_mm": pa.Column(float, nullable=True),
        "precip_mm_present": pa.Column(bool),
    },
    strict=True,
    coerce=True,
)


class ObservationRecord(BaseModel):
    """One row of OBSERVATION_FRAME."""

    model_config = ConfigDict(extra="forbid")

    station_id: str
    timestamp: AwareDatetime
    temp_c: float | None
    temp_c_present: bool
    precip_mm: float | None
    precip_mm_present: bool
