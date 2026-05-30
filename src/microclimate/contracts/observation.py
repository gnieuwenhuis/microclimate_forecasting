"""Standardized observation frame + single-record model (L0)."""

from __future__ import annotations

import pandera.pandas as pa
from pydantic import AwareDatetime, BaseModel, ConfigDict

# Every observation source must emit exactly these columns. Each measurement is paired
# with a `<field>_present` mask so a down feed degrades to imputed+masked, never crashes.
# Eight physical variables; relative humidity is NOT stored (dewpoint is canonical).
OBSERVATION_FRAME = pa.DataFrameSchema(
    {
        "station_id": pa.Column(str),
        "timestamp": pa.Column("datetime64[ns, UTC]"),
        "temp_c": pa.Column(float, nullable=True),
        "temp_c_present": pa.Column(bool),
        "dewpoint_c": pa.Column(float, nullable=True),
        "dewpoint_c_present": pa.Column(bool),
        "surface_pressure_hpa": pa.Column(float, nullable=True),
        "surface_pressure_hpa_present": pa.Column(bool),
        "precip_mm": pa.Column(float, nullable=True),
        "precip_mm_present": pa.Column(bool),
        "cloud_cover_fraction": pa.Column(float, nullable=True),
        "cloud_cover_fraction_present": pa.Column(bool),
        "solar_radiation_wm2": pa.Column(float, nullable=True),
        "solar_radiation_wm2_present": pa.Column(bool),
        "wind_speed_ms": pa.Column(float, nullable=True),
        "wind_speed_ms_present": pa.Column(bool),
        "wind_dir_deg": pa.Column(float, nullable=True),
        "wind_dir_deg_present": pa.Column(bool),
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
    dewpoint_c: float | None
    dewpoint_c_present: bool
    surface_pressure_hpa: float | None
    surface_pressure_hpa_present: bool
    precip_mm: float | None
    precip_mm_present: bool
    cloud_cover_fraction: float | None
    cloud_cover_fraction_present: bool
    solar_radiation_wm2: float | None
    solar_radiation_wm2_present: bool
    wind_speed_ms: float | None
    wind_speed_ms_present: bool
    wind_dir_deg: float | None
    wind_dir_deg_present: bool
