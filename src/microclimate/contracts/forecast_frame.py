"""NWP forecast DataFrame schema (L0).

One row per lead hour; no _present masks — NWP fields are complete.
Distinct from ForecastDocument/ForecastStep (published JSON models in forecast.py).
"""

from __future__ import annotations

import pandera.pandas as pa

FORECAST_FRAME = pa.DataFrameSchema(
    {
        "issue_time": pa.Column("datetime64[ns, UTC]"),
        "lead_hour": pa.Column(int, pa.Check.in_range(1, 48)),  # type: ignore[reportUnknownMemberType]
        "valid_time": pa.Column("datetime64[ns, UTC]"),
        "temp_c": pa.Column(float, nullable=False),
        "dewpoint_c": pa.Column(float, nullable=False),
        "surface_pressure_hpa": pa.Column(float, nullable=False),
        "precip_mm": pa.Column(float, nullable=False),
        "cloud_cover_fraction": pa.Column(float, nullable=False),
        "solar_radiation_wm2": pa.Column(float, nullable=False),
        "wind_speed_ms": pa.Column(float, nullable=False),
        "wind_dir_deg": pa.Column(float, nullable=False),
    },
    strict=True,
    coerce=True,
)
