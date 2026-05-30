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
        # temp_c / dewpoint_c are intentionally unbounded — air temperature and dew point
        # have no natural physical floor/ceiling, so a hard range would false-reject
        # legitimately extreme readings. The fields below have genuine physical bounds; the
        # pressure range (800–1100 hPa) also catches a Pa→hPa unit-conversion bug.
        "temp_c": pa.Column(float, nullable=False),
        "dewpoint_c": pa.Column(float, nullable=False),
        "surface_pressure_hpa": pa.Column(float, pa.Check.in_range(800, 1100), nullable=False),  # type: ignore[reportUnknownMemberType]
        "precip_mm": pa.Column(float, pa.Check.ge(0), nullable=False),  # type: ignore[reportUnknownMemberType]
        "cloud_cover_fraction": pa.Column(float, pa.Check.in_range(0, 1), nullable=False),  # type: ignore[reportUnknownMemberType]
        "solar_radiation_wm2": pa.Column(float, pa.Check.ge(0), nullable=False),  # type: ignore[reportUnknownMemberType]
        "wind_speed_ms": pa.Column(float, pa.Check.ge(0), nullable=False),  # type: ignore[reportUnknownMemberType]
        "wind_dir_deg": pa.Column(float, pa.Check.in_range(0, 360), nullable=False),  # type: ignore[reportUnknownMemberType]
    },
    strict=True,
    coerce=True,
)
