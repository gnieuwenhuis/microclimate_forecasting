"""HRDPS via Open-Meteo — single pure-HTTP+JSON NWP source for live + historical (ADR-0019).

Live  → api.open-meteo.com/v1/forecast (current run, full leads).
Hist  → historical-forecast-api.open-meteo.com/v1/forecast (deep, stitched short-lead).
Both return {"hourly": {"time": [...], "<var>": [...]}}; one parser serves both.
No nwp_core, no cfgrib/xarray — precip/solar arrive already de-accumulated.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd

from microclimate.connectors.base import ForecastUnavailable
from microclimate.contracts.forecast_frame import FORECAST_FRAME
from microclimate.contracts.physical_vars import PHYSICAL_VARS

# canonical column → Open-Meteo hourly variable name.
_OPENMETEO_VAR_MAP: dict[str, str] = {
    "temp_c": "temperature_2m",
    "dewpoint_c": "dew_point_2m",
    "surface_pressure_hpa": "surface_pressure",
    "precip_mm": "precipitation",
    "cloud_cover_fraction": "cloud_cover",
    "solar_radiation_wm2": "shortwave_radiation",
    "wind_speed_ms": "wind_speed_10m",
    "wind_dir_deg": "wind_direction_10m",
}
_PCT_TO_FRACTION: float = 100.0
# Open-Meteo emits naive ISO timestamps in UTC when timezone=GMT.
_OM_TIME_FMT: str = "%Y-%m-%dT%H:%M"

_LIVE_URL: str = "https://api.open-meteo.com/v1/forecast"
_HISTORICAL_URL: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_GEM_MODEL: str = "gem_hrdps_continental"
_HOURLY_CSV: str = ",".join(_OPENMETEO_VAR_MAP[c] for c in PHYSICAL_VARS)
# Use the live endpoint when the run is recent enough to still be on it; else the deep archive.
_LIVE_CUTOFF = timedelta(days=2)


def _build_request(  # pyright: ignore[reportUnusedFunction]  # called by OpenMeteoSource (later task)
    issue_time: datetime,
    lat: float,
    lon: float,
    lead_hours: Sequence[int],
    *,
    now: datetime,
) -> tuple[str, dict[str, str | int | float]]:
    """Return (url, params). Recent issue_time → live endpoint; older → historical archive."""
    if issue_time.tzinfo is not None:
        issue_utc = issue_time.astimezone(UTC)
    else:
        issue_utc = issue_time.replace(tzinfo=UTC)
    params: dict[str, str | int | float] = {
        "latitude": lat,
        "longitude": lon,
        "models": _GEM_MODEL,
        "cell_selection": "land",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "hourly": _HOURLY_CSV,
    }
    if issue_utc >= now.astimezone(UTC) - _LIVE_CUTOFF:
        return _LIVE_URL, params
    end = (issue_utc + timedelta(hours=max(lead_hours))).date()
    params["start_date"] = issue_utc.date().isoformat()
    params["end_date"] = end.isoformat()
    return _HISTORICAL_URL, params


def _parse_hourly_to_forecast_frame(  # pyright: ignore[reportUnusedFunction]  # called by OpenMeteoSource (later task)
    payload: dict[str, object],
    *,
    issue_time: datetime,
    lead_hours: Sequence[int],
) -> pd.DataFrame:
    """Map an Open-Meteo `hourly` payload to a FORECAST_FRAME-valid DataFrame.

    Selects, for each requested lead h, the value at valid_time = issue_time + h.
    Raises ForecastUnavailable if any requested valid_time is absent or null.
    """
    if issue_time.tzinfo is not None:
        issue_utc = issue_time.astimezone(UTC)
    else:
        issue_utc = issue_time.replace(tzinfo=UTC)
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ForecastUnavailable("Open-Meteo payload missing 'hourly'/'time'.")
    times: list[str] = list(hourly["time"])  # type: ignore[arg-type]
    index_by_time: dict[str, int] = {t: i for i, t in enumerate(times)}

    missing_vars = [om for om in _OPENMETEO_VAR_MAP.values() if om not in hourly]
    if missing_vars:
        raise ForecastUnavailable(f"Open-Meteo payload missing variable(s): {missing_vars}.")

    rows: list[dict[str, object]] = []
    for h in lead_hours:
        valid = issue_utc + timedelta(hours=int(h))
        key = valid.strftime(_OM_TIME_FMT)
        idx = index_by_time.get(key)
        if idx is None:
            raise ForecastUnavailable(
                f"Open-Meteo series has no entry for valid_time {key} (lead_hour={h})."
            )
        row: dict[str, object] = {
            "issue_time": pd.Timestamp(issue_utc),
            "lead_hour": int(h),
            "valid_time": pd.Timestamp(valid),
        }
        for canon in PHYSICAL_VARS:
            raw = hourly[_OPENMETEO_VAR_MAP[canon]][idx]  # type: ignore[index]
            if raw is None:
                raise ForecastUnavailable(
                    f"Open-Meteo {canon!r} is null at valid_time {key} (lead_hour={h})."
                )
            value = float(raw)  # type: ignore[arg-type]  # raw is object from dict[str, object]
            if canon == "cloud_cover_fraction":
                value = max(0.0, min(1.0, value / _PCT_TO_FRACTION))
            elif canon in ("precip_mm", "solar_radiation_wm2"):
                value = max(0.0, value)
            row[canon] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    return FORECAST_FRAME.validate(df)
