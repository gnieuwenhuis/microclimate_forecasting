"""Shared xarray.Dataset → FORECAST_FRAME normalisation core for HRDPS connectors.

This module encodes **HRDPS v1 unit conventions** and is shared by both HRDPS
connectors (hrdps_datamart, issue-5; hrdps_caspar/CaSPAr, issue-6).  The
seam — GRIB2 download + cfgrib decode — lives in each connector; this core is
pure xarray/numpy with no I/O and no cfgrib dependency.

HRDPS v1 unit assumptions (must be revisited if the upstream model changes):
    temp_c                 K    → °C  (subtract 273.15)
    dewpoint_c             K    → °C  (subtract 273.15)
    surface_pressure_hpa   Pa   → hPa (divide by 100)
    precip_mm              accumulated kg/m² from run-start → per-hour mm
                           (de-accumulate: diff vs previous hour; clamp ≥ 0)
    cloud_cover_fraction   %    → fraction (divide by 100; clamp to [0, 1])
    solar_radiation_wm2    accumulated J/m² from run-start → mean W/m² over the hour
                           (de-accumulate: diff vs previous hour, ÷3600 s; clamp ≥ 0)
    wind_speed_ms          m/s  (pass-through)
    wind_dir_deg           degrees 0–360 (pass-through)

Dataset contract (the seam normalises raw GRIB2 to this):
    Coordinates:
        lead_hour  — 1-D integer, ascending, covering every requested hour AND
                     the hour immediately before the smallest requested hour
                     (needed for precip de-accumulation; typically 0 is present).
        latitude   — 2-D float over dims (y, x); curvilinear, matching HRDPS.
        longitude  — 2-D float over dims (y, x); may be 0–360 or −180..180 —
                     normalised before distance comparison.
    Data variables:
        Named by whatever the connector's seam produces (canonical names for Datamart;
        CaSPAr netCDF names for CaSPAr); referenced through the caller-supplied
        ``var_map`` (canonical name → dataset variable name).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from microclimate.contracts.forecast_frame import FORECAST_FRAME

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_KELVIN_OFFSET: float = 273.15
_PA_TO_HPA: float = 100.0
_PCT_TO_FRACTION: float = 100.0
_SECONDS_PER_HOUR: float = 3600.0

# Canonical variable names that every var_map must cover (one entry per
# FORECAST_FRAME physical column).  Mirrors the _PHYS_VARS pattern used in
# the envcanada connector.
_CANONICAL_VARS: tuple[str, ...] = (
    "temp_c",
    "dewpoint_c",
    "surface_pressure_hpa",
    "precip_mm",
    "cloud_cover_fraction",
    "solar_radiation_wm2",
    "wind_speed_ms",
    "wind_dir_deg",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _norm_lon(lon: float | np.ndarray) -> float | np.ndarray:  # type: ignore[reportUnknownMemberType]
    """Normalise longitude(s) to the half-open interval [−180, 180).

    Accepts a scalar float or a numpy array; returns the same type.
    """
    return (lon + 180.0) % 360.0 - 180.0  # type: ignore[reportUnknownMemberType]


def _nearest_cell(
    ds: xr.Dataset,
    lat: float,
    lon: float,
) -> tuple[int, int]:
    """Return (iy, ix) of the grid cell nearest to (lat, lon).

    Uses squared Euclidean distance in (latitude, longitude) space after
    normalising both the dataset longitudes and the target longitude to
    [−180, 180).  No scipy dependency — pure numpy argmin.
    """
    lat_arr: np.ndarray = ds.coords["latitude"].values  # type: ignore[reportUnknownMemberType]
    lon_arr_raw: np.ndarray = ds.coords["longitude"].values  # type: ignore[reportUnknownMemberType]

    # I-2: Enforce the 2-D curvilinear grid contract.  A 1-D coordinate would
    # let np.unravel_index succeed (shape is a 1-tuple) but silently produce a
    # wrong (iy, ix) unpack; catching it here gives a clear message instead.
    if lat_arr.ndim != 2 or lon_arr_raw.ndim != 2:
        raise ValueError(
            "nwp_core expects 2-D curvilinear latitude/longitude (dims y,x); "
            f"got ndim={lat_arr.ndim} for latitude and ndim={lon_arr_raw.ndim} "
            "for longitude."
        )
    lon_arr: np.ndarray = _norm_lon(lon_arr_raw)  # type: ignore[reportUnknownMemberType]
    lon_target = float(_norm_lon(lon))

    dist2: np.ndarray = (lat_arr - lat) ** 2 + (lon_arr - lon_target) ** 2  # type: ignore[reportUnknownMemberType]
    flat_idx = int(np.argmin(dist2))
    iy, ix = np.unravel_index(flat_idx, dist2.shape)
    return int(iy), int(ix)


def _check_lead_hours_present(
    ds: xr.Dataset,
    lead_hours: Sequence[int],
) -> None:
    """Raise ValueError if any requested hour (or its predecessor) is absent.

    De-accumulation requires ``h−1`` to be in the dataset for every requested
    ``h``, so we validate both the requested hours and their predecessors.
    """
    available: set[int] = set(int(v) for v in ds.coords["lead_hour"].values)
    for h in lead_hours:
        if h < 1:
            raise ValueError(
                f"lead_hour={h} is out of range: FORECAST_FRAME requires lead_hour ≥ 1."
            )
        if h not in available:
            raise ValueError(
                f"Requested lead_hour={h} is not present in the dataset. "
                f"Available lead hours: {sorted(available)}."
            )
        prev = h - 1
        if prev not in available:
            raise ValueError(
                f"Precip de-accumulation for lead_hour={h} requires lead_hour={prev} "
                f"to be present in the dataset, but it is absent. "
                f"Available lead hours: {sorted(available)}."
            )


def _sample(da: xr.DataArray, lead: int, iy: int, ix: int) -> float:
    """Extract a scalar float at (lead_hour=lead, y=iy, x=ix)."""
    return float(da.sel(lead_hour=lead).isel(y=iy, x=ix).values)  # type: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dataset_to_forecast_frame(
    ds: xr.Dataset,
    var_map: Mapping[str, str],
    *,
    issue_time: datetime,
    lat: float,
    lon: float,
    lead_hours: Sequence[int],
) -> pd.DataFrame:
    """Normalise an HRDPS xarray.Dataset to a FORECAST_FRAME-valid DataFrame.

    Args:
        ds:          Dataset conforming to the HRDPS Dataset contract (see module
                     docstring).  Must contain a 1-D ``lead_hour`` coordinate and
                     2-D ``latitude``/``longitude`` coordinates over dims (y, x).
        var_map:     Mapping from each of the 8 canonical column names
                     (``temp_c``, ``dewpoint_c``, …) to the dataset variable that
                     supplies it (the seam's variable name).
        issue_time:  UTC-aware model run initialisation time.
        lat:         Target latitude (decimal degrees).
        lon:         Target longitude (decimal degrees; either convention).
        lead_hours:  Ordered sequence of forecast lead hours to extract (1–48,
                     each must be present in ``ds``).

    Returns:
        DataFrame validated against ``FORECAST_FRAME`` (``coerce=True``), with
        one row per requested lead hour.

    Raises:
        ValueError: If ``var_map`` is missing a canonical key, if a mapped
                    dataset variable does not exist, or if any requested
                    ``lead_hour`` (or the hour preceding it, needed for precip
                    de-accumulation) is absent from the dataset.
    """
    # -----------------------------------------------------------------------
    # 1. Validate var_map completeness and dataset variable existence (I-1)
    # -----------------------------------------------------------------------
    missing_canonical = [k for k in _CANONICAL_VARS if k not in var_map]
    if missing_canonical:
        raise ValueError(
            f"var_map is missing the following canonical key(s): {missing_canonical}. "
            f"All of {list(_CANONICAL_VARS)} must be present."
        )
    missing_ds_vars = [
        f"{canon!r} → {var_map[canon]!r}" for canon in _CANONICAL_VARS if var_map[canon] not in ds
    ]
    if missing_ds_vars:
        raise ValueError(
            f"var_map references dataset variable(s) that do not exist in ds: {missing_ds_vars}."
        )

    # -----------------------------------------------------------------------
    # 2. Validate all requested lead hours (and their predecessors) are present
    # -----------------------------------------------------------------------
    _check_lead_hours_present(ds, lead_hours)

    # -----------------------------------------------------------------------
    # 3. Find the nearest grid cell
    # -----------------------------------------------------------------------
    iy, ix = _nearest_cell(ds, lat, lon)

    # -----------------------------------------------------------------------
    # 4. Cache all DataArrays once (M-2) and hoist loop-invariant values (M-1)
    # -----------------------------------------------------------------------
    temp_da = ds[var_map["temp_c"]]
    dew_da = ds[var_map["dewpoint_c"]]
    press_da = ds[var_map["surface_pressure_hpa"]]
    precip_da = ds[var_map["precip_mm"]]
    cloud_da = ds[var_map["cloud_cover_fraction"]]
    solar_da = ds[var_map["solar_radiation_wm2"]]
    wspd_da = ds[var_map["wind_speed_ms"]]
    wdir_da = ds[var_map["wind_dir_deg"]]

    # M-1: issue_utc is loop-invariant; compute it once before the loop.
    issue_utc = issue_time if issue_time.tzinfo is not None else issue_time.replace(tzinfo=UTC)
    issue_ts = pd.Timestamp(issue_utc)

    # -----------------------------------------------------------------------
    # 5. Extract variables at the nearest cell for each lead hour
    # -----------------------------------------------------------------------
    rows: list[dict[str, object]] = []
    for h in lead_hours:
        # -- Non-precip variables: sample → convert ---------------------
        temp_k = _sample(temp_da, h, iy, ix)
        dew_k = _sample(dew_da, h, iy, ix)
        press_pa = _sample(press_da, h, iy, ix)
        cloud_pct = _sample(cloud_da, h, iy, ix)
        wspd = _sample(wspd_da, h, iy, ix)
        wdir = _sample(wdir_da, h, iy, ix)

        # -- Unit conversions ----------------------------------------
        temp_c = temp_k - _KELVIN_OFFSET
        dew_c = dew_k - _KELVIN_OFFSET
        press_hpa = press_pa / _PA_TO_HPA
        cloud_frac = max(0.0, min(1.0, cloud_pct / _PCT_TO_FRACTION))

        # -- Precip: de-accumulate acc(h) − acc(h-1), clamp ≥ 0 ----
        acc_h = _sample(precip_da, h, iy, ix)
        acc_prev = _sample(precip_da, h - 1, iy, ix)
        precip_mm = max(0.0, acc_h - acc_prev)

        # -- Solar: HRDPS DSWRF is accumulated J/m² from run start. De-accumulate
        #    like precip, then ÷Δt → mean W/m² over the hour. Clamp ≥ 0. (ADR-0014)
        solar_acc_h = _sample(solar_da, h, iy, ix)
        solar_acc_prev = _sample(solar_da, h - 1, iy, ix)
        solar_wm2 = max(0.0, (solar_acc_h - solar_acc_prev) / _SECONDS_PER_HOUR)

        valid_time = pd.Timestamp(issue_utc + timedelta(hours=h))

        rows.append(
            {
                "issue_time": issue_ts,
                "lead_hour": h,
                "valid_time": valid_time,
                "temp_c": temp_c,
                "dewpoint_c": dew_c,
                "surface_pressure_hpa": press_hpa,
                "precip_mm": precip_mm,
                "cloud_cover_fraction": cloud_frac,
                "solar_radiation_wm2": solar_wm2,
                "wind_speed_ms": wspd,
                "wind_dir_deg": wdir,
            }
        )

    # -----------------------------------------------------------------------
    # 6. Build DataFrame and validate against FORECAST_FRAME
    # -----------------------------------------------------------------------
    df: pd.DataFrame = pd.DataFrame(rows)  # type: ignore[reportUnknownMemberType]
    return FORECAST_FRAME.validate(df)  # type: ignore[reportUnknownMemberType]
