"""Tests for the shared xarray→FORECAST_FRAME NWP normalisation core.

All tests are hermetic (pure xarray/numpy, no network, no cfgrib).
The ``build_hrdps_dataset`` fixture is shared in conftest.py and will also be
used by Task-B (hrdps_geomet / hrdps_caspar) connector tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from microclimate.connectors.nwp_core import dataset_to_forecast_frame
from microclimate.contracts.forecast_frame import FORECAST_FRAME

from .conftest import VAR_MAP, build_hrdps_dataset

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_ISSUE_TIME = datetime(2026, 5, 1, 0, 0, 0, tzinfo=UTC)
_TARGET_LAT = 51.0
_TARGET_LON = -114.0  # near the "target" cell in the synthetic dataset


# ---------------------------------------------------------------------------
# 1. Schema conformance + canonical value pinning
# ---------------------------------------------------------------------------


def test_schema_conformance_and_canonical_values() -> None:
    """dataset_to_forecast_frame emits a FORECAST_FRAME-valid DataFrame with pinned values."""
    ds = build_hrdps_dataset()

    df = dataset_to_forecast_frame(
        ds,
        VAR_MAP,
        issue_time=_ISSUE_TIME,
        lat=_TARGET_LAT,
        lon=_TARGET_LON,
        lead_hours=[1, 2, 3],
    )

    # Must not raise
    FORECAST_FRAME.validate(df)

    assert len(df) == 3
    assert list(df["lead_hour"]) == [1, 2, 3]

    row1 = df[df["lead_hour"] == 1].iloc[0]
    row2 = df[df["lead_hour"] == 2].iloc[0]
    row3 = df[df["lead_hour"] == 3].iloc[0]

    # temp: 288.15 K → 15.0 °C
    assert row1["temp_c"] == pytest.approx(15.0)  # type: ignore[reportUnknownMemberType]
    # pressure: 90000 Pa → 900.0 hPa
    assert row1["surface_pressure_hpa"] == pytest.approx(900.0)  # type: ignore[reportUnknownMemberType]
    # cloud: 50 % → 0.5
    assert row1["cloud_cover_fraction"] == pytest.approx(0.5)  # type: ignore[reportUnknownMemberType]
    # wind dir: 270 deg (pass-through)
    assert row1["wind_dir_deg"] == pytest.approx(270.0)  # type: ignore[reportUnknownMemberType]
    # wind speed: 5.0 m/s (pass-through)
    assert row1["wind_speed_ms"] == pytest.approx(5.0)  # type: ignore[reportUnknownMemberType]
    # solar: 300 W/m² (pass-through)
    assert row1["solar_radiation_wm2"] == pytest.approx(300.0)  # type: ignore[reportUnknownMemberType]

    # valid_time = issue_time + lead_hour
    for _row, lh in zip([row1, row2, row3], [1, 2, 3], strict=True):
        expected_vt = pd.Timestamp(_ISSUE_TIME + timedelta(hours=lh))
        assert _row["valid_time"] == expected_vt  # type: ignore[reportUnknownMemberType]

    # issue_time column is constant and equals what we passed in
    assert (df["issue_time"] == pd.Timestamp(_ISSUE_TIME)).all()


# ---------------------------------------------------------------------------
# 2. Precip de-accumulation
# ---------------------------------------------------------------------------


def test_precip_deaccumulation_values() -> None:
    """Accumulated precip [0, 0.5, 2.0, 2.0] → per-hour [0.5, 1.5, 0.0]."""
    ds = build_hrdps_dataset()

    df = dataset_to_forecast_frame(
        ds,
        VAR_MAP,
        issue_time=_ISSUE_TIME,
        lat=_TARGET_LAT,
        lon=_TARGET_LON,
        lead_hours=[1, 2, 3],
    )

    assert df[df["lead_hour"] == 1]["precip_mm"].iloc[0] == pytest.approx(0.5)  # type: ignore[reportUnknownMemberType]
    assert df[df["lead_hour"] == 2]["precip_mm"].iloc[0] == pytest.approx(1.5)  # type: ignore[reportUnknownMemberType]
    assert df[df["lead_hour"] == 3]["precip_mm"].iloc[0] == pytest.approx(0.0)  # type: ignore[reportUnknownMemberType]


def test_precip_negative_diff_clamped_to_zero() -> None:
    """Accumulation decrease between hours (reset/rounding) → clamps to 0.0, not negative."""
    ds = build_hrdps_dataset()

    # Override the accumulated precip at the target cell with a DECREASE at hour 2
    # so acc[2] < acc[1].  All other cells are unchanged.
    # Target cell is (iy=0, ix=0) per build_hrdps_dataset.
    precip_var = ds[VAR_MAP["precip_mm"]].copy()
    # lead_hour dim: [0, 1, 2, 3].  Set acc at lead=2 to 0.3 (< acc at lead=1 = 0.5)
    precip_data = precip_var.values.copy()  # shape: (lead_hour, y, x)
    lead_idx_2 = list(ds.coords["lead_hour"].values).index(2)
    precip_data[lead_idx_2, 0, 0] = 0.3  # decrease at target cell
    ds_modified = ds.copy()
    ds_modified[VAR_MAP["precip_mm"]] = xr.DataArray(
        precip_data,
        dims=precip_var.dims,
        coords=precip_var.coords,  # type: ignore[reportUnknownMemberType]
    )

    df = dataset_to_forecast_frame(
        ds_modified,
        VAR_MAP,
        issue_time=_ISSUE_TIME,
        lat=_TARGET_LAT,
        lon=_TARGET_LON,
        lead_hours=[1, 2],
    )

    # lead 1: acc[1] - acc[0] = 0.5 - 0.0 = 0.5
    assert df[df["lead_hour"] == 1]["precip_mm"].iloc[0] == pytest.approx(0.5)  # type: ignore[reportUnknownMemberType]
    # lead 2: acc[2] - acc[1] = 0.3 - 0.5 = -0.2 → clamped to 0.0
    assert df[df["lead_hour"] == 2]["precip_mm"].iloc[0] == pytest.approx(0.0)  # type: ignore[reportUnknownMemberType]
    assert df[df["lead_hour"] == 2]["precip_mm"].iloc[0] >= 0.0


# ---------------------------------------------------------------------------
# 3. Nearest-cell selection
# ---------------------------------------------------------------------------


def test_nearest_cell_selects_correct_cell() -> None:
    """Target closest to cell (0,0) → temp from that cell, not the others."""
    ds = build_hrdps_dataset()

    df = dataset_to_forecast_frame(
        ds,
        VAR_MAP,
        issue_time=_ISSUE_TIME,
        lat=_TARGET_LAT,
        lon=_TARGET_LON,
        lead_hours=[1],
    )

    # Cell (iy=0, ix=0) has temp 288.15 K → 15.0 °C
    assert df["temp_c"].iloc[0] == pytest.approx(15.0)  # type: ignore[reportUnknownMemberType]


def test_nearest_cell_wrong_target_gets_different_temp() -> None:
    """Target near cell (1,1) returns that cell's distinct temperature."""
    ds = build_hrdps_dataset()

    # Cell (iy=1, ix=1) has lat=52.0, lon=-113.0 per build_hrdps_dataset
    df = dataset_to_forecast_frame(
        ds,
        VAR_MAP,
        issue_time=_ISSUE_TIME,
        lat=52.0,
        lon=-113.0,
        lead_hours=[1],
    )

    # Cell (1,1) has temp 293.15 K → 20.0 °C; MUST differ from 15.0
    assert df["temp_c"].iloc[0] == pytest.approx(20.0)  # type: ignore[reportUnknownMemberType]
    assert df["temp_c"].iloc[0] != pytest.approx(15.0)  # type: ignore[reportUnknownMemberType]


def test_nearest_cell_lon_convention_normalization() -> None:
    """Longitude convention mismatch (0–360 dataset vs −180..180 target) is normalized."""
    ds = build_hrdps_dataset()

    # Re-express dataset longitudes in 0–360 (add 360 to negative values)
    lon_data = ds.coords["longitude"].values.copy()
    lon_data_360 = np.where(lon_data < 0, lon_data + 360.0, lon_data)
    new_coords = {k: ds.coords[k] for k in ds.coords}
    new_coords["longitude"] = xr.DataArray(
        lon_data_360,
        dims=ds.coords["longitude"].dims,
    )
    ds_360 = ds.assign_coords(new_coords)  # type: ignore[reportUnknownMemberType]

    # Target lat=51.0, lon=-114.0 (negative / −180..180 convention)
    df = dataset_to_forecast_frame(
        ds_360,
        VAR_MAP,
        issue_time=_ISSUE_TIME,
        lat=_TARGET_LAT,
        lon=_TARGET_LON,  # −114.0
        lead_hours=[1],
    )

    # Should still pick cell (0,0) → temp 15.0 °C
    assert df["temp_c"].iloc[0] == pytest.approx(15.0)  # type: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# 4. Missing requested lead hour raises ValueError
# ---------------------------------------------------------------------------


def test_missing_lead_hour_raises_value_error() -> None:
    """Requesting a lead_hour not present in the dataset raises ValueError with clear message."""
    ds = build_hrdps_dataset()  # has lead_hours [0, 1, 2, 3]

    with pytest.raises(ValueError, match="lead_hour"):
        dataset_to_forecast_frame(
            ds,
            VAR_MAP,
            issue_time=_ISSUE_TIME,
            lat=_TARGET_LAT,
            lon=_TARGET_LON,
            lead_hours=[5],  # 5 not in [0, 1, 2, 3]
        )


def test_missing_prev_hour_for_precip_raises_value_error() -> None:
    """lead_hour=1 when lead_hour=0 absent raises ValueError (de-accumulation requires h-1)."""
    ds = build_hrdps_dataset()
    # Drop lead_hour=0 from the dataset
    ds_no_zero = ds.sel(lead_hour=[1, 2, 3])

    with pytest.raises(ValueError, match="lead_hour"):
        dataset_to_forecast_frame(
            ds_no_zero,
            VAR_MAP,
            issue_time=_ISSUE_TIME,
            lat=_TARGET_LAT,
            lon=_TARGET_LON,
            lead_hours=[1],  # needs lead_hour=0 for precip de-accum
        )
