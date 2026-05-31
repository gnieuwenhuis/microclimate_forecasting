"""Unit tests for the pure helpers behind build_snapshot."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd

from microclimate.features.snapshot_builder import (
    _PHYSICAL_VARS,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
    _align_obs_to_lag_grid,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
    _flatten_forecast,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
    _temporal_features,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
)

_PHYS = _PHYSICAL_VARS


def _forecast_frame(lead_hours: list[int]) -> pd.DataFrame:
    """Minimal FORECAST_FRAME-shaped frame; var value encodes (var index + lead)."""
    rows: list[dict[str, object]] = []
    for lh in lead_hours:
        row: dict[str, object] = {"lead_hour": int(lh)}
        for i, var in enumerate(_PHYS):
            row[var] = float(i) + float(lh)
        rows.append(row)
    return pd.DataFrame(rows)


def test_temporal_features_keys_and_values() -> None:
    # 2026-01-01 06:00 UTC → day-of-year 1, hour 6.
    t0 = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
    feats = _temporal_features(t0)

    assert set(feats) == {"t0_hour_sin", "t0_hour_cos", "t0_doy_sin", "t0_doy_cos"}
    assert feats["t0_hour_sin"] == math.sin(2 * math.pi * 6 / 24.0)
    assert feats["t0_hour_cos"] == math.cos(2 * math.pi * 6 / 24.0)
    assert feats["t0_doy_sin"] == math.sin(2 * math.pi * 1 / 365.25)
    assert feats["t0_doy_cos"] == math.cos(2 * math.pi * 1 / 365.25)


def test_flatten_forecast_cardinality_and_keys() -> None:
    frame = _forecast_frame([1, 2, 3])
    flat = _flatten_forecast(frame)

    assert len(flat) == 8 * 3  # 8 vars x 3 leads
    assert flat["nwp_temp_c_h1"] == 0.0 + 1.0
    assert flat["nwp_temp_c_h3"] == 0.0 + 3.0
    assert flat["nwp_wind_dir_deg_h2"] == 7.0 + 2.0


_OBS_VALUES: dict[str, float] = {
    "temp_c": 15.0,
    "dewpoint_c": 5.0,
    "surface_pressure_hpa": 900.0,
    "precip_mm": 0.5,
    "cloud_cover_fraction": 0.5,
    "solar_radiation_wm2": 300.0,
    "wind_speed_ms": 5.0,
    "wind_dir_deg": 270.0,
}


def _obs_frame(
    station_id: str,
    timestamps: list[datetime],
    *,
    absent: set[tuple[int, str]] | None = None,
) -> pd.DataFrame:
    """OBSERVATION_FRAME-shaped frame. `absent` = {(row_index, var)} → value NaN, present False."""
    absent = absent or set()
    data: dict[str, list[object]] = {
        "station_id": [station_id] * len(timestamps),
        "timestamp": list(pd.to_datetime(timestamps, utc=True)),
    }
    for var in _PHYS:
        col_val: list[object] = []
        col_present: list[object] = []
        for idx in range(len(timestamps)):
            if (idx, var) in absent:
                col_val.append(float("nan"))
                col_present.append(False)
            else:
                col_val.append(_OBS_VALUES[var])
                col_present.append(True)
        data[var] = col_val
        data[f"{var}_present"] = col_present
    return pd.DataFrame(data)


_T0 = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)


def test_align_present_values_land_in_correct_lag() -> None:
    ts = [_T0, _T0 - timedelta(hours=1), _T0 - timedelta(hours=2)]
    feats, masks = _align_obs_to_lag_grid(_obs_frame("S1", ts), "S1", _T0, lag_hours=2)

    assert len(feats) == 8 * 3 and len(masks) == 8 * 3
    assert set(feats) == set(masks)
    assert feats["obs_S1_temp_c_lag0"] == 15.0
    assert masks["obs_S1_temp_c_lag0"] is True
    assert feats["obs_S1_surface_pressure_hpa_lag2"] == 900.0


def test_align_missing_hour_is_nan_and_masked() -> None:
    feats, masks = _align_obs_to_lag_grid(_obs_frame("S1", [_T0]), "S1", _T0, lag_hours=2)

    assert masks["obs_S1_temp_c_lag0"] is True
    assert masks["obs_S1_temp_c_lag1"] is False
    assert math.isnan(feats["obs_S1_temp_c_lag1"])
    assert masks["obs_S1_temp_c_lag2"] is False
    assert math.isnan(feats["obs_S1_temp_c_lag2"])


def test_align_present_false_is_nan_and_masked() -> None:
    ts = [_T0]
    feats, masks = _align_obs_to_lag_grid(
        _obs_frame("S1", ts, absent={(0, "surface_pressure_hpa")}), "S1", _T0, lag_hours=0
    )

    assert masks["obs_S1_temp_c_lag0"] is True
    assert masks["obs_S1_surface_pressure_hpa_lag0"] is False
    assert math.isnan(feats["obs_S1_surface_pressure_hpa_lag0"])


def test_align_filters_rows_after_t0() -> None:
    ts = [_T0 + timedelta(hours=1), _T0]
    feats, masks = _align_obs_to_lag_grid(_obs_frame("S1", ts), "S1", _T0, lag_hours=0)

    assert masks["obs_S1_temp_c_lag0"] is True
    assert feats["obs_S1_temp_c_lag0"] == 15.0


def test_align_none_frame_all_absent() -> None:
    feats, masks = _align_obs_to_lag_grid(None, "S1", _T0, lag_hours=2)

    assert len(feats) == 8 * 3
    assert all(m is False for m in masks.values())
    assert all(math.isnan(v) for v in feats.values())
