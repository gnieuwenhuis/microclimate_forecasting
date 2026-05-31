"""Unit tests for the pure helpers behind build_snapshot."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pandas as pd

from microclimate.features.snapshot_builder import (
    _flatten_forecast,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
    _temporal_features,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
)

_PHYS = (
    "temp_c",
    "dewpoint_c",
    "surface_pressure_hpa",
    "precip_mm",
    "cloud_cover_fraction",
    "solar_radiation_wm2",
    "wind_speed_ms",
    "wind_dir_deg",
)


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
