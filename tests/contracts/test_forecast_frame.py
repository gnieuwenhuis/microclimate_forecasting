"""Tests for the FORECAST_FRAME Pandera schema (L0)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from microclimate.contracts.forecast_frame import FORECAST_FRAME

_ISSUE = datetime(2026, 5, 30, 0, tzinfo=UTC)
_VALID_1 = datetime(2026, 5, 30, 1, tzinfo=UTC)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "issue_time": pd.to_datetime([_ISSUE]),
            "lead_hour": [1],
            "valid_time": pd.to_datetime([_VALID_1]),
            "temp_c": [12.0],
            "dewpoint_c": [5.0],
            "surface_pressure_hpa": [1013.25],
            "precip_mm": [0.0],
            "cloud_cover_fraction": [0.3],
            "solar_radiation_wm2": [200.0],
            "wind_speed_ms": [4.5],
            "wind_dir_deg": [270.0],
        }
    )


def test_valid_forecast_frame_passes() -> None:
    FORECAST_FRAME.validate(_valid_frame())


def test_lead_hour_out_of_range_fails() -> None:
    frame = _valid_frame().copy()
    frame["lead_hour"] = [0]
    with pytest.raises(Exception):  # noqa: B017
        FORECAST_FRAME.validate(frame)


def test_lead_hour_above_range_fails() -> None:
    frame = _valid_frame().copy()
    frame["lead_hour"] = [49]
    with pytest.raises(Exception):  # noqa: B017
        FORECAST_FRAME.validate(frame)


def test_extra_column_fails_strict() -> None:
    frame = _valid_frame()
    frame["humidity_pct"] = [75.0]
    with pytest.raises(Exception):  # noqa: B017
        FORECAST_FRAME.validate(frame)


def test_missing_column_fails() -> None:
    frame = _valid_frame().drop(columns=["wind_dir_deg"])
    with pytest.raises(Exception):  # noqa: B017
        FORECAST_FRAME.validate(frame)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("precip_mm", -1.0),
        ("cloud_cover_fraction", 1.5),
        ("solar_radiation_wm2", -10.0),
        ("wind_speed_ms", -3.0),
        ("wind_dir_deg", 400.0),
        ("surface_pressure_hpa", 90000.0),  # Pa not converted to hPa
        ("surface_pressure_hpa", 0.0),
    ],
)
def test_out_of_range_physical_value_fails(column: str, bad_value: float) -> None:
    frame = _valid_frame().copy()
    frame[column] = [bad_value]
    with pytest.raises(Exception):  # noqa: B017
        FORECAST_FRAME.validate(frame)
