from __future__ import annotations

import pandas as pd
import pytest

from microclimate.contracts.observation import OBSERVATION_FRAME


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["9835"],
            "timestamp": pd.to_datetime(["2026-05-30T00:00:00Z"]),
            "temp_c": [12.3],
            "temp_c_present": [True],
            "dewpoint_c": [5.0],
            "dewpoint_c_present": [True],
            "surface_pressure_hpa": [1013.25],
            "surface_pressure_hpa_present": [True],
            "precip_mm": [0.0],
            "precip_mm_present": [True],
            "cloud_cover_fraction": [0.3],
            "cloud_cover_fraction_present": [True],
            "solar_radiation_wm2": [200.0],
            "solar_radiation_wm2_present": [True],
            "wind_speed_ms": [4.5],
            "wind_speed_ms_present": [True],
            "wind_dir_deg": [270.0],
            "wind_dir_deg_present": [True],
        }
    )


def test_valid_observation_frame_passes() -> None:
    OBSERVATION_FRAME.validate(_valid_frame())


def test_missing_mask_column_fails() -> None:
    frame = _valid_frame().drop(columns=["temp_c_present"])
    with pytest.raises(Exception):  # noqa: B017
        OBSERVATION_FRAME.validate(frame)


def test_extra_column_fails() -> None:
    frame = _valid_frame()
    frame["humidity"] = [50.0]
    with pytest.raises(Exception):  # noqa: B017
        OBSERVATION_FRAME.validate(frame)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("precip_mm", -1.0),
        ("cloud_cover_fraction", 1.5),
        ("solar_radiation_wm2", -10.0),
        ("wind_speed_ms", -3.0),
        ("wind_dir_deg", 400.0),
        ("surface_pressure_hpa", 0.0),
    ],
)
def test_out_of_range_present_value_fails(column: str, bad_value: float) -> None:
    frame = _valid_frame().copy()
    frame[column] = [bad_value]
    with pytest.raises(Exception):  # noqa: B017
        OBSERVATION_FRAME.validate(frame)


def test_masked_absent_value_passes_range_checks() -> None:
    # A masked-absent reading (NaN value, present=False) must NOT trip the range checks.
    frame = _valid_frame().copy()
    frame["cloud_cover_fraction"] = [float("nan")]
    frame["cloud_cover_fraction_present"] = [False]
    frame["surface_pressure_hpa"] = [float("nan")]
    frame["surface_pressure_hpa_present"] = [False]
    OBSERVATION_FRAME.validate(frame)
