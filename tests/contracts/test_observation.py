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
            "precip_mm": [0.0],
            "precip_mm_present": [True],
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
