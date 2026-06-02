"""Hermetic tests for the Open-Meteo connector (no network)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from microclimate.contracts.forecast_frame import FORECAST_FRAME

_FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


def test_parse_historical_fixture_to_forecast_frame() -> None:
    from microclimate.connectors.sources.openmeteo import _parse_hourly_to_forecast_frame

    payload = _load("openmeteo_historical.json")
    # Fixture starts 2024-06-01T00:00; pick t0 there so leads 1..3 map to 01:00/02:00/03:00.
    t0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    df = _parse_hourly_to_forecast_frame(payload, issue_time=t0, lead_hours=[1, 2, 3])

    FORECAST_FRAME.validate(df)
    assert list(df["lead_hour"]) == [1, 2, 3]
    assert (df["cloud_cover_fraction"] >= 0).all() and (df["cloud_cover_fraction"] <= 1).all()
    for _, row in df.iterrows():
        assert row["valid_time"] == pd.Timestamp(t0) + pd.Timedelta(hours=int(row["lead_hour"]))


def test_parse_raises_when_lead_hour_absent() -> None:
    from microclimate.connectors.base import ForecastUnavailable
    from microclimate.connectors.sources.openmeteo import _parse_hourly_to_forecast_frame

    payload = _load("openmeteo_historical.json")
    far = datetime(2024, 6, 3, 23, 0, tzinfo=UTC)  # t0+1 falls outside the fixture window
    with pytest.raises(ForecastUnavailable):
        _parse_hourly_to_forecast_frame(payload, issue_time=far, lead_hours=[1])


def test_parse_raises_when_variable_is_null() -> None:
    from microclimate.connectors.base import ForecastUnavailable
    from microclimate.connectors.sources.openmeteo import _parse_hourly_to_forecast_frame

    payload = _load("openmeteo_historical.json")
    payload["hourly"]["temperature_2m"][1] = None  # type: ignore[index]
    t0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    with pytest.raises(ForecastUnavailable):
        _parse_hourly_to_forecast_frame(payload, issue_time=t0, lead_hours=[1])
