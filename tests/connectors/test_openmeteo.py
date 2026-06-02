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


def test_request_routing_and_shared_params() -> None:
    from microclimate.connectors.sources.openmeteo import _build_request

    now = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)
    live_url, live_params = _build_request(
        datetime(2026, 6, 2, 6, 0, tzinfo=UTC), 49.70, -112.77, [1, 48], now=now
    )
    hist_url, hist_params = _build_request(
        datetime(2024, 6, 1, 0, 0, tzinfo=UTC), 49.70, -112.77, [1, 48], now=now
    )

    assert live_url.startswith("https://api.open-meteo.com/")
    assert hist_url.startswith("https://historical-forecast-api.open-meteo.com/")
    shared = (
        "latitude", "longitude", "models", "cell_selection",
        "wind_speed_unit", "timezone", "hourly",
    )
    for k in shared:
        assert live_params[k] == hist_params[k], k
    assert live_params["cell_selection"] == "land"
    assert live_params["models"] == "gem_hrdps_continental"
    assert hist_params["start_date"] == "2024-06-01" and hist_params["end_date"] == "2024-06-03"
    assert "start_date" not in live_params
