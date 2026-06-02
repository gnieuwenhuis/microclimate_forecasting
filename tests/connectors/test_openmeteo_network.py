"""Live Open-Meteo smoke tests (network-marked; deselected by default)."""
from __future__ import annotations

import json

import pytest

from microclimate.connectors.http import http_get

_SHARED = (
    "latitude=49.70&longitude=-112.77&models=gem_hrdps_continental"
    "&cell_selection=land&wind_speed_unit=ms&timezone=GMT"
    "&hourly=temperature_2m,dew_point_2m,surface_pressure,precipitation,"
    "cloud_cover,shortwave_radiation,wind_speed_10m,wind_direction_10m"
)


@pytest.mark.network
def test_live_forecast_returns_hourly() -> None:
    body = http_get(f"https://api.open-meteo.com/v1/forecast?{_SHARED}&forecast_days=3")
    hourly = json.loads(body)["hourly"]
    assert "temperature_2m" in hourly and len(hourly["time"]) >= 49


@pytest.mark.network
def test_historical_forecast_returns_deep_hourly() -> None:
    body = http_get(
        "https://historical-forecast-api.open-meteo.com/v1/forecast?"
        f"{_SHARED}&start_date=2024-06-01&end_date=2024-06-02"
    )
    hourly = json.loads(body)["hourly"]
    assert len(hourly["time"]) == 48 and hourly["temperature_2m"][0] is not None
