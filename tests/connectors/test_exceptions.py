"""Tests for typed connector exceptions (L2)."""

from __future__ import annotations

from microclimate.connectors.base import (
    ConnectorError,
    ForecastUnavailable,
    SourceUnavailable,
    StationNotFound,
)


def test_connector_error_is_exception_subclass() -> None:
    assert issubclass(ConnectorError, Exception)


def test_source_unavailable_is_connector_error() -> None:
    assert issubclass(SourceUnavailable, ConnectorError)


def test_forecast_unavailable_is_connector_error() -> None:
    assert issubclass(ForecastUnavailable, ConnectorError)


def test_station_not_found_is_connector_error() -> None:
    assert issubclass(StationNotFound, ConnectorError)


def test_exceptions_are_raiseable() -> None:
    import pytest

    with pytest.raises(SourceUnavailable):
        raise SourceUnavailable("test source down")

    with pytest.raises(ForecastUnavailable):
        raise ForecastUnavailable("no forecast available")

    with pytest.raises(StationNotFound):
        raise StationNotFound("station 9835 not found")
