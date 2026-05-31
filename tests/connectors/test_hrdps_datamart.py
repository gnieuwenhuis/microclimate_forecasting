"""Hermetic unit tests for HrdpsDatamartSource.

All tests use an injectable opener returning a synthetic xr.Dataset from
build_hrdps_dataset — no cfgrib, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

import microclimate.connectors  # noqa: F401  # type: ignore[reportUnusedImport]  (populates registry)
from microclimate.connectors.base import ForecastUnavailable, SourceUnavailable
from microclimate.connectors.registry import get_source, is_registered
from microclimate.connectors.sources.hrdps_datamart import HrdpsDatamartSource
from microclimate.contracts.forecast_frame import FORECAST_FRAME

from .conftest import build_hrdps_dataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ISSUE_TIME = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)
_LAT = 51.0
_LON = -114.0
_LEAD_HOURS = [1, 2, 3]


def _make_source() -> HrdpsDatamartSource:
    """Return an HrdpsDatamartSource with a hermetic opener returning a synthetic Dataset."""
    ds = build_hrdps_dataset(lead_hours=(0, 1, 2, 3))
    return HrdpsDatamartSource(opener=lambda _issue, _leads: ds)


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_happy_path_forecast_frame_valid() -> None:
    """fetch_forecast returns a FORECAST_FRAME-valid DataFrame on success."""
    source = _make_source()
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    FORECAST_FRAME.validate(df)


def test_happy_path_lead_hour_column() -> None:
    """lead_hour column exactly matches requested lead hours."""
    source = _make_source()
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    assert list(df["lead_hour"]) == _LEAD_HOURS


def test_happy_path_valid_time_column() -> None:
    """valid_time == issue_time + lead_hour for every row."""
    source = _make_source()
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    for _, row in df.iterrows():
        expected = pd.Timestamp(_ISSUE_TIME + timedelta(hours=int(row["lead_hour"])))
        assert row["valid_time"] == expected


def test_happy_path_all_vars_non_null() -> None:
    """All 8 physical variables must be non-null for every row."""
    source = _make_source()
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    phys_vars = [
        "temp_c",
        "dewpoint_c",
        "surface_pressure_hpa",
        "precip_mm",
        "cloud_cover_fraction",
        "solar_radiation_wm2",
        "wind_speed_ms",
        "wind_dir_deg",
    ]
    for var in phys_vars:
        assert df[var].notna().all(), f"{var} has null values"


def test_happy_path_pinned_temp_c() -> None:
    """target cell (51.0/-114.0) has t2m=288.15 K → 15.0 °C after conversion."""
    source = _make_source()
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    # All lead hours at the target cell have the same constant temperature.
    assert float(df["temp_c"].iloc[0]) == pytest.approx(15.0, abs=1e-6)  # type: ignore[reportUnknownMemberType]
    assert float(df["temp_c"].iloc[1]) == pytest.approx(15.0, abs=1e-6)  # type: ignore[reportUnknownMemberType]
    assert float(df["temp_c"].iloc[2]) == pytest.approx(15.0, abs=1e-6)  # type: ignore[reportUnknownMemberType]


def test_happy_path_pinned_surface_pressure_hpa() -> None:
    """target cell has sp=90000 Pa → 900.0 hPa after conversion."""
    source = _make_source()
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    assert float(df["surface_pressure_hpa"].iloc[0]) == pytest.approx(900.0, abs=1e-6)  # type: ignore[reportUnknownMemberType]


def test_happy_path_pinned_precip_de_accumulation() -> None:
    """Precip de-accumulation: tp=[0.0, 0.5, 2.0, 2.0] → per-hour [0.5, 1.5, 0.0]."""
    source = _make_source()
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=[1, 2, 3])
    # lead_hour=1: acc(1)-acc(0) = 0.5-0.0 = 0.5
    assert float(df.loc[df["lead_hour"] == 1, "precip_mm"].iloc[0]) == pytest.approx(0.5, abs=1e-6)  # type: ignore[reportUnknownMemberType]
    # lead_hour=2: acc(2)-acc(1) = 2.0-0.5 = 1.5
    assert float(df.loc[df["lead_hour"] == 2, "precip_mm"].iloc[0]) == pytest.approx(1.5, abs=1e-6)  # type: ignore[reportUnknownMemberType]
    # lead_hour=3: acc(3)-acc(2) = 2.0-2.0 = 0.0
    assert float(df.loc[df["lead_hour"] == 3, "precip_mm"].iloc[0]) == pytest.approx(0.0, abs=1e-6)  # type: ignore[reportUnknownMemberType]


def test_happy_path_pinned_cloud_cover_fraction() -> None:
    """target cell has tcc=50 % → 0.5 after conversion."""
    source = _make_source()
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    assert float(df["cloud_cover_fraction"].iloc[0]) == pytest.approx(0.5, abs=1e-6)  # type: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# 2. Exception contract
# ---------------------------------------------------------------------------


def test_source_unavailable_from_opener_propagates_unchanged() -> None:
    """opener raising SourceUnavailable → fetch_forecast propagates SourceUnavailable."""

    def failing_opener(issue_time: datetime, lead_hours: object) -> None:  # type: ignore[return]  # intentionally raises
        raise SourceUnavailable("Datamart unreachable")

    source = HrdpsDatamartSource(opener=failing_opener)  # type: ignore[arg-type]
    with pytest.raises(SourceUnavailable):
        source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)


def test_forecast_unavailable_from_opener_propagates_unchanged() -> None:
    """opener raising ForecastUnavailable → fetch_forecast propagates ForecastUnavailable."""

    def failing_opener(issue_time: datetime, lead_hours: object) -> None:  # type: ignore[return]  # intentionally raises
        raise ForecastUnavailable("run missing")

    source = HrdpsDatamartSource(opener=failing_opener)  # type: ignore[arg-type]
    with pytest.raises(ForecastUnavailable):
        source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)


def test_truncated_run_missing_lead_hour_raises_forecast_unavailable() -> None:
    """Opener returning dataset missing lead_hour=0 → core raises ValueError → ForecastUnavailable.

    This is the hermetic truncated-run test: no cfgrib, no network.  The opener
    returns a synthetic dataset that is missing lead_hour=0, which is required
    for de-accumulation when lead_hours=[1].  The core raises ValueError and
    fetch_forecast must wrap it as ForecastUnavailable with the ValueError chained
    as __cause__.
    """
    # build_hrdps_dataset with lead_hours=(1, 2, 3) — missing hour 0.
    ds = build_hrdps_dataset(lead_hours=(1, 2, 3))
    source = HrdpsDatamartSource(opener=lambda _issue, _leads: ds)
    with pytest.raises(ForecastUnavailable) as exc_info:
        source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=[1])
    # The core's ValueError must be chained as __cause__.
    assert isinstance(exc_info.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# 3. Registry checks
# ---------------------------------------------------------------------------


def test_registry_hrdps_datamart_is_registered() -> None:
    """hrdps_datamart is registered and returns an HrdpsDatamartSource."""
    source = get_source("hrdps_datamart")
    assert isinstance(source, HrdpsDatamartSource)


def test_registry_hrdps_datamart_is_live() -> None:
    """Registry-instantiated HrdpsDatamartSource has is_live == True."""
    source = get_source("hrdps_datamart")
    assert isinstance(source, HrdpsDatamartSource)
    assert source.is_live is True


def test_registry_hrdps_geomet_is_not_registered() -> None:
    """hrdps_geomet key must be gone after the rename."""
    assert is_registered("hrdps_geomet") is False


# ---------------------------------------------------------------------------
# 4. Network smoke test (deselected in normal CI via addopts="-m 'not network'")
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_network_smoke_open_latest_run() -> None:  # pragma: no cover
    """Smoke test: hit the real MSC Datamart to verify the seam is reachable.

    This test is skipped in hermetic environments (eccodes/network unavailable).
    It is only meant to be run manually when eccodes is available and network
    access to dd.weather.gc.ca is confirmed.

    NOTE: URL patterns and GRIB shortNames in _open_latest_run are unverified
    against live Datamart data — this test may fail until those are confirmed.
    """
    from microclimate.connectors.sources.hrdps_datamart import (
        _open_latest_run,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
    )

    issue = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)
    ds = _open_latest_run(issue, [1, 2])
    assert "lead_hour" in ds.coords
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
