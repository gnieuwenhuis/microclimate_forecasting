"""Hermetic unit tests for HrdpsCasparSource.

All tests use an injectable opener returning a synthetic xr.Dataset from
build_hrdps_dataset(var_map=CASPAR_VAR_MAP) — no cfgrib, no network, no real archive.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

import microclimate.connectors  # noqa: F401  # type: ignore[reportUnusedImport]  (populates registry)
from microclimate.connectors.base import ForecastUnavailable, SourceUnavailable
from microclimate.connectors.registry import get_source
from microclimate.connectors.sources.hrdps_caspar import (
    CASPAR_VAR_MAP,
    HrdpsCasparSource,
    _archive_path,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
    _resolve_existing_archive,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
)
from microclimate.connectors.sources.hrdps_datamart import HRDPS_VAR_MAP, HrdpsDatamartSource
from microclimate.contracts.forecast_frame import FORECAST_FRAME

from .conftest import build_hrdps_dataset

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

_ISSUE_TIME = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)
_LAT = 51.0
_LON = -114.0
_LEAD_HOURS = [1, 2, 3]


def _make_caspar_ds() -> xr.Dataset:
    """Return a synthetic Dataset keyed by CASPAR_VAR_MAP variable names."""
    return build_hrdps_dataset(var_map=CASPAR_VAR_MAP, lead_hours=(0, 1, 2, 3))


# ---------------------------------------------------------------------------
# 1. Pinned archive layout: _archive_path
# ---------------------------------------------------------------------------


def test_archive_path_basic(tmp_path: Path) -> None:
    """Pinned layout: 2026-05-30 00:00 UTC → 2026/05/hrdps_2026053000.grib2."""
    result = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    assert result == tmp_path / "2026" / "05" / "hrdps_2026053000.grib2"


def test_archive_path_single_digit_month(tmp_path: Path) -> None:
    """Single-digit month is zero-padded: 2026-03-07 09:00 UTC → 2026/03/hrdps_2026030709."""
    issue = datetime(2026, 3, 7, 9, 0, tzinfo=UTC)
    result = _archive_path(tmp_path, issue, ".nc")
    assert result == tmp_path / "2026" / "03" / "hrdps_2026030709.nc"


def test_archive_path_midnight_hour(tmp_path: Path) -> None:
    """Midnight (hour=0) is zero-padded: 2025-12-01 00:00 UTC → hrdps_2025120100."""
    issue = datetime(2025, 12, 1, 0, 0, tzinfo=UTC)
    result = _archive_path(tmp_path, issue, ".grib2")
    assert result == tmp_path / "2025" / "12" / "hrdps_2025120100.grib2"


# ---------------------------------------------------------------------------
# 2. Happy path
# ---------------------------------------------------------------------------


def test_happy_path_forecast_frame_valid(tmp_path: Path) -> None:
    """fetch_forecast returns a FORECAST_FRAME-valid DataFrame on success."""
    # Place an empty file at the expected path so resolution succeeds.
    expected = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    ds = _make_caspar_ds()
    source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds)
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    FORECAST_FRAME.validate(df)


def test_happy_path_lead_hour_column(tmp_path: Path) -> None:
    """lead_hour column exactly matches requested lead hours."""
    expected = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    ds = _make_caspar_ds()
    source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds)
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    assert list(df["lead_hour"]) == _LEAD_HOURS


def test_happy_path_valid_time_column(tmp_path: Path) -> None:
    """valid_time == issue_time + lead_hour for every row."""
    expected = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    ds = _make_caspar_ds()
    source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds)
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    for _, row in df.iterrows():
        expected_ts = pd.Timestamp(_ISSUE_TIME + timedelta(hours=int(row["lead_hour"])))
        assert row["valid_time"] == expected_ts


def test_happy_path_all_vars_non_null(tmp_path: Path) -> None:
    """All 8 physical variables must be non-null for every row."""
    expected = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    ds = _make_caspar_ds()
    source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds)
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


def test_happy_path_pinned_temp_c(tmp_path: Path) -> None:
    """target cell (51.0/-114.0) has TT=288.15 K → 15.0 °C after conversion."""
    expected = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    ds = _make_caspar_ds()
    source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds)
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    assert float(df["temp_c"].iloc[0]) == pytest.approx(15.0, abs=1e-6)  # type: ignore[reportUnknownMemberType]


def test_happy_path_pinned_surface_pressure_hpa(tmp_path: Path) -> None:
    """target cell has PN=90000 Pa → 900.0 hPa after conversion."""
    expected = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    ds = _make_caspar_ds()
    source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds)
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    assert float(df["surface_pressure_hpa"].iloc[0]) == pytest.approx(900.0, abs=1e-6)  # type: ignore[reportUnknownMemberType]


def test_happy_path_pinned_precip_de_accumulation(tmp_path: Path) -> None:
    """Precip de-accumulation: PR=[0.0, 0.5, 2.0, 2.0] → per-hour [0.5, 1.5, 0.0]."""
    expected = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    ds = _make_caspar_ds()
    source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds)
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=[1, 2, 3])
    assert float(df.loc[df["lead_hour"] == 1, "precip_mm"].iloc[0]) == pytest.approx(0.5, abs=1e-6)  # type: ignore[reportUnknownMemberType]
    assert float(df.loc[df["lead_hour"] == 2, "precip_mm"].iloc[0]) == pytest.approx(1.5, abs=1e-6)  # type: ignore[reportUnknownMemberType]
    assert float(df.loc[df["lead_hour"] == 3, "precip_mm"].iloc[0]) == pytest.approx(0.0, abs=1e-6)  # type: ignore[reportUnknownMemberType]


def test_happy_path_pinned_cloud_cover_fraction(tmp_path: Path) -> None:
    """target cell has NT=50 % → 0.5 after conversion."""
    expected = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    ds = _make_caspar_ds()
    source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds)
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    assert float(df["cloud_cover_fraction"].iloc[0]) == pytest.approx(0.5, abs=1e-6)  # type: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# 3. Exception contract
# ---------------------------------------------------------------------------


def test_missing_file_raises_forecast_unavailable(tmp_path: Path) -> None:
    """archive_root with no matching file → ForecastUnavailable."""
    ds = _make_caspar_ds()
    source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds)
    with pytest.raises(ForecastUnavailable):
        source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)


def test_source_unavailable_from_opener_propagates_unchanged(tmp_path: Path) -> None:
    """opener raising SourceUnavailable → fetch_forecast propagates SourceUnavailable."""
    expected = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    def failing_opener(path: Path) -> xr.Dataset:  # noqa: ARG001
        raise SourceUnavailable("disk I/O error")

    source = HrdpsCasparSource(archive_root=tmp_path, opener=failing_opener)
    with pytest.raises(SourceUnavailable):
        source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)


def test_truncated_run_raises_forecast_unavailable(tmp_path: Path) -> None:
    """Dataset missing lead_hour=0 → core ValueError → ForecastUnavailable (chained)."""
    expected = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    ds = build_hrdps_dataset(var_map=CASPAR_VAR_MAP, lead_hours=(1, 2, 3))  # missing hour 0
    source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds)
    with pytest.raises(ForecastUnavailable) as exc_info:
        source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=[1])
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_no_archive_root_raises_forecast_unavailable() -> None:
    """fetch_forecast with no configured archive_root raises ForecastUnavailable."""
    ds = _make_caspar_ds()
    source = HrdpsCasparSource.__new__(HrdpsCasparSource)
    # Directly set _archive_root to None to force the error path, bypassing env.
    source._archive_root = None  # type: ignore[reportPrivateUsage]
    source._opener = lambda _p: ds  # type: ignore[reportPrivateUsage]
    with pytest.raises(ForecastUnavailable, match="CASPAR_ARCHIVE_ROOT"):
        source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)


# ---------------------------------------------------------------------------
# 4. Extension resolution
# ---------------------------------------------------------------------------


def test_extension_resolution_nc_fallback(tmp_path: Path) -> None:
    """Creates .nc file only; connector resolves it even if .grib2 absent."""
    expected_nc = _archive_path(tmp_path, _ISSUE_TIME, ".nc")
    expected_nc.parent.mkdir(parents=True, exist_ok=True)
    expected_nc.touch()

    ds = _make_caspar_ds()
    source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds)
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    FORECAST_FRAME.validate(df)


def test_extension_resolution_grib2_preferred_over_nc(tmp_path: Path) -> None:
    """When both .grib2 and .nc exist, .grib2 wins (first in _SUPPORTED_EXTS)."""
    grib2_path = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    nc_path = _archive_path(tmp_path, _ISSUE_TIME, ".nc")
    grib2_path.parent.mkdir(parents=True, exist_ok=True)
    grib2_path.touch()
    nc_path.touch()

    resolved = _resolve_existing_archive(tmp_path, _ISSUE_TIME)
    assert resolved == grib2_path


# ---------------------------------------------------------------------------
# 5. Skew check: both connectors through the shared core produce identical output
# ---------------------------------------------------------------------------


def test_no_train_serve_skew(tmp_path: Path) -> None:
    """Both connectors through the shared core must produce frame-equal DataFrames.

    This is the key anti-skew assertion: HRDPS_VAR_MAP and CASPAR_VAR_MAP use
    different variable names, but the same underlying values flow through the
    same dataset_to_forecast_frame core.  Both DataFrames must be equal.
    """
    # Build two datasets with IDENTICAL underlying values, different variable names.
    ds_dm = build_hrdps_dataset(var_map=HRDPS_VAR_MAP, lead_hours=(0, 1, 2, 3))
    ds_cp = build_hrdps_dataset(var_map=CASPAR_VAR_MAP, lead_hours=(0, 1, 2, 3))

    # Create the archive file so CaSPAr path resolution succeeds.
    caspar_file = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    caspar_file.parent.mkdir(parents=True, exist_ok=True)
    caspar_file.touch()

    datamart_source = HrdpsDatamartSource(opener=lambda _issue, _leads: ds_dm)
    caspar_source = HrdpsCasparSource(archive_root=tmp_path, opener=lambda _p: ds_cp)

    df_dm = datamart_source.fetch_forecast(
        issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS
    )
    df_cp = caspar_source.fetch_forecast(
        issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS
    )

    pd.testing.assert_frame_equal(df_dm, df_cp)


# ---------------------------------------------------------------------------
# 6. Registry checks
# ---------------------------------------------------------------------------


def test_registry_hrdps_caspar_is_registered() -> None:
    """hrdps_caspar is registered and returns an HrdpsCasparSource."""
    source = get_source("hrdps_caspar")
    assert isinstance(source, HrdpsCasparSource)


def test_registry_hrdps_caspar_is_not_live() -> None:
    """Registry-instantiated HrdpsCasparSource has is_live == False."""
    source = get_source("hrdps_caspar")
    assert isinstance(source, HrdpsCasparSource)
    assert source.is_live is False


# ---------------------------------------------------------------------------
# 7. CASPAR_ARCHIVE_ROOT env var
# ---------------------------------------------------------------------------


def test_env_var_archive_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CASPAR_ARCHIVE_ROOT env var is used when no archive_root arg is passed."""
    monkeypatch.setenv("CASPAR_ARCHIVE_ROOT", str(tmp_path))

    # Create the expected file so resolution succeeds.
    expected = _archive_path(tmp_path, _ISSUE_TIME, ".grib2")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.touch()

    ds = _make_caspar_ds()
    # Construct with NO archive_root arg — must pick up env var.
    source = HrdpsCasparSource(opener=lambda _p: ds)
    df = source.fetch_forecast(issue_time=_ISSUE_TIME, lat=_LAT, lon=_LON, lead_hours=_LEAD_HOURS)
    FORECAST_FRAME.validate(df)
