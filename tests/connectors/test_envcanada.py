"""Tests for the Environment Canada bulk-CSV observation connector."""

from __future__ import annotations

import math
import pathlib
from datetime import UTC, datetime

import pandas as pd
import pytest

from microclimate.connectors.base import SourceUnavailable, StationNotFound
from microclimate.connectors.sources.envcanada import EnvCanadaSource
from microclimate.contracts.observation import OBSERVATION_FRAME

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "envcanada"


def _load_fixture(name: str) -> str:
    return (_FIXTURE_DIR / name).read_text(encoding="utf-8-sig")


def _make_fetcher(csv_text: str):
    """Return a fetcher callable that always returns the given CSV text."""

    def fetcher(station_id: str, year: int, month: int) -> str:  # noqa: ARG001
        return csv_text

    return fetcher


# ---------------------------------------------------------------------------
# 1. Core mapping — OBSERVATION_FRAME validation + spot-check conversions
# ---------------------------------------------------------------------------


def test_frame_passes_observation_frame_schema() -> None:
    """fetch_historical returns a frame that passes OBSERVATION_FRAME.validate()."""
    csv_text = _load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    # Must not raise
    OBSERVATION_FRAME.validate(df)


def test_spot_check_05_LST_conversions() -> None:
    """Row 05:00 LST: verify unit conversions are correct."""
    csv_text = _load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    # 05:00 LST = 12:00 UTC (LST + 7h)
    ts_05 = pd.Timestamp("2026-04-27 12:00:00", tz="UTC")
    row = df[df["timestamp"] == ts_05]
    assert len(row) == 1, f"Expected exactly one row at 12:00 UTC, got {len(row)}"

    assert row["temp_c"].iloc[0] == pytest.approx(-6.8)  # type: ignore[reportUnknownMemberType]
    # 90.91 kPa × 10 = 909.1 hPa
    assert row["surface_pressure_hpa"].iloc[0] == pytest.approx(909.1, abs=0.01)  # type: ignore[reportUnknownMemberType]
    # 13 × 10 = 130°
    assert row["wind_dir_deg"].iloc[0] == pytest.approx(130.0)  # type: ignore[reportUnknownMemberType]
    # 11 km/h / 3.6 ≈ 3.056 m/s
    assert row["wind_speed_ms"].iloc[0] == pytest.approx(11.0 / 3.6, abs=0.001)  # type: ignore[reportUnknownMemberType]


def test_spot_check_timestamp_is_utc() -> None:
    """Timestamps are UTC (LST + 7h offset applied correctly)."""
    csv_text = _load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    # 03:00 LST + 7h = 10:00 UTC
    ts_03_utc = pd.Timestamp("2026-04-27 10:00:00", tz="UTC")
    assert df["timestamp"].isin([ts_03_utc]).any()


def test_station_id_column_matches_argument() -> None:
    """station_id column equals the argument passed, not the CSV Climate ID."""
    csv_text = _load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    assert (df["station_id"] == "49268").all()


# ---------------------------------------------------------------------------
# 2. Per-row masks — 06:00 LST row
# ---------------------------------------------------------------------------


def test_per_row_missing_fields_at_06_LST() -> None:
    """Row 06:00 LST: precip, wind_dir, wind_speed, surface_pressure are absent."""
    csv_text = _load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    # 06:00 LST = 13:00 UTC
    ts_06 = pd.Timestamp("2026-04-27 13:00:00", tz="UTC")
    row = df[df["timestamp"] == ts_06]
    assert len(row) == 1

    # These four must be absent (NaN + _present=False)
    assert math.isnan(row["precip_mm"].iloc[0])
    assert row["precip_mm_present"].iloc[0] == False  # noqa: E712
    assert math.isnan(row["wind_dir_deg"].iloc[0])
    assert row["wind_dir_deg_present"].iloc[0] == False  # noqa: E712
    assert math.isnan(row["wind_speed_ms"].iloc[0])
    assert row["wind_speed_ms_present"].iloc[0] == False  # noqa: E712
    assert math.isnan(row["surface_pressure_hpa"].iloc[0])
    assert row["surface_pressure_hpa_present"].iloc[0] == False  # noqa: E712

    # Temp and dewpoint ARE present for that row
    assert row["temp_c_present"].iloc[0] == True  # noqa: E712
    assert not math.isnan(row["temp_c"].iloc[0])
    assert row["dewpoint_c_present"].iloc[0] == True  # noqa: E712
    assert not math.isnan(row["dewpoint_c"].iloc[0])


# ---------------------------------------------------------------------------
# 3. Columns not in ECCC CSV → always absent
# ---------------------------------------------------------------------------


def test_cloud_cover_always_absent() -> None:
    """cloud_cover_fraction is NaN + _present=False for all rows."""
    csv_text = _load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    assert df["cloud_cover_fraction_present"].eq(False).all()
    assert df["cloud_cover_fraction"].isna().all()


def test_solar_radiation_always_absent() -> None:
    """solar_radiation_wm2 is NaN + _present=False for all rows."""
    csv_text = _load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    assert df["solar_radiation_wm2_present"].eq(False).all()
    assert df["solar_radiation_wm2"].isna().all()


# ---------------------------------------------------------------------------
# 4. Dewpoint derivation from RH
# ---------------------------------------------------------------------------


def test_dewpoint_derived_from_rh() -> None:
    """When Dew Point cell is blank, dewpoint is derived from T + RH via Magnus-Tetens."""
    csv_text = _load_fixture("dewpoint_derive.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    start = datetime(2023, 6, 1, 0, 0, tzinfo=UTC)
    end = datetime(2023, 6, 1, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    assert len(df) == 1
    assert df["dewpoint_c_present"].iloc[0] == True  # noqa: E712
    assert df["dewpoint_c"].iloc[0] == pytest.approx(10.6, abs=0.3)  # type: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# 5. fetch_historical end-boundary filtering
# ---------------------------------------------------------------------------


def test_historical_end_boundary_excludes_later_rows() -> None:
    """Rows after `end` are excluded; rows before `start` are excluded."""
    csv_text = _load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    # Include only 03:00–05:00 LST = 10:00–12:00 UTC
    start = datetime(2026, 4, 27, 10, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    assert len(df) > 0
    assert df["timestamp"].max() <= pd.Timestamp(end)
    assert df["timestamp"].min() >= pd.Timestamp(start)
    # 06:00 LST = 13:00 UTC must NOT be present
    ts_06_utc = pd.Timestamp("2026-04-27 13:00:00", tz="UTC")
    assert ts_06_utc not in df["timestamp"].values


def test_historical_sorted_by_timestamp() -> None:
    """fetch_historical result is sorted ascending by timestamp."""
    csv_text = _load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    timestamps = df["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# 6. fetch_live: since-filter + drop not-yet-reported rows
# ---------------------------------------------------------------------------


def test_live_since_filter_and_drop_empty_rows() -> None:
    """fetch_live: only rows >= since; truncated (no-measurement) rows are dropped."""
    csv_text = _load_fixture("live_partial.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    # 2026-05-29 21:00 LST = 2026-05-30 04:00 UTC
    # 2026-05-29 22:00 LST = 2026-05-30 05:00 UTC
    # Set since to after the 04:00 UTC row → only 05:00 UTC should be returned
    since = datetime(2026, 5, 30, 5, 0, tzinfo=UTC)
    df = source.fetch_live("49268", since)

    assert len(df) == 1
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-05-30 05:00:00", tz="UTC")
    OBSERVATION_FRAME.validate(df)


def test_live_drops_truncated_rows() -> None:
    """Truncated future rows (no measurement data) are never emitted."""
    csv_text = _load_fixture("live_partial.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    # since set before both real rows
    since = datetime(2026, 5, 30, 3, 0, tzinfo=UTC)
    df = source.fetch_live("49268", since)

    # Only the two real rows must be present, not the 3 truncated ones
    assert len(df) == 2
    # None of the timestamps should be from 2026-05-30 00:00 LST (07:00 UTC) onward
    assert df["timestamp"].max() <= pd.Timestamp("2026-05-30 06:00:00", tz="UTC")


def test_live_sorted_by_timestamp() -> None:
    """fetch_live result is sorted ascending by timestamp."""
    csv_text = _load_fixture("live_partial.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(csv_text))

    since = datetime(2026, 5, 29, 0, 0, tzinfo=UTC)
    df = source.fetch_live("49268", since)

    timestamps = df["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# 7. Exceptions
# ---------------------------------------------------------------------------


def test_non_station_body_raises_station_not_found() -> None:
    """A non-ECCC-station response (e.g. HTML) raises StationNotFound."""
    source = EnvCanadaSource(fetcher=_make_fetcher("<html>error</html>"))

    with pytest.raises(StationNotFound):
        source.fetch_historical(
            "BOGUS", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 31, tzinfo=UTC)
        )


def test_empty_body_raises_station_not_found() -> None:
    """An empty response body raises StationNotFound."""
    source = EnvCanadaSource(fetcher=_make_fetcher(""))

    with pytest.raises(StationNotFound):
        source.fetch_historical(
            "BOGUS", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 31, tzinfo=UTC)
        )


def test_source_unavailable_propagates() -> None:
    """If the fetcher raises SourceUnavailable, it propagates unchanged."""

    def failing_fetcher(station_id: str, year: int, month: int) -> str:
        raise SourceUnavailable("network down")

    source = EnvCanadaSource(fetcher=failing_fetcher)

    with pytest.raises(SourceUnavailable):
        source.fetch_historical(
            "49268", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 31, tzinfo=UTC)
        )


# ---------------------------------------------------------------------------
# 8. Registry — zero-arg instantiation still works
# ---------------------------------------------------------------------------


def test_no_args_instantiation() -> None:
    """EnvCanadaSource() (no args) must instantiate without error."""
    import microclimate.connectors  # noqa: F401  # type: ignore[reportUnusedImport]  (populates registry)
    from microclimate.connectors.registry import get_source

    source = get_source("envcanada")
    assert isinstance(source, EnvCanadaSource)


# ---------------------------------------------------------------------------
# Optional live / network test
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_network_fetch_historical() -> None:
    """Hits the real ECCC endpoint; requires internet access."""
    source = EnvCanadaSource()
    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 31, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    assert len(df) > 0
    OBSERVATION_FRAME.validate(df)
    assert df["timestamp"].max() <= pd.Timestamp(end)
    assert df["timestamp"].min() >= pd.Timestamp(start)
