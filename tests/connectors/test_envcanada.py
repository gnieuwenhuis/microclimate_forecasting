"""Tests for the Environment Canada bulk-CSV observation connector."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from microclimate.connectors.base import SourceUnavailable, StationNotFound
from microclimate.connectors.sources.envcanada import EnvCanadaSource
from microclimate.contracts.observation import OBSERVATION_FRAME

from .conftest import load_fixture, make_fetcher

# ---------------------------------------------------------------------------
# 1. Core mapping — OBSERVATION_FRAME validation + spot-check conversions
# ---------------------------------------------------------------------------


def test_frame_passes_observation_frame_schema() -> None:
    """fetch_historical returns a frame that passes OBSERVATION_FRAME.validate()."""
    csv_text = load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    # Must not raise
    OBSERVATION_FRAME.validate(df)


def test_spot_check_05_LST_conversions() -> None:
    """Row 05:00 LST: verify unit conversions are correct."""
    csv_text = load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

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
    csv_text = load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    # 03:00 LST + 7h = 10:00 UTC
    ts_03_utc = pd.Timestamp("2026-04-27 10:00:00", tz="UTC")
    assert df["timestamp"].isin([ts_03_utc]).any()


def test_station_id_column_matches_argument() -> None:
    """station_id column equals the argument passed, not the CSV Climate ID."""
    csv_text = load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    assert (df["station_id"] == "49268").all()


# ---------------------------------------------------------------------------
# 2. Per-row masks — 06:00 LST row
# ---------------------------------------------------------------------------


def test_per_row_missing_fields_at_06_LST() -> None:
    """Row 06:00 LST: precip, wind_dir, wind_speed, surface_pressure are absent."""
    csv_text = load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

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
    csv_text = load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    assert df["cloud_cover_fraction_present"].eq(False).all()
    assert df["cloud_cover_fraction"].isna().all()


def test_solar_radiation_always_absent() -> None:
    """solar_radiation_wm2 is NaN + _present=False for all rows."""
    csv_text = load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

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
    csv_text = load_fixture("dewpoint_derive.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

    start = datetime(2023, 6, 1, 0, 0, tzinfo=UTC)
    end = datetime(2023, 6, 1, 23, 59, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    assert len(df) == 1
    assert df["dewpoint_c_present"].iloc[0] == True  # noqa: E712
    # Exact Magnus-Tetens for T=15.0, RH=75 ≈ 10.604 — pins the formula.
    assert df["dewpoint_c"].iloc[0] == pytest.approx(10.604, abs=0.01)  # type: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# 5. fetch_historical end-boundary filtering
# ---------------------------------------------------------------------------


def test_historical_end_boundary_excludes_later_rows() -> None:
    """Rows after `end` are excluded; rows before `start` are excluded."""
    csv_text = load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

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
    csv_text = load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

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
    csv_text = load_fixture("live_partial.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

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
    csv_text = load_fixture("live_partial.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

    # since set before both real rows
    since = datetime(2026, 5, 30, 3, 0, tzinfo=UTC)
    df = source.fetch_live("49268", since)

    # Only the two real rows must be present, not the 3 truncated ones
    assert len(df) == 2
    # None of the timestamps should be from 2026-05-30 00:00 LST (07:00 UTC) onward
    assert df["timestamp"].max() <= pd.Timestamp("2026-05-30 06:00:00", tz="UTC")


def test_live_sorted_by_timestamp() -> None:
    """fetch_live result is sorted ascending by timestamp."""
    csv_text = load_fixture("live_partial.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

    since = datetime(2026, 5, 29, 0, 0, tzinfo=UTC)
    df = source.fetch_live("49268", since)

    timestamps = df["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# 7. Exceptions — non-station bodies and network failures still raise
# ---------------------------------------------------------------------------


def test_non_station_body_raises_station_not_found() -> None:
    """A non-ECCC-station response (e.g. HTML) raises StationNotFound."""
    source = EnvCanadaSource(fetcher=make_fetcher("<html>error</html>"))

    with pytest.raises(StationNotFound):
        source.fetch_historical(
            "BOGUS", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 31, tzinfo=UTC)
        )


def test_empty_body_raises_station_not_found() -> None:
    """An empty response body raises StationNotFound."""
    source = EnvCanadaSource(fetcher=make_fetcher(""))

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


def test_valid_header_empty_data_returns_empty_frame() -> None:
    """A valid ECCC station CSV with zero data rows returns an empty schema-valid frame.

    This is ADR-0002 graceful degradation: a valid station with no data in the window
    is NOT the same as a missing/invalid station.
    """
    csv_text = load_fixture("empty_month.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

    df = source.fetch_historical(
        "49268", datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 3, 31, tzinfo=UTC)
    )
    assert len(df) == 0
    OBSERVATION_FRAME.validate(df)


def test_valid_header_empty_data_live_returns_empty_frame() -> None:
    """fetch_live with a valid-header but no-data CSV returns an empty schema-valid frame."""
    csv_text = load_fixture("empty_month.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(csv_text))

    df = source.fetch_live("49268", datetime(2026, 5, 1, tzinfo=UTC))
    assert len(df) == 0
    OBSERVATION_FRAME.validate(df)


# ---------------------------------------------------------------------------
# 8. Multi-month fetch_historical (m-3)
# ---------------------------------------------------------------------------


def test_multi_month_fetch_historical_stitches_months() -> None:
    """fetch_historical spanning two months stitches rows from both months.

    Verifies:
    (a) The fetcher is called once per month in the range.
    (b) Rows from both months appear in the combined result, sorted by timestamp.
    (c) The window filter trims both ends.
    (d) An empty middle month does NOT abort the fetch.
    """
    april_csv = load_fixture("hourly_window.csv")  # April 2026 rows (03:00–09:00 LST)
    empty_csv = load_fixture("empty_month.csv")  # Empty month (no data rows)
    may_csv = load_fixture("may_window.csv")  # May 2026 rows (03:00–04:00 LST)

    call_log: list[tuple[int, int]] = []

    # Span: late April 2026 → early May 2026 (3 calendar months: Apr, empty-interstitial
    # skipped in this two-month case — we test with Apr + May only, two months)
    fixtures: dict[tuple[int, int], str] = {
        (2026, 4): april_csv,
        (2026, 5): may_csv,
    }

    def multi_fetcher(station_id: str, year: int, month: int) -> str:  # noqa: ARG001
        call_log.append((year, month))
        return fixtures.get((year, month), empty_csv)

    source = EnvCanadaSource(fetcher=multi_fetcher)

    # Window: 2026-04-27 10:00 UTC → 2026-05-01 12:00 UTC
    # April fixture has rows from 03:00 LST (10:00 UTC) to 09:00 LST (16:00 UTC) on 2026-04-27.
    # May fixture has rows at 03:00 LST (10:00 UTC) and 04:00 LST (11:00 UTC) on 2026-05-01.
    start = datetime(2026, 4, 27, 10, 0, tzinfo=UTC)
    end = datetime(2026, 5, 1, 11, 0, tzinfo=UTC)

    df = source.fetch_historical("49268", start, end)

    # (a) Fetcher called for each month in range
    assert (2026, 4) in call_log
    assert (2026, 5) in call_log

    # (b) Rows from both months present and sorted
    assert len(df) > 0
    timestamps = df["timestamp"].tolist()
    assert timestamps == sorted(timestamps)

    # Check that both April and May timestamps appear (use isin to preserve tz-awareness)
    april_ts = pd.Timestamp("2026-04-27 10:00:00", tz="UTC")  # 03:00 LST + 7h
    may_ts = pd.Timestamp("2026-05-01 10:00:00", tz="UTC")  # 03:00 LST + 7h
    assert df["timestamp"].isin([april_ts]).any()
    assert df["timestamp"].isin([may_ts]).any()

    # (c) Window filter trims: no row after end (2026-05-01 11:00 UTC)
    assert df["timestamp"].max() <= pd.Timestamp(end)
    assert df["timestamp"].min() >= pd.Timestamp(start)

    # (d) An empty interstitial month does not abort: test three-month span with empty middle
    call_log.clear()
    fixtures_with_empty: dict[tuple[int, int], str] = {
        (2026, 4): april_csv,
        (2026, 5): empty_csv,  # empty middle month
        (2026, 6): may_csv,  # reuse may CSV but timestamped in June — treat as data present
    }

    def multi_fetcher_with_empty(station_id: str, year: int, month: int) -> str:  # noqa: ARG001
        call_log.append((year, month))
        return fixtures_with_empty.get((year, month), empty_csv)

    source2 = EnvCanadaSource(fetcher=multi_fetcher_with_empty)
    # Span April–June (LST-keyed); May is empty. Window covers all three LST months.
    # April rows are inside the window; May is empty (no abort); June CSV has May-dated
    # timestamps which fall outside the window filter → only April rows survive.
    # end2 must be >= 2026-06-01 07:00 UTC so that end_lst falls in June LST (UTC-7).
    start2 = datetime(2026, 4, 27, 10, 0, tzinfo=UTC)
    end2 = datetime(2026, 6, 2, 0, 0, tzinfo=UTC)  # spans Apr, May, Jun LST months
    df2 = source2.fetch_historical("49268", start2, end2)

    # Fetcher called for all three months
    assert (2026, 4) in call_log
    assert (2026, 5) in call_log
    assert (2026, 6) in call_log

    # April rows still returned despite empty May
    assert len(df2) > 0
    assert df2["timestamp"].isin([april_ts]).any()
    OBSERVATION_FRAME.validate(df2)


# ---------------------------------------------------------------------------
# 9. UTC/LST month-boundary correctness
# ---------------------------------------------------------------------------


def test_historical_fetches_lst_keyed_month_at_utc_boundary() -> None:
    """fetch_historical at a UTC/LST month boundary returns the row from the LST-keyed month.

    Scenario: a row stored as ``2026-04-30 20:00 LST`` has UTC timestamp
    ``2026-05-01 03:00 UTC`` (20:00 + 7h). It lives in the **April** CSV.
    A UTC window ``2026-05-01 00:00–06:00 UTC`` must therefore fetch the April CSV
    and return that row.

    Against the old UTC-keyed code (which would fetch only May) this test returns an
    empty frame, causing the ``assert len(df) == 1`` to fail.
    """
    april_csv = load_fixture("april_lst_boundary.csv")
    # May has no data for this window
    empty_csv = load_fixture("empty_month.csv")

    call_log: list[tuple[int, int]] = []
    fixtures: dict[tuple[int, int], str] = {
        (2026, 4): april_csv,
        (2026, 5): empty_csv,
    }

    def boundary_fetcher(station_id: str, year: int, month: int) -> str:  # noqa: ARG001
        call_log.append((year, month))
        return fixtures.get((year, month), empty_csv)

    source = EnvCanadaSource(fetcher=boundary_fetcher)

    # UTC window: 2026-05-01 03:00 → 2026-05-01 06:00 UTC
    # The row 2026-04-30 20:00 LST = 2026-05-01 03:00 UTC falls inside this window,
    # but it lives in the April CSV (keyed by LST month).
    start = datetime(2026, 5, 1, 3, 0, tzinfo=UTC)
    end = datetime(2026, 5, 1, 6, 0, tzinfo=UTC)
    df = source.fetch_historical("49268", start, end)

    # The April-CSV row (LST 2026-04-30 20:00 → UTC 2026-05-01 03:00) must be returned.
    expected_ts = pd.Timestamp("2026-05-01 03:00:00", tz="UTC")
    assert len(df) == 1, f"Expected 1 row, got {len(df)} — old UTC-keyed code returns 0"
    assert df["timestamp"].iloc[0] == expected_ts

    # The fetcher must have been asked for April (the LST-keyed month), not May.
    assert (2026, 4) in call_log, "Fetcher was not asked for (2026, 4) — LST-keyed month missing"
    assert (2026, 5) not in call_log, "Fetcher asked for (2026, 5) — should only need April"

    OBSERVATION_FRAME.validate(df)


def test_lst_month_range_covers_both_sides_of_boundary() -> None:
    """_month_range on LST-converted bounds spans the correct months.

    Unit test for the LST-bounds month-selection logic used in both fetch methods.
    Verifies that a UTC window straddling the April→May boundary at UTC-midnight
    (which is still April in LST) asks for April, not only May.
    """
    from microclimate.connectors.sources.envcanada import (
        _LST_UTC_OFFSET,  # type: ignore[reportPrivateUsage]
        _month_range,  # type: ignore[reportPrivateUsage]
    )

    # UTC window: 2026-05-01 00:00 → 2026-05-01 06:00
    # In LST: 2026-04-30 17:00 → 2026-04-30 23:00  → entirely in April LST month
    start_utc = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    end_utc = datetime(2026, 5, 1, 6, 0, tzinfo=UTC)

    start_lst = start_utc - _LST_UTC_OFFSET
    end_lst = end_utc - _LST_UTC_OFFSET
    months = _month_range(start_lst, end_lst)

    assert months == [(2026, 4)], (
        f"Expected [(2026, 4)] from LST-keyed range, got {months} — "
        "old UTC-keyed code would produce [(2026, 5)]"
    )


# ---------------------------------------------------------------------------
# 10. Registry — zero-arg instantiation still works
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
