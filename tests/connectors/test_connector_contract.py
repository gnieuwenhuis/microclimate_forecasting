from __future__ import annotations

import pandas as pd
import pytest

import microclimate.connectors  # noqa: F401  # type: ignore[reportUnusedImport]  (populates the registry)
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.registry import get_source, registered_keys
from microclimate.contracts.forecast_frame import FORECAST_FRAME
from microclimate.contracts.observation import OBSERVATION_FRAME

from .conftest import build_hrdps_dataset, load_fixture, make_fetcher

_KEYS = sorted(k for k in registered_keys() if not k.startswith("_"))

# ---------------------------------------------------------------------------
# Sources whose fetch_* is not yet implemented — skip gracefully.
# ---------------------------------------------------------------------------
_NOT_YET_IMPLEMENTED: frozenset[str] = frozenset({"acis"})


# ---------------------------------------------------------------------------
# Structural contract (all registered sources)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", _KEYS)
def test_source_conforms_to_contract(key: str) -> None:
    source = get_source(key)
    assert isinstance(source, (NWPSource, ObservationSource))
    if isinstance(source, ObservationSource):
        assert source.historical_coverage in {"deep", "shallow", "none"}
        assert callable(source.fetch_historical)
        assert callable(source.fetch_live)
    else:
        assert isinstance(source.is_live, bool)
        assert callable(source.fetch_forecast)


# ---------------------------------------------------------------------------
# Behavioural contract — hermetic for envcanada + hrdps_datamart; skip the rest.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", _KEYS)
def test_source_behavioral_contract(key: str) -> None:
    """Behavioural assertions for observation sources.

    * envcanada: driven hermetically with fixture-backed fetcher.
    * hrdps_datamart: driven hermetically with injectable opener + synthetic Dataset.
    * acis / hrdps_caspar: skipped until fetch_* is implemented.
    """
    if key in _NOT_YET_IMPLEMENTED:
        pytest.skip(f"{key}: fetch_* not yet implemented")

    if key == "envcanada":
        _assert_envcanada_behavioral_contract()
        return

    if key == "hrdps_datamart":
        _assert_hrdps_datamart_behavioral_contract()
        return

    if key == "hrdps_caspar":
        _assert_hrdps_caspar_behavioral_contract()
        return

    # If a new key is added without being covered here, fail loudly.
    pytest.fail(f"No behavioral contract assertion defined for source key {key!r}")


def _assert_hrdps_datamart_behavioral_contract() -> None:
    """Hermetic behavioural assertions for HrdpsDatamartSource."""
    from datetime import UTC, datetime, timedelta

    import pandas as pd

    from microclimate.connectors.sources.hrdps_datamart import HrdpsDatamartSource

    issue_time = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)
    lead_hours = [1, 2, 3]

    ds = build_hrdps_dataset(lead_hours=(0, 1, 2, 3))

    source = HrdpsDatamartSource(opener=lambda _issue, _leads: ds)
    df = source.fetch_forecast(
        issue_time=issue_time,
        lat=51.0,
        lon=-114.0,
        lead_hours=lead_hours,
    )

    # Must validate against FORECAST_FRAME.
    FORECAST_FRAME.validate(df)

    # lead_hour column must be exactly [1, 2, 3].
    assert list(df["lead_hour"]) == lead_hours

    # valid_time == issue_time + lead_hour for every row.
    for _, row in df.iterrows():
        expected = pd.Timestamp(issue_time + timedelta(hours=int(row["lead_hour"])))
        assert row["valid_time"] == expected

    # All 8 physical variables must be non-null.
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

    # Spot-check converted value: target cell (0,0) has t2m=288.15 K → 15.0 °C.
    assert float(df["temp_c"].iloc[0]) == pytest.approx(15.0, abs=1e-6)  # type: ignore[reportUnknownMemberType]


def _assert_hrdps_caspar_behavioral_contract() -> None:
    """Hermetic behavioural assertions for HrdpsCasparSource."""
    import tempfile
    from datetime import UTC, datetime, timedelta
    from pathlib import Path

    import pandas as pd

    from microclimate.connectors.sources.hrdps_caspar import (
        CASPAR_VAR_MAP,
        HrdpsCasparSource,
        _archive_path,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
    )

    issue_time = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)
    lead_hours = [1, 2, 3]

    ds = build_hrdps_dataset(var_map=CASPAR_VAR_MAP, lead_hours=(0, 1, 2, 3))

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_root = Path(tmpdir)
        # Create an empty file at the pinned path so resolution finds it.
        expected = _archive_path(archive_root, issue_time, ".grib2")
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.touch()

        source = HrdpsCasparSource(archive_root=archive_root, opener=lambda _p: ds)
        df = source.fetch_forecast(
            issue_time=issue_time,
            lat=51.0,
            lon=-114.0,
            lead_hours=lead_hours,
        )

    # Must validate against FORECAST_FRAME.
    FORECAST_FRAME.validate(df)

    # lead_hour column must be exactly [1, 2, 3].
    assert list(df["lead_hour"]) == lead_hours

    # valid_time == issue_time + lead_hour for every row.
    for _, row in df.iterrows():
        expected_ts = pd.Timestamp(issue_time + timedelta(hours=int(row["lead_hour"])))
        assert row["valid_time"] == expected_ts

    # All 8 physical variables must be non-null.
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

    # Spot-check converted value: target cell has TT=288.15 K → 15.0 °C.
    assert float(df["temp_c"].iloc[0]) == pytest.approx(15.0, abs=1e-6)  # type: ignore[reportUnknownMemberType]


def _assert_envcanada_behavioral_contract() -> None:
    """Hermetic behavioral assertions for EnvCanadaSource."""
    from datetime import UTC, datetime

    from microclimate.connectors.sources.envcanada import EnvCanadaSource

    # ------------------------------------------------------------------
    # 1. fetch_historical — OBSERVATION_FRAME conformance + end boundary
    # ------------------------------------------------------------------
    window_csv = load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=make_fetcher(window_csv))

    start = datetime(2026, 4, 27, 0, 0, tzinfo=UTC)
    end = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)  # 12:00 UTC = 05:00 LST
    hist_df = source.fetch_historical("49268", start, end)

    # Must validate against the schema
    OBSERVATION_FRAME.validate(hist_df)

    # End boundary: no row after `end`
    assert hist_df["timestamp"].max() <= pd.Timestamp(end)

    # Start boundary: no row before `start`
    assert hist_df["timestamp"].min() >= pd.Timestamp(start)

    # cloud/solar must always be absent (ECCC CSV never carries them)
    assert hist_df["cloud_cover_fraction_present"].eq(False).all()
    assert hist_df["solar_radiation_wm2_present"].eq(False).all()
    assert hist_df["cloud_cover_fraction"].isna().all()
    assert hist_df["solar_radiation_wm2"].isna().all()

    # ------------------------------------------------------------------
    # 2. fetch_live — since boundary + OBSERVATION_FRAME conformance
    # ------------------------------------------------------------------
    live_csv = load_fixture("live_partial.csv")
    live_source = EnvCanadaSource(fetcher=make_fetcher(live_csv))

    # 2026-05-29 21:00 LST = 2026-05-30 04:00 UTC; 22:00 LST = 05:00 UTC
    # Set since between the two rows to test the filter
    since = datetime(2026, 5, 30, 5, 0, tzinfo=UTC)
    live_df = live_source.fetch_live("49268", since)

    OBSERVATION_FRAME.validate(live_df)

    # Since boundary: no row before `since`
    assert live_df["timestamp"].min() >= pd.Timestamp(since)

    # Omitted variables absent in live frame too
    assert live_df["cloud_cover_fraction_present"].eq(False).all()
    assert live_df["solar_radiation_wm2_present"].eq(False).all()
