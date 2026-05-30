from __future__ import annotations

import pathlib
from datetime import UTC

import pandas as pd
import pytest

import microclimate.connectors  # noqa: F401  # type: ignore[reportUnusedImport]  (populates the registry)
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.registry import get_source, registered_keys
from microclimate.contracts.observation import OBSERVATION_FRAME

_KEYS = sorted(k for k in registered_keys() if not k.startswith("_"))

_FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "envcanada"

# ---------------------------------------------------------------------------
# Sources whose fetch_* is not yet implemented — skip gracefully.
# ---------------------------------------------------------------------------
_NOT_YET_IMPLEMENTED: frozenset[str] = frozenset({"acis", "hrdps_geomet", "hrdps_caspar"})


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
# Behavioural contract — hermetic for envcanada; skip the rest.
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> str:
    return (_FIXTURE_DIR / name).read_text(encoding="utf-8-sig")


def _make_fetcher(csv_text: str):
    def fetcher(station_id: str, year: int, month: int) -> str:  # noqa: ARG001
        return csv_text

    return fetcher


@pytest.mark.parametrize("key", _KEYS)
def test_source_behavioral_contract(key: str) -> None:
    """Behavioural assertions for observation sources.

    * envcanada: driven hermetically with fixture-backed fetcher.
    * acis / hrdps_geomet / hrdps_caspar: skipped until fetch_* is implemented.
    """
    if key in _NOT_YET_IMPLEMENTED:
        pytest.skip(f"{key}: fetch_* not yet implemented")

    if key == "envcanada":
        _assert_envcanada_behavioral_contract()
        return

    # If a new key is added without being covered here, fail loudly.
    pytest.fail(f"No behavioral contract assertion defined for source key {key!r}")


def _assert_envcanada_behavioral_contract() -> None:
    """Hermetic behavioral assertions for EnvCanadaSource."""
    from datetime import datetime

    from microclimate.connectors.sources.envcanada import EnvCanadaSource

    # ------------------------------------------------------------------
    # 1. fetch_historical — OBSERVATION_FRAME conformance + end boundary
    # ------------------------------------------------------------------
    window_csv = _load_fixture("hourly_window.csv")
    source = EnvCanadaSource(fetcher=_make_fetcher(window_csv))

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
    live_csv = _load_fixture("live_partial.csv")
    live_source = EnvCanadaSource(fetcher=_make_fetcher(live_csv))

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
