from __future__ import annotations

import pandas as pd
import pytest

import microclimate.connectors  # noqa: F401  # type: ignore[reportUnusedImport]  (populates the registry)
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.registry import get_source, registered_keys
from microclimate.contracts.forecast_frame import FORECAST_FRAME
from microclimate.contracts.observation import OBSERVATION_FRAME

from .conftest import load_fixture, make_fetcher

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
# Behavioural contract — hermetic for envcanada + openmeteo; skip the rest.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", _KEYS)
def test_source_behavioral_contract(key: str) -> None:
    """Behavioural assertions for observation sources.

    * envcanada: driven hermetically with fixture-backed fetcher.
    * openmeteo: driven hermetically with fixture-backed fetcher.
    * acis: skipped until fetch_* is implemented.
    """
    if key in _NOT_YET_IMPLEMENTED:
        pytest.skip(f"{key}: fetch_* not yet implemented")

    if key == "envcanada":
        _assert_envcanada_behavioral_contract()
        return

    if key == "openmeteo":
        _assert_openmeteo_behavioral_contract()
        return

    pytest.fail(f"No behavioral contract assertion defined for source key {key!r}")


def _assert_openmeteo_behavioral_contract() -> None:
    """Hermetic behavioural assertions for OpenMeteoSource (fixture-backed fetcher)."""
    import json
    from datetime import UTC, datetime
    from pathlib import Path
    from typing import Any

    from microclimate.connectors.sources.openmeteo import OpenMeteoSource

    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "openmeteo_historical.json").read_text()
    )

    def _fixed_fetcher(url: str, *, params: Any = None) -> str:  # noqa: ARG001
        return json.dumps(payload)

    source = OpenMeteoSource(
        fetcher=_fixed_fetcher,
        now=datetime(2026, 6, 2, 12, 0, tzinfo=UTC),
    )
    df = source.fetch_forecast(datetime(2024, 6, 1, 0, 0, tzinfo=UTC), 49.70, -112.77, [1, 2, 3])
    FORECAST_FRAME.validate(df)
    assert list(df["lead_hour"]) == [1, 2, 3]


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
