from __future__ import annotations

import microclimate.connectors  # noqa: F401  # type: ignore[reportUnusedImport]
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.registry import get_source, registered_keys


def test_all_v1_sources_registered() -> None:
    assert {"openmeteo", "envcanada", "acis"} <= registered_keys()


def test_nwp_sources_typed() -> None:
    openmeteo = get_source("openmeteo")
    assert isinstance(openmeteo, NWPSource)
    assert openmeteo.is_live is True


def test_observation_sources_are_deep() -> None:
    for key in ("envcanada", "acis"):
        source = get_source(key)
        assert isinstance(source, ObservationSource)
        assert source.historical_coverage == "deep"
