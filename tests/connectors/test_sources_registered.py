from __future__ import annotations

import microclimate.connectors  # noqa: F401  # type: ignore[reportUnusedImport]
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.registry import get_source, registered_keys


def test_all_v1_sources_registered() -> None:
    assert {"hrdps_geomet", "hrdps_caspar", "envcanada", "acis"} <= registered_keys()


def test_nwp_sources_typed() -> None:
    geomet = get_source("hrdps_geomet")
    caspar = get_source("hrdps_caspar")
    assert isinstance(geomet, NWPSource)
    assert isinstance(caspar, NWPSource)
    assert geomet.is_live is True
    assert caspar.is_live is False


def test_observation_sources_are_deep() -> None:
    for key in ("envcanada", "acis"):
        source = get_source(key)
        assert isinstance(source, ObservationSource)
        assert source.historical_coverage == "deep"
