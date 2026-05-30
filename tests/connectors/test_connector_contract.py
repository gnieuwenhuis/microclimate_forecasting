from __future__ import annotations

import pytest

import microclimate.connectors  # noqa: F401  # type: ignore[reportUnusedImport]  (populates the registry)
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.registry import get_source, registered_keys

_KEYS = sorted(k for k in registered_keys() if not k.startswith("_"))


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


@pytest.mark.skip(reason="behavioral checks added when fetch_* is implemented")
@pytest.mark.parametrize("key", _KEYS)
def test_source_behavioral_contract(key: str) -> None:
    # Future: assert OBSERVATION_FRAME conformance, the <= issue_time boundary, masks on
    # missing data, and that declared historical_coverage matches a real probe window.
    raise NotImplementedError
