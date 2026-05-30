from __future__ import annotations

import pytest

from microclimate.connectors.base import NWPSource, ObservationSource


def test_incomplete_observation_source_cannot_instantiate() -> None:
    class Broken(ObservationSource):
        @property
        def historical_coverage(self) -> str:  # type: ignore[override]
            return "deep"

        # missing fetch_historical / fetch_live

    with pytest.raises(TypeError):
        Broken()  # type: ignore[abstract]


def test_incomplete_nwp_source_cannot_instantiate() -> None:
    class Broken(NWPSource):
        @property
        def is_live(self) -> bool:
            return True

        # missing fetch_forecast

    with pytest.raises(TypeError):
        Broken()  # type: ignore[abstract]
