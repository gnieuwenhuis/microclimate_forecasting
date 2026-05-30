from __future__ import annotations

from datetime import datetime
from collections.abc import Sequence

import pandas as pd
import pytest

from microclimate.config.schema import (
    DeploymentConfig,
    FeatureGroupSwitches,
    LabelConfig,
    NwpConfig,
    OutputConfig,
    SeedConfig,
    StationRef,
    TrainingConfig,
)
from microclimate.connectors.base import HistoricalCoverage, NWPSource, ObservationSource
from microclimate.connectors.registry import (
    get_source,
    register_source,
    registered_keys,
    validate_config_sources,
)


@register_source("_test_nwp")
class _TestNwp(NWPSource):  # type: ignore[reportUnusedClass]
    @property
    def is_live(self) -> bool:
        return True

    def fetch_forecast(
        self, issue_time: datetime, lat: float, lon: float, lead_hours: Sequence[int]
    ) -> pd.DataFrame:
        raise NotImplementedError


@register_source("_test_deep")
class _TestDeep(ObservationSource):  # type: ignore[reportUnusedClass]
    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError


@register_source("_test_shallow")
class _TestShallow(ObservationSource):  # type: ignore[reportUnusedClass]
    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "shallow"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError


def _config(target_key: str, sources: list[str]) -> DeploymentConfig:
    return DeploymentConfig(
        deployment_id="t",
        target=StationRef(station_id="s", connector_key=target_key, lat=0.0, lon=0.0),
        neighbors=[],
        enabled_sources=sources,
        nwp=NwpConfig(
            product="hrdps",
            live_connector="_test_nwp",
            historical_connector="_test_nwp",
            sampling="nearest_grid_cell",
        ),
        lag_hours=6,
        feature_groups=FeatureGroupSwitches(nwp=True, observations=True),
        label=LabelConfig(precip_occurrence_threshold_mm=0.2),
        training=TrainingConfig(seed=SeedConfig(source="caspar", start="2017-05-22"), holdout_months=12),
        output=OutputConfig(forecast_json="forecasts/t.json"),
    )


def test_registered_and_lookup() -> None:
    assert "_test_deep" in registered_keys()
    assert isinstance(get_source("_test_deep"), ObservationSource)


def test_duplicate_key_rejected() -> None:
    with pytest.raises(ValueError):

        @register_source("_test_deep")
        class _Dupe(ObservationSource):  # type: ignore[reportUnusedClass]
            @property
            def historical_coverage(self) -> HistoricalCoverage:
                return "deep"

            def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
                raise NotImplementedError

            def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
                raise NotImplementedError


def test_deep_source_passes_validation() -> None:
    validate_config_sources(_config("_test_deep", ["_test_nwp", "_test_deep"]))


def test_unregistered_source_rejected() -> None:
    with pytest.raises(ValueError):
        validate_config_sources(_config("_test_deep", ["_test_nwp", "_test_deep", "ghost"]))


def test_non_deep_target_rejected() -> None:
    with pytest.raises(ValueError):
        validate_config_sources(_config("_test_shallow", ["_test_nwp", "_test_shallow"]))
