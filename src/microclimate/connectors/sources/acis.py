"""Alberta Climate Information Service — deep dual-feed (stub)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from microclimate.connectors.base import HistoricalCoverage, ObservationSource
from microclimate.connectors.registry import register_source


@register_source("acis")
class AcisSource(ObservationSource):
    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError
