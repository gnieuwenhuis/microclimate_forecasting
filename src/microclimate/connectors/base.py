"""Source abstractions and the dual-feed contract (L2, ADR-0002/0008)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import pandas as pd

HistoricalCoverage = Literal["deep", "shallow", "none"]


class Source(ABC):  # noqa: B024
    """Common base for every data connector."""


class NWPSource(Source):
    @property
    @abstractmethod
    def is_live(self) -> bool: ...

    @abstractmethod
    def fetch_forecast(
        self, issue_time: datetime, lat: float, lon: float, lead_hours: Sequence[int]
    ) -> pd.DataFrame: ...


class ObservationSource(Source):
    @property
    @abstractmethod
    def historical_coverage(self) -> HistoricalCoverage: ...

    @abstractmethod
    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame: ...

    @abstractmethod
    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame: ...
