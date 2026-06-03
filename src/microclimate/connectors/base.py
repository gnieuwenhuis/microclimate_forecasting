"""Source abstractions and the dual-feed contract (L2, ADR-0002/0008)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import pandas as pd

HistoricalCoverage = Literal["deep", "shallow", "none"]


# ---------------------------------------------------------------------------
# Typed connector exceptions
# ---------------------------------------------------------------------------


class ConnectorError(Exception):
    """Base class for all connector-level errors."""


class SourceUnavailable(ConnectorError):
    """Raised when a data source is unreachable or returns an unexpected error."""


class ForecastUnavailable(ConnectorError):
    """Raised when a forecast cannot be retrieved for the requested issue_time/location."""


class StationNotFound(ConnectorError):
    """Raised when the requested station_id does not exist in the source."""


class Source(ABC):  # noqa: B024
    """Common base for every data connector."""


class NWPSource(Source):
    @property
    @abstractmethod
    def is_live(self) -> bool: ...

    @abstractmethod
    def fetch_forecast(
        self, issue_time: datetime, lat: float, lon: float, lead_hours: Sequence[int]
    ) -> pd.DataFrame:
        """Return a FORECAST_FRAME for the requested leads.

        May return the available **contiguous lead prefix** (≤ ``lead_hours``) when later leads
        are unavailable — e.g. beyond the model's reach; callers treat fewer leads as a truncated
        horizon. Raise ``ForecastUnavailable`` only when no lead is available.
        """
        ...


class ObservationSource(Source):
    @property
    @abstractmethod
    def historical_coverage(self) -> HistoricalCoverage: ...

    @abstractmethod
    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame: ...

    @abstractmethod
    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame: ...
