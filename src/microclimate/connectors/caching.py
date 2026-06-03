"""In-memory caching wrapper for observation sources during bulk assembly/backfill (L2).

Both the seed backfill and the training-data assembly iterate many issue-times and would
otherwise re-fetch the same observation history for every issue-time (and re-download the same
monthly CSV repeatedly). ``CachingObservationSource`` prefetches each station's full window once
and serves as-of slices from memory, so net network volume becomes ~one obs fetch per station
plus the NWP fetches. Historical-read only — ``fetch_live`` is unsupported (bulk paths never serve
live inference).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from microclimate.connectors.base import HistoricalCoverage, ObservationSource


class CachingObservationSource(ObservationSource):
    """Prefetch each station's full window once; serve ``fetch_historical`` slices from memory."""

    def __init__(
        self,
        inner: ObservationSource,
        station_ids: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> None:
        self._inner = inner
        self._cache: dict[str, pd.DataFrame] = {
            sid: inner.fetch_historical(sid, start, end) for sid in station_ids
        }

    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return self._inner.historical_coverage

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        df = self._cache.get(station_id)
        if df is None:
            return self._inner.fetch_historical(station_id, start, end)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return df[(df["timestamp"] >= s) & (df["timestamp"] <= e)].reset_index(drop=True)

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError("CachingObservationSource is historical-read only")
