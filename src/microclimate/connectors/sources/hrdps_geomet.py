"""HRDPS via MSC GeoMet/Datamart — live NWP source (stub)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from microclimate.connectors.base import NWPSource
from microclimate.connectors.registry import register_source


@register_source("hrdps_geomet")
class HrdpsGeoMetSource(NWPSource):
    @property
    def is_live(self) -> bool:
        return True

    def fetch_forecast(
        self, issue_time: datetime, lat: float, lon: float, lead_hours: Sequence[int]
    ) -> pd.DataFrame:
        raise NotImplementedError
