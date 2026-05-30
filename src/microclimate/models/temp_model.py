"""Temperature regressor wrapper (L4, stub). lead_hour is a feature."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from microclimate.contracts.snapshot import FeatureSnapshot


class TemperatureRegressor:
    version: str = "0.0.0"

    def fit(self, rows: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, snapshot: FeatureSnapshot) -> dict[int, float]:
        """Return {lead_hour: temperature_c}."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> TemperatureRegressor:
        raise NotImplementedError
