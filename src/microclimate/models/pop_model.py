"""Precipitation-occurrence classifier wrapper with calibration (L4, stub)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from microclimate.contracts.snapshot import FeatureSnapshot


class PrecipOccurrenceClassifier:
    version: str = "0.0.0"

    def fit(self, rows: pd.DataFrame) -> None:
        raise NotImplementedError

    def calibrate(self, rows: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, snapshot: FeatureSnapshot) -> dict[int, float]:
        """Return {lead_hour: calibrated_pop in [0, 1]}."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> PrecipOccurrenceClassifier:
        raise NotImplementedError
