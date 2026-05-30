"""Write a ForecastDocument to JSON — only through the validated model (L5, stub)."""

from __future__ import annotations

from pathlib import Path

from microclimate.contracts.forecast import ForecastDocument


def write_forecast(doc: ForecastDocument, path: Path) -> None:
    raise NotImplementedError
