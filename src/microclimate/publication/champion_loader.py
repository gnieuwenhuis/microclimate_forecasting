"""Load the current champion model named by the registry, or None for the baseline (L5).

Shared by the training pipeline (re-evaluate the champion on the holdout) and the inference
pipeline (serve it). `publication` sits above `models` in the layer order, so importing the
model classes here is a legal downward import.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pandas as pd

from microclimate.connectors.http import http_get_bytes
from microclimate.contracts.registry import Task, manifest_key
from microclimate.models.pop_model import PrecipOccurrenceClassifier
from microclimate.models.temp_model import TemperatureRegressor
from microclimate.publication import champion_publisher as cp
from microclimate.publication.registry_store import read_registry


class _Predictor(Protocol):
    """Duck-type contract for a fitted model that scores feature-matrix rows."""

    def predict(self, rows: pd.DataFrame) -> pd.Series: ...


def load_champion(
    deployment_id: str,
    registry_path: Path,
    task: Task,
    work_dir: Path,
    *,
    fetch_bytes: Callable[[str], bytes] = lambda url: http_get_bytes(url),
) -> _Predictor | None:
    """Load the registry's current champion model for a task, or None when it's the baseline.

    None when there is no entry or the entry is the ``"baseline"`` sentinel. Otherwise downloads
    the entry's ``release_asset_url`` (via ``fetch_bytes``) and loads the task's model class.
    Raises on download/load failure (the caller decides fallback).
    """
    manifest = read_registry(registry_path)
    entry = manifest.entries.get(manifest_key(deployment_id, task))
    if entry is None or entry.version == "baseline":
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    local = work_dir / cp.asset_filename(entry.version)
    local.write_bytes(fetch_bytes(entry.release_asset_url))
    if task == "temp":
        return TemperatureRegressor.load(local)
    return PrecipOccurrenceClassifier.load(local)
