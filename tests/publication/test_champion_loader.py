from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from microclimate.contracts.registry import RegistryEntry, RegistryManifest, manifest_key
from microclimate.models.temp_model import TemperatureRegressor
from microclimate.publication import champion_publisher as cp
from microclimate.publication.champion_loader import load_champion
from microclimate.publication.registry_store import write_registry


def _fit_tiny_temp() -> TemperatureRegressor:
    rows = pd.DataFrame(
        {
            "feature_schema_version": ["1.0.0"] * 4,
            "lead_hour": [1, 2, 3, 4],
            "label_temp_c": [1.0, 2.0, 3.0, 4.0],
            "nwp_temp_c": [1.0, 2.0, 3.0, 4.0],
        }
    )
    m = TemperatureRegressor()
    m.fit(rows)
    return m


def test_load_champion_none_when_no_entry(tmp_path: Path) -> None:
    write_registry(RegistryManifest(), tmp_path / "registry.json")
    assert load_champion("lethbridge", tmp_path / "registry.json", "temp", tmp_path / "wd") is None


def test_load_champion_none_when_baseline_entry(tmp_path: Path) -> None:
    entry = RegistryEntry(
        version="baseline",
        release_asset_url="x",
        promoted_at=datetime(2026, 6, 3, tzinfo=UTC),
        holdout_metrics={},
    )
    m = RegistryManifest(entries={manifest_key("lethbridge", "temp"): entry})
    write_registry(m, tmp_path / "registry.json")
    assert load_champion("lethbridge", tmp_path / "registry.json", "temp", tmp_path / "wd") is None


def test_load_champion_downloads_and_loads(tmp_path: Path) -> None:
    version = "lethbridge-temp-20260603T0000Z"
    asset = cp.save_champion(_fit_tiny_temp(), tmp_path / "stage", version)
    raw = asset.read_bytes()
    entry = RegistryEntry(
        version=version,
        release_asset_url=f"https://example/{version}.joblib",
        promoted_at=datetime(2026, 6, 3, tzinfo=UTC),
        holdout_metrics={"mae": 1.0},
    )
    m = RegistryManifest(entries={manifest_key("lethbridge", "temp"): entry})
    write_registry(m, tmp_path / "registry.json")

    loaded = load_champion(
        "lethbridge",
        tmp_path / "registry.json",
        "temp",
        tmp_path / "wd",
        fetch_bytes=lambda _url: raw,
    )
    assert loaded is not None
    assert isinstance(loaded, TemperatureRegressor)
