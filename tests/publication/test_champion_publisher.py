from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from microclimate.models.temp_model import TemperatureRegressor
from microclimate.publication.champion_publisher import (
    asset_filename,
    champion_version,
    release_asset_url,
    release_tag,
    save_champion,
)


def test_version_and_url_are_deterministic() -> None:
    t = datetime(2026, 6, 3, 14, 5, tzinfo=UTC)
    v = champion_version("lethbridge", "temp", t)
    assert v == "lethbridge-temp-20260603T1405Z"
    assert release_tag(v) == "champion-lethbridge-temp-20260603T1405Z"
    assert asset_filename(v) == "lethbridge-temp-20260603T1405Z.joblib"
    url = release_asset_url("gnieuwenhuis/microclimate_forecasting", v)
    assert url == (
        "https://github.com/gnieuwenhuis/microclimate_forecasting/releases/download/"
        "champion-lethbridge-temp-20260603T1405Z/lethbridge-temp-20260603T1405Z.joblib"
    )


def test_save_champion_writes_loadable_file(tmp_path: Path) -> None:
    model = _fit_tiny_temp_model()
    path = save_champion(model, tmp_path, "lethbridge-temp-20260603T1405Z")
    assert path == tmp_path / "lethbridge-temp-20260603T1405Z.joblib"
    TemperatureRegressor.load(path)  # must round-trip


def _fit_tiny_temp_model() -> TemperatureRegressor:
    # feature_columns() returns every column not in NON_FEATURE_COLUMNS.
    # TemperatureRegressor.fit() needs:
    #   - a "feature_schema_version" column (single unique value)
    #   - at least one feature column (here: "lead_hour")
    #   - a "label_temp_c" column with ≥1 non-null value
    rows = pd.DataFrame(
        {
            "feature_schema_version": ["v1"],
            "lead_hour": [1],
            "label_temp_c": [15.0],
        }
    )
    model = TemperatureRegressor()
    model.fit(rows)
    return model
