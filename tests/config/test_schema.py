from __future__ import annotations

import pytest
from pydantic import ValidationError

from microclimate.config.schema import DeploymentConfig


def _raw() -> dict[str, object]:
    return {
        "deployment_id": "demo",
        "target": {
            "station_id": "9835",
            "connector_key": "acis",
            "lat": 49.68872,
            "lon": -112.74494,
            "elevation_m": 903,
        },
        "neighbors": [
            {
                "station_id": "3033875",
                "connector_key": "envcanada",
                "lat": 49.6303,
                "lon": -112.7989,
                "elevation_m": None,
            }
        ],
        "enabled_sources": ["hrdps_datamart", "hrdps_caspar", "envcanada", "acis"],
        "nwp": {
            "product": "hrdps",
            "live_connector": "hrdps_datamart",
            "historical_connector": "hrdps_caspar",
            "sampling": "nearest_grid_cell",
        },
        "horizon_hours": 48,
        "lag_hours": 6,
        "feature_groups": {"nwp": True, "observations": True},
        "label": {"precip_occurrence_threshold_mm": 0.2},
        "training": {"seed": {"source": "caspar", "start": "2017-05-22"}, "holdout_months": 12},
        "output": {"forecast_json": "forecasts/demo.json"},
    }


def test_valid_config() -> None:
    config = DeploymentConfig.model_validate(_raw())
    assert config.target.connector_key == "acis"
    assert config.neighbors[0].elevation_m is None


def test_unknown_key_rejected() -> None:
    raw = _raw()
    raw["training_strategy"] = "seeded"  # removed field — must be rejected
    with pytest.raises(ValidationError):
        DeploymentConfig.model_validate(raw)
