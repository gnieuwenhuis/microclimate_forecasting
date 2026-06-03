"""NaN obs/static features must survive the JSON round-trip (regression for store read)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION, FeatureSnapshot


def test_nan_features_survive_json_roundtrip() -> None:
    snap = FeatureSnapshot(
        deployment_id="lethbridge",
        issue_time=datetime(2024, 6, 1, 0, tzinfo=UTC),
        nwp_features={"nwp_temp_c_h1": 10.0},
        observation_features={
            "obs_8804_cloud_cover_fraction_lag3": float("nan"),
            "obs_x_temp_c_lag0": 5.0,
        },
        observation_masks={"obs_8804_cloud_cover_fraction_lag3": False, "obs_x_temp_c_lag0": True},
        static_features={"static_lat": 49.7, "static_elevation_m": float("nan")},
        temporal_features={"t0_hour_sin": 0.0},
        lead_hours=(1,),
        schema_version=SNAPSHOT_SCHEMA_VERSION,
    )
    restored = FeatureSnapshot.model_validate_json(snap.model_dump_json())
    assert math.isnan(restored.observation_features["obs_8804_cloud_cover_fraction_lag3"])
    assert math.isnan(restored.static_features["static_elevation_m"])
    assert restored.observation_features["obs_x_temp_c_lag0"] == 5.0
    assert restored.observation_masks["obs_8804_cloud_cover_fraction_lag3"] is False
