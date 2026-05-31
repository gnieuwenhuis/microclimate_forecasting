from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from microclimate.contracts.snapshot import FeatureSnapshot


def test_valid_snapshot_constructs() -> None:
    snap = FeatureSnapshot(
        deployment_id="lethbridge",
        issue_time=datetime(2026, 5, 30, tzinfo=UTC),
        nwp_features={"t2m_lead1": 11.0},
        observation_features={"target_temp_lag1": 10.5},
        observation_masks={"target_temp_lag1": True},
        static_features={"lat": 49.68872},
        temporal_features={"hour_sin": 0.0},
        lead_hours=(1, 2, 3),
        schema_version="1",
    )
    assert snap.deployment_id == "lethbridge"
    assert snap.lead_hours == (1, 2, 3)


def test_snapshot_schema_version_constant() -> None:
    from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION

    assert isinstance(SNAPSHOT_SCHEMA_VERSION, str)
    assert SNAPSHOT_SCHEMA_VERSION  # non-empty


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        FeatureSnapshot(
            deployment_id="lethbridge",
            issue_time=datetime(2026, 5, 30),  # naive — no tzinfo
            nwp_features={},
            observation_features={},
            observation_masks={},
            static_features={},
            temporal_features={},
            lead_hours=(1,),
            schema_version="1",
        )
