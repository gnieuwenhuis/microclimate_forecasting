from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from microclimate.contracts.feature_matrix import FEATURE_ROW, FEATURE_SCHEMA_VERSION


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_schema_version": [FEATURE_SCHEMA_VERSION],
            "deployment_id": ["lethbridge"],
            "issue_time": pd.to_datetime(["2026-05-30T00:00:00Z"]),
            "lead_hour": [1],
            "valid_time": pd.to_datetime(["2026-05-30T01:00:00Z"]),
            "nwp_temp_c": [11.2],  # dynamic feature column — allowed
        }
    )


def test_feature_schema_version_is_a_string() -> None:
    assert isinstance(FEATURE_SCHEMA_VERSION, str)
    assert FEATURE_SCHEMA_VERSION


def test_accepts_identity_plus_dynamic_feature_columns() -> None:
    FEATURE_ROW.validate(_valid_frame())


def test_rejects_lead_hour_out_of_range() -> None:
    frame = _valid_frame()
    frame["lead_hour"] = [49]
    with pytest.raises(pandera.errors.SchemaError):
        FEATURE_ROW.validate(frame)


def test_has_no_label_columns() -> None:
    # Feature matrix is label-free (scope A); labels are attached downstream.
    assert "label_temp_c" not in FEATURE_ROW.columns
    assert "label_precip_occurrence" not in FEATURE_ROW.columns
