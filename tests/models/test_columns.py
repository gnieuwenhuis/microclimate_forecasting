# tests/models/test_columns.py
from __future__ import annotations

import pandas as pd

from microclimate.models._columns import NON_FEATURE_COLUMNS, feature_columns


def test_feature_columns_excludes_ids_times_labels_keeps_lead_hour() -> None:
    df = pd.DataFrame(
        columns=[
            "feature_schema_version",
            "deployment_id",
            "issue_time",
            "valid_time",
            "label_temp_c",
            "label_precip_occurrence",
            "lead_hour",
            "nwp_temp_c",
            "obs_T1_temp_c_lag0",
        ]
    )
    feats = feature_columns(df)
    assert "lead_hour" in feats
    assert "nwp_temp_c" in feats
    assert "obs_T1_temp_c_lag0" in feats
    assert NON_FEATURE_COLUMNS.isdisjoint(feats)
