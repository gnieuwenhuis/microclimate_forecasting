from __future__ import annotations

import pandas as pd

from microclimate.contracts.training_store import TRAINING_ROW


def test_training_row_accepts_extra_feature_columns() -> None:
    frame = pd.DataFrame(
        {
            "schema_version": ["1"],
            "deployment_id": ["lethbridge"],
            "issue_time": pd.to_datetime(["2026-05-30T00:00:00Z"]),
            "lead_hour": [1],
            "valid_time": pd.to_datetime(["2026-05-30T01:00:00Z"]),
            "label_temp_c": [11.0],
            "label_precip_occurrence": [0],
            "nwp_t2m_lead1": [11.2],  # dynamic feature column — allowed
        }
    )
    TRAINING_ROW.validate(frame)
