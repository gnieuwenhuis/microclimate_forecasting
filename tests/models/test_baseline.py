from __future__ import annotations

import pandas as pd

from microclimate.models.baseline import BASELINE_VERSION, baseline_predictions


def test_baseline_temp_passthrough_and_pop_threshold() -> None:
    rows = pd.DataFrame(
        {
            "lead_hour": [1, 2, 3],
            "nwp_temp_c": [10.0, 11.0, 12.0],
            "nwp_precip_mm": [0.0, 0.2, 0.5],  # threshold 0.2: [no, yes(inclusive), yes]
        }
    )
    out = baseline_predictions(rows, threshold_mm=0.2)
    assert list(out["pred_temp_c"]) == [10.0, 11.0, 12.0]
    assert list(out["pred_pop"]) == [0.0, 1.0, 1.0]
    # original columns preserved (reshape needs lead_hour/valid_time downstream)
    assert "lead_hour" in out.columns


def test_baseline_version_constant() -> None:
    assert BASELINE_VERSION == "baseline"
