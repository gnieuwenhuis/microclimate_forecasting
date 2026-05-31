from __future__ import annotations

import pandas as pd

from microclimate.evaluation.metrics import (
    nwp_pop_baseline,
    pop_skill_by_lead,
    reliability_table,
    temp_skill_by_lead,
)


def test_nwp_pop_baseline_thresholds_inclusively() -> None:
    df = pd.DataFrame({"nwp_precip_mm": [0.0, 0.2, 0.5]})
    out = nwp_pop_baseline(df, threshold_mm=0.2)
    assert list(out) == [0.0, 1.0, 1.0]  # >= threshold is occurrence


def test_temp_skill_by_lead() -> None:
    df = pd.DataFrame(
        {
            "lead_hour": [1, 1],
            "pred_temp_c": [1.0, 1.0],
            "label_temp_c": [2.0, 2.0],
            "nwp_temp_c": [0.0, 0.0],  # baseline is twice as wrong
        }
    )
    out = temp_skill_by_lead(df).set_index("lead_hour")
    assert out.loc[1, "mae"] == 1.0
    assert out.loc[1, "rmse"] == 1.0
    assert out.loc[1, "baseline_mae"] == 2.0
    assert out.loc[1, "baseline_rmse"] == 2.0
    assert out.loc[1, "skill"] == 0.5  # MAE skill: 1 - 1/2
    assert out.loc[1, "n"] == 2


def test_pop_skill_by_lead() -> None:
    df = pd.DataFrame(
        {
            "lead_hour": [1, 1],
            "pred_pop": [0.5, 0.5],
            "label_precip_occurrence": [1, 0],
            "baseline_pop": [0.0, 0.0],
        }
    )
    out = pop_skill_by_lead(df).set_index("lead_hour")
    assert out.loc[1, "brier"] == 0.25
    assert out.loc[1, "baseline_brier"] == 0.5
    assert out.loc[1, "bss"] == 0.5


def test_reliability_table_bins() -> None:
    df = pd.DataFrame({"pred_pop": [0.05, 0.95], "label_precip_occurrence": [0, 1]})
    out = reliability_table(df, n_bins=10)
    assert len(out) == 10
    first = out.iloc[0]
    last = out.iloc[-1]
    assert first["count"] == 1 and first["observed_freq"] == 0.0
    assert last["count"] == 1 and last["observed_freq"] == 1.0
    assert abs(first["mean_pred"] - 0.05) < 1e-9
    assert abs(last["mean_pred"] - 0.95) < 1e-9
