from __future__ import annotations

import pandas as pd

from microclimate.evaluation.publish_gate import evaluate_challenger


class _ConstTemp:
    def __init__(self, value: float) -> None:
        self._v = value

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        return pd.Series([self._v] * len(rows), index=rows.index, name="pred_temp_c")


def _temp_holdout() -> tuple[pd.DataFrame, pd.Series]:
    holdout = pd.DataFrame({"lead_hour": [1, 2, 3, 4], "label_temp_c": [10.0, 10.0, 10.0, 10.0]})
    baseline = pd.Series([12.0, 12.0, 12.0, 12.0], index=holdout.index)
    return holdout, baseline


def test_temp_promotes_when_strictly_beats_baseline_and_champion() -> None:
    holdout, baseline = _temp_holdout()
    res = evaluate_challenger("temp", _ConstTemp(10.5), _ConstTemp(11.0), baseline, holdout)
    assert res.promote is True
    assert res.metrics["mae"] < res.metrics["champion_mae"] < res.metrics["baseline_mae"]
    assert "mae_skill" in res.metrics


def test_temp_no_promote_when_worse_than_champion() -> None:
    holdout, baseline = _temp_holdout()
    res = evaluate_challenger("temp", _ConstTemp(11.5), _ConstTemp(11.0), baseline, holdout)
    assert res.promote is False


def test_temp_no_promote_on_tie() -> None:
    holdout, baseline = _temp_holdout()
    res = evaluate_challenger("temp", _ConstTemp(12.0), None, baseline, holdout)
    assert res.promote is False


def test_temp_promotes_off_baseline_when_champion_none() -> None:
    holdout, baseline = _temp_holdout()
    res = evaluate_challenger("temp", _ConstTemp(10.0), None, baseline, holdout)
    assert res.promote is True


class _ConstPop:
    def __init__(self, p: float) -> None:
        self._p = p

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        return pd.Series([self._p] * len(rows), index=rows.index, name="pred_pop")


def test_pop_no_promote_when_brier_not_better_than_baseline() -> None:
    holdout = pd.DataFrame({"lead_hour": [1, 2, 3, 4], "label_precip_occurrence": [1, 0, 1, 0]})
    baseline = pd.Series([0.5, 0.5, 0.5, 0.5], index=holdout.index)  # Brier 0.25
    # _ConstPop(0.6): Brier = ((.6-1)^2*2 + (.6-0)^2*2)/4 = (0.32+0.72)/4 = 0.26 > 0.25
    res = evaluate_challenger("pop", _ConstPop(0.6), None, baseline, holdout)
    assert res.promote is False


def test_pop_promotes_when_brier_beats_baseline() -> None:
    holdout = pd.DataFrame({"lead_hour": [1, 2, 3, 4], "label_precip_occurrence": [1, 0, 1, 0]})
    baseline = pd.Series([0.5, 0.5, 0.5, 0.5], index=holdout.index)  # Brier 0.25

    # _ConstPop(0.4): Brier = ((.4-1)^2*2 + (.4-0)^2*2)/4 = (0.72+0.32)/4 = 0.26 ... still >0.25
    # use a genuinely better predictor: predict the label exactly via a per-row stub
    class _Perfect:
        def predict(self, rows: pd.DataFrame) -> pd.Series:
            return rows["label_precip_occurrence"].astype(float).rename("pred_pop")

    res = evaluate_challenger("pop", _Perfect(), None, baseline, holdout)
    assert res.promote is True and res.metrics["brier"] == 0.0


def test_temp_no_promote_when_holdout_labels_all_nan() -> None:
    import numpy as np

    holdout = pd.DataFrame({"lead_hour": [1, 2], "label_temp_c": [np.nan, np.nan]})
    baseline = pd.Series([12.0, 12.0], index=holdout.index)
    res = evaluate_challenger("temp", _ConstTemp(10.0), None, baseline, holdout)
    assert res.promote is False  # nan error -> all comparisons False -> fail-safe
