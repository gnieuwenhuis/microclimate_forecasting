from __future__ import annotations

import pandas as pd
import pytest

from microclimate.evaluation.publish_gate import GateResult, evaluate_challenger


def test_gate_result_shape() -> None:
    result = GateResult(promote=False, reason="stub", metrics={})
    assert result.promote is False


def test_evaluate_challenger_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        evaluate_challenger(
            task="temp",
            challenger=object(),
            champion=None,
            baseline=pd.DataFrame(),
            holdout=pd.DataFrame(),
        )
