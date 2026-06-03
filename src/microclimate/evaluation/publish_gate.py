"""Champion/challenger publish gate (L4). Imports no model classes (independence)."""

from __future__ import annotations

from typing import Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict

from microclimate.contracts.registry import Task


class _Predictor(Protocol):
    """Duck-type contract for any fitted model."""

    def predict(self, rows: pd.DataFrame) -> pd.Series[float]: ...


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    promote: bool
    reason: str
    metrics: dict[str, float]


def _error(task: Task, pred: pd.Series[float], holdout: pd.DataFrame) -> float:
    """Overall holdout error for the task: temp -> MAE, pop -> Brier (lower is better)."""
    if task == "temp":
        label: pd.Series[float] = holdout["label_temp_c"]
        keep = label.notna() & pred.notna()
        return float((pred[keep] - label[keep]).abs().mean())
    label = holdout["label_precip_occurrence"].astype("float64")
    keep = label.notna() & pred.notna()
    return float(((pred[keep] - label[keep]) ** 2).mean())


def evaluate_challenger(
    task: Task,
    challenger: _Predictor,
    champion: _Predictor | None,
    baseline: pd.Series[float],
    holdout: pd.DataFrame,
) -> GateResult:
    """Promote only if the challenger strictly beats both raw HRDPS and the incumbent.

    ``challenger``/``champion`` are fitted models exposing ``.predict(holdout) -> pd.Series``;
    ``champion`` is None when the current champion is the baseline. ``baseline`` is the
    raw-HRDPS prediction per holdout row (caller-supplied). Lower error wins (temp MAE, pop Brier).
    """
    challenger_pred: pd.Series[float] = challenger.predict(holdout)
    champion_pred: pd.Series[float] = (
        champion.predict(holdout) if champion is not None else baseline
    )

    m_chal = _error(task, challenger_pred, holdout)
    m_base = _error(task, baseline, holdout)
    m_champ = _error(task, champion_pred, holdout)

    key = "mae" if task == "temp" else "brier"
    metrics: dict[str, float] = {key: m_chal, f"baseline_{key}": m_base, f"champion_{key}": m_champ}
    metrics["mae_skill" if task == "temp" else "bss"] = (
        1.0 - m_chal / m_base if m_base > 0 else float("nan")
    )

    promote = m_chal < m_base and m_chal < m_champ
    reason = f"{key}={m_chal:.4f} vs baseline={m_base:.4f}, champion={m_champ:.4f} -> " + (
        "PROMOTE" if promote else "keep champion"
    )
    return GateResult(promote=promote, reason=reason, metrics=metrics)
