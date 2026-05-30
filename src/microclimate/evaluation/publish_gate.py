"""Champion/challenger publish gate (L4, stub). Imports no model classes (independence)."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from microclimate.contracts.registry import Task


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    promote: bool
    reason: str
    metrics: dict[str, float]


def evaluate_challenger(
    task: Task,
    challenger: object,
    champion: object | None,
    baseline: pd.DataFrame,
    holdout: pd.DataFrame,
) -> GateResult:
    """Promote only if the challenger beats both raw HRDPS and the incumbent."""
    raise NotImplementedError
