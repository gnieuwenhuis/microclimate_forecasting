"""Per-lead-hour skill metrics relative to a baseline (L4, stub)."""

from __future__ import annotations

import pandas as pd


def mae_skill(
    predictions: pd.DataFrame, baseline: pd.DataFrame, truth: pd.DataFrame
) -> dict[int, float]:
    """Temperature MAE skill vs baseline, keyed by lead_hour."""
    raise NotImplementedError


def brier_skill(
    predictions: pd.DataFrame, baseline: pd.DataFrame, truth: pd.DataFrame
) -> dict[int, float]:
    """PoP Brier skill vs baseline, keyed by lead_hour."""
    raise NotImplementedError
