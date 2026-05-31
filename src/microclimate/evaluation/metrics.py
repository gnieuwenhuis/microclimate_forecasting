"""Forecast skill metrics vs the raw-HRDPS baseline + PoP reliability (L5, pure).

Per-lead-hour aggregation (CONTEXT.md: metrics reported per lead hour). Shared by the
notebook now and the publish gate later. No models import (sibling independence).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def temp_skill_by_lead(
    df: pd.DataFrame,
    *,
    pred_col: str = "pred_temp_c",
    label_col: str = "label_temp_c",
    baseline_col: str = "nwp_temp_c",
) -> pd.DataFrame:
    """Per-lead MAE/RMSE and RMSE skill vs baseline. skill = 1 - rmse / baseline_rmse."""
    d = df.dropna(subset=[pred_col, label_col, baseline_col]).copy()
    d["_ae"] = (d[pred_col] - d[label_col]).abs()
    d["_se"] = (d[pred_col] - d[label_col]) ** 2
    d["_bse"] = (d[baseline_col] - d[label_col]) ** 2
    g = d.groupby("lead_hour")
    out = pd.DataFrame(
        {
            "mae": g["_ae"].mean(),
            "rmse": np.sqrt(g["_se"].mean()),
            "baseline_rmse": np.sqrt(g["_bse"].mean()),
            "n": g.size(),
        }
    ).reset_index()
    out["skill"] = 1.0 - out["rmse"] / out["baseline_rmse"]
    return out


def pop_skill_by_lead(
    df: pd.DataFrame,
    *,
    prob_col: str = "pred_pop",
    label_col: str = "label_precip_occurrence",
    baseline_col: str = "baseline_pop",
) -> pd.DataFrame:
    """Per-lead Brier score and Brier Skill Score vs baseline. bss = 1 - brier/baseline_brier."""
    d = df.dropna(subset=[prob_col, label_col, baseline_col]).copy()
    d[label_col] = d[label_col].astype(float)
    d["_bs"] = (d[prob_col] - d[label_col]) ** 2
    d["_bbs"] = (d[baseline_col] - d[label_col]) ** 2
    g = d.groupby("lead_hour")
    out = pd.DataFrame(
        {"brier": g["_bs"].mean(), "baseline_brier": g["_bbs"].mean(), "n": g.size()}
    ).reset_index()
    out["bss"] = 1.0 - out["brier"] / out["baseline_brier"]
    return out


def reliability_table(
    df: pd.DataFrame,
    *,
    prob_col: str = "pred_pop",
    label_col: str = "label_precip_occurrence",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability-diagram bins: predicted-prob bin vs observed frequency."""
    d = df.dropna(subset=[prob_col, label_col]).copy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(d[prob_col].to_numpy(), edges[1:-1]), 0, n_bins - 1)
    d["_bin"] = bins
    records: list[dict[str, float]] = []
    for b in range(n_bins):
        g = d[d["_bin"] == b]
        records.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_pred": float(g[prob_col].mean()) if len(g) else float("nan"),
                "observed_freq": (
                    float(g[label_col].astype(float).mean()) if len(g) else float("nan")
                ),
                "count": float(len(g)),
            }
        )
    return pd.DataFrame(records)
