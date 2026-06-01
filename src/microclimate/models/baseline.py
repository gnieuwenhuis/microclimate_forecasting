"""Raw-HRDPS baseline forecaster (L4).

The initial published champion (ADR-0016) and the floor a trained model must beat: temperature
is the HRDPS 2 m passthrough; PoP is the raw-HRDPS occurrence call (precip ≥ threshold → 1).
Pure; no I/O. Self-contained — does NOT import `evaluation` (models/evaluation are import-linter
siblings); the one-line occurrence rule is duplicated, matching `evaluation.nwp_pop_baseline`.
"""

from __future__ import annotations

import pandas as pd

BASELINE_VERSION = "baseline"


def baseline_predictions(rows: pd.DataFrame, threshold_mm: float) -> pd.DataFrame:
    """Add `pred_temp_c` (= nwp_temp_c) and `pred_pop` (= 1.0 if nwp_precip_mm ≥ threshold)."""
    out = rows.copy()
    out["pred_temp_c"] = rows["nwp_temp_c"]
    out["pred_pop"] = (rows["nwp_precip_mm"] >= threshold_mm).astype(float)
    return out
