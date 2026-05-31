"""Label attachment: label-free feature matrix -> labeled feature matrix (L3, pure).

Pure and connector-free. The *future* read of target observations (values at valid_time,
which are after issue_time) happens in the assembler (pipelines.training_data), where a
training-only future read is legal; this function receives that frame already read. The
output is the **labeled feature matrix** the models train on -- NOT the persisted
TRAINING_ROW (which is raw snapshot + labels in the training store, ADR-0012).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_labels(
    matrix: pd.DataFrame,
    target_obs: pd.DataFrame,
    threshold_mm: float,
) -> pd.DataFrame:
    """Add `label_temp_c` and `label_precip_occurrence` by joining target obs at valid_time.

    `label_precip_occurrence` is 1 when observed precip >= threshold_mm, else 0, and <NA>
    where the target observation for that valid_time is missing (ADR-0008 degradation).
    """
    obs = target_obs[["timestamp", "temp_c", "precip_mm"]].copy()
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], utc=True)
    obs = obs.drop_duplicates(subset="timestamp").set_index("timestamp")

    valid = pd.to_datetime(matrix["valid_time"], utc=True)
    temp = valid.map(obs["temp_c"])
    precip = valid.map(obs["precip_mm"])

    # NaN where the target obs is missing, else 1/0 — built via float so the missing
    # entries become <NA> cleanly under the nullable Int64 dtype.
    occurrence = np.where(precip.isna().to_numpy(), np.nan, precip.to_numpy() >= threshold_mm)

    out = matrix.copy()
    out["label_temp_c"] = temp.astype("float64")
    out["label_precip_occurrence"] = pd.array(occurrence, dtype="Int64")
    return out
