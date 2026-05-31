"""Training-data assembly: the shared seam reused by the notebook and (later) the training
pipeline (L6).

Iterates issue-times through the shared build_snapshot -> build_features path, performs the
single training-only *future* read of target observations (values at valid_time), and
attaches labels. This future read is legal here (backfill/training) and categorically absent
from build_snapshot/build_features, preserving the ADR-0011 no-leakage guarantee.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd

from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.features.feature_builder import build_features
from microclimate.features.labeler import attach_labels
from microclimate.features.snapshot_builder import build_snapshot


def assemble_training_rows(
    config: DeploymentConfig,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    issue_times: Iterable[datetime],
) -> pd.DataFrame:
    """Build a labeled feature matrix spanning all given issue_times."""
    matrices: list[pd.DataFrame] = []
    for issue_time in issue_times:
        snapshot = build_snapshot(config, issue_time, nwp, observations)
        matrices.append(build_features(snapshot, config))
    if not matrices:
        raise ValueError("issue_times is empty; nothing to assemble")
    matrix = pd.concat(matrices, ignore_index=True)

    # Single batched future read of the target station across the whole valid-time span.
    target_source = observations[config.target.connector_key]
    start = matrix["valid_time"].min().to_pydatetime()
    end = matrix["valid_time"].max().to_pydatetime()
    target_obs = target_source.fetch_historical(config.target.station_id, start, end)

    return attach_labels(matrix, target_obs, config.label.precip_occurrence_threshold_mm)


def assemble_or_load(
    config: DeploymentConfig,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    issue_times: Iterable[datetime],
    *,
    cache_path: Path,
) -> pd.DataFrame:
    """Read assembled rows from a local Parquet cache, else assemble and write it.

    Local-dev convenience so notebook re-runs don't re-pull CaSPAr. The cache is keyed by
    the caller's chosen path; rotate the path when the issue-time range or snapshot schema
    changes. Derived features are recomputed by build_features on read, so the derived
    FEATURE_SCHEMA_VERSION is intentionally NOT part of the key (ADR-0012).
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    rows = assemble_training_rows(config, nwp, observations, issue_times)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(cache_path, index=False)
    return rows
