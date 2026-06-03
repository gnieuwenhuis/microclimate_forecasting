"""Training-data assembly: the shared seam reused by the notebook and (later) the training
pipeline (L6).

Iterates issue-times through the shared build_snapshot -> build_features path, performs the
single training-only *future* read of target observations (values at valid_time), and
attaches labels. This future read is legal here (backfill/training) and categorically absent
from build_snapshot/build_features, preserving the ADR-0011 no-leakage guarantee.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.connectors.caching import CachingObservationSource
from microclimate.features.feature_builder import build_features
from microclimate.features.labeler import attach_labels
from microclimate.features.snapshot_builder import build_snapshot
from microclimate.training_store.store import TrainingStore


def assemble_training_rows(
    config: DeploymentConfig,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    issue_times: Iterable[datetime],
) -> pd.DataFrame:
    """Build a labeled feature matrix spanning all given issue_times."""
    times = list(issue_times)
    if not times:
        raise ValueError("issue_times is empty; nothing to assemble")

    # Prefetch each station's full observation window once and serve as-of slices from memory,
    # so a multi-issue-time assembly does O(stations) obs fetches instead of O(issue_times) — the
    # same optimisation backfill_store uses. The window covers the lagged as-of reads and the
    # forward label read (valid_time up to max issue_time + horizon).
    win_start = min(times) - timedelta(hours=config.lag_hours)
    win_end = max(times) + timedelta(hours=config.horizon_hours)
    obs: dict[str, ObservationSource] = {}
    for key, src in observations.items():
        ids_for_key = [
            ref.station_id for ref in (config.target, *config.neighbors) if ref.connector_key == key
        ]
        obs[key] = CachingObservationSource(src, ids_for_key, win_start, win_end)

    matrices: list[pd.DataFrame] = []
    for issue_time in times:
        snapshot = build_snapshot(config, issue_time, nwp, obs)
        matrices.append(build_features(snapshot, config))
    matrix = pd.concat(matrices, ignore_index=True)

    # Single batched future read of the target station across the whole valid-time span
    # (served from the prefetched cache).
    target_source = obs[config.target.connector_key]
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

    Local-dev convenience so notebook re-runs don't re-pull CaSPAr. The cache stores the
    fully-assembled rows (derived features already built by build_features) and returns them
    as-is on a hit — it does NOT re-run build_features on read. The cache is keyed solely by
    the caller's chosen path, so rotate the path whenever anything that would change the rows
    changes: the issue-time range, the snapshot schema, or the derived FEATURE_SCHEMA_VERSION.
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    rows = assemble_training_rows(config, nwp, observations, issue_times)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(cache_path, index=False)
    return rows


def assemble_from_store(config: DeploymentConfig, store: TrainingStore) -> pd.DataFrame:
    """Build the labeled feature matrix from persisted store snapshots + labels (no network).

    The store-backed counterpart to ``assemble_training_rows``: it re-derives features from each
    stored ``FeatureSnapshot`` (``build_features``) and rejoins the labels written during the
    seed backfill, on ``(issue_time, lead_hour)``. Output columns match the live-assembly path,
    so it feeds ``chronological_split`` and the models identically — but is local, fast, and
    repeatable (the network pull happened once, in ``backfill_store``).
    """
    snapshots = store.read_snapshots(config.deployment_id)
    if not snapshots:
        raise ValueError(f"training store has no snapshots for {config.deployment_id!r}")
    matrix = pd.concat(
        [build_features(snapshot, config) for snapshot in snapshots], ignore_index=True
    )
    matrix["issue_time"] = pd.to_datetime(matrix["issue_time"], utc=True)

    labels = store.read_labels(config.deployment_id)
    if labels.empty:
        raise ValueError(f"training store has no labels for {config.deployment_id!r}")
    labels = labels.copy()
    labels["issue_time"] = pd.to_datetime(labels["issue_time"], utc=True)
    label_cols = ["issue_time", "lead_hour", "label_temp_c", "label_precip_occurrence"]
    return matrix.merge(labels[label_cols], on=["issue_time", "lead_hour"], how="left")


def chronological_split(
    rows: pd.DataFrame,
    *,
    train_frac: float = 0.6,
    calib_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into chronological train | calib | test by *whole* issue_time.

    Splits on unique sorted issue_times (never across a single issue_time) so adjacent,
    strongly-correlated rows can't leak between sets. The remainder after train+calib is the
    test holdout. Temp trains on train+calib; PoP trains on train and calibrates on calib.
    """
    if train_frac + calib_frac >= 1.0:
        raise ValueError("train_frac + calib_frac must leave a non-empty test holdout")
    times = pd.Index(sorted(rows["issue_time"].unique()))
    n = len(times)
    n_train = int(n * train_frac)
    n_calib = int(n * calib_frac)
    # Guard the degenerate small-n case: int() truncation can zero out a slice (e.g. n<5
    # with the default calib_frac=0.2 → empty calib), which would later make the PoP
    # calibrator fit on no rows. Fail loudly here instead.
    if n_train == 0 or n_calib == 0 or n_train + n_calib >= n:
        raise ValueError(
            f"n={n} unique issue_times is too few to form non-empty train/calib/test "
            f"splits at train_frac={train_frac}, calib_frac={calib_frac}"
        )
    train_times = set(times[:n_train])
    calib_times = set(times[n_train : n_train + n_calib])
    test_times = set(times[n_train + n_calib :])
    return (
        rows[rows["issue_time"].isin(train_times)].copy(),
        rows[rows["issue_time"].isin(calib_times)].copy(),
        rows[rows["issue_time"].isin(test_times)].copy(),
    )
