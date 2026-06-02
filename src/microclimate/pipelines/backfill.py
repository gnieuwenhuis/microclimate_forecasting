"""Retrain-time seed backfill: pull deep HRDPS history into the training store (ADR-0019).

Idempotent and additive — re-running coalesces by (issue_time, lead_hour) and never prunes.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd

from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import (
    ForecastUnavailable,
    NWPSource,
    ObservationSource,
    SourceUnavailable,
)
from microclimate.contracts.snapshot import FeatureSnapshot
from microclimate.features.feature_builder import build_features
from microclimate.features.labeler import attach_labels
from microclimate.features.snapshot_builder import build_snapshot
from microclimate.training_store.store import TrainingStore

_RUN_HOURS: tuple[int, ...] = (0, 6, 12, 18)

_LABEL_COLS = ["issue_time", "lead_hour", "valid_time", "label_temp_c", "label_precip_occurrence"]


def hrdps_issue_times(start: datetime, end: datetime) -> list[datetime]:
    """All HRDPS run init times (00/06/12/18 UTC) in [start, end], ascending."""
    s = start.astimezone(UTC) if start.tzinfo else start.replace(tzinfo=UTC)
    e = end.astimezone(UTC) if end.tzinfo else end.replace(tzinfo=UTC)
    cur = s.replace(hour=0, minute=0, second=0, microsecond=0)
    out: list[datetime] = []
    while cur <= e:
        if cur.hour in _RUN_HOURS and cur >= s:
            out.append(cur)
        cur += timedelta(hours=6)
    return out


def backfill_store(
    config: DeploymentConfig,
    *,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    store: TrainingStore,
    issue_times: Sequence[datetime],
    pause_s: float = 0.12,  # ~500/min, under the 600/min free limit
) -> int:
    """Build + persist snapshots and labels for each issue_time. Returns count newly written.

    Idempotent: skips issue_times already stored. Additive: TrainingStore coalesces and never
    prunes.
    """
    fresh_snapshots: list[FeatureSnapshot] = []
    for t0 in issue_times:
        if store.has_snapshot(config.deployment_id, t0):
            continue
        try:
            snapshot = build_snapshot(config, t0, nwp, observations)
        except (ForecastUnavailable, SourceUnavailable):
            continue  # run absent from the archive — skip; re-run stays gapless
        store.append_snapshot(snapshot)
        fresh_snapshots.append(snapshot)
        if pause_s:
            time.sleep(pause_s)

    if fresh_snapshots:
        matrices = [build_features(s, config) for s in fresh_snapshots]
        matrix = pd.concat(matrices, ignore_index=True)
        target = observations[config.target.connector_key]
        start = matrix["valid_time"].min().to_pydatetime()
        end = matrix["valid_time"].max().to_pydatetime()
        target_obs = target.fetch_historical(config.target.station_id, start, end)
        labeled = attach_labels(matrix, target_obs, config.label.precip_occurrence_threshold_mm)
        store.write_labels(config.deployment_id, labeled[_LABEL_COLS])
    return len(fresh_snapshots)
