"""Coalesced, partitioned-Parquet training store.

Persists raw FeatureSnapshots (serialized to a JSON blob) and a separate labels table under a
root directory. Dumb: no feature derivation — build_features is a read-time transform owned by
the training pipeline (ADR-0012), and TRAINING_ROW is the read-time join product. The
private-repo git sync (ADR-0009) is a CI/pipeline concern outside this module; in production
the root is a checkout of the private repo.

Layout (one coalesced Parquet per partition; grown by read-modify-write, atomic via
temp+os.replace; deduped at write time):
    {root}/snapshots/deployment_id={id}/ym={YYYYMM}/data.parquet
    {root}/labels/deployment_id={id}/ym={YYYYMM}/data.parquet
The training-data branch is force-pushed as a single state commit (ADR-0017/0018).
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION, FeatureSnapshot

_SNAPSHOT_COLUMNS = [
    "deployment_id",
    "issue_time",
    "schema_version",
    "snapshot_json",
    "written_at",
]

_LABEL_COLUMNS = [
    "deployment_id",
    "issue_time",
    "lead_hour",
    "valid_time",
    "label_temp_c",
    "label_precip_occurrence",
    "written_at",
]


def _ym(ts: datetime) -> str:
    return ts.strftime("%Y%m")


def _to_utc(ts: datetime) -> pd.Timestamp:
    """Normalize a datetime to a UTC pandas Timestamp (tz-aware → convert; naive → assume UTC).

    Matches the connectors' naive-is-UTC convention and keeps the store's stored/compared
    times unambiguously UTC, so dedupe ordering and range filtering never hit a tz-naive
    vs tz-aware comparison error.
    """
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")  # type: ignore[reportUnknownMemberType]


def _range_bounds(
    start: datetime | None, end: datetime | None
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Normalize [start, end] to UTC Timestamps and require start <= end when both are given."""
    s = _to_utc(start) if start is not None else None
    e = _to_utc(end) if end is not None else None
    if s is not None and e is not None and s > e:
        raise ValueError(f"start {s.isoformat()} is after end {e.isoformat()}")
    return s, e


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Atomically write df to the Parquet file ``dest`` (temp in the same dir + os.replace)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{uuid.uuid4().hex}.tmp"
    df.to_parquet(tmp, index=False)  # type: ignore[reportUnknownMemberType]
    os.replace(tmp, dest)


def _merge_and_dedupe(
    dest: Path, new: pd.DataFrame, *, subset: list[str], dt_cols: list[str]
) -> pd.DataFrame:
    """Read the existing partition file (if any), append ``new``, normalise ``dt_cols`` to UTC,
    dedupe on ``subset`` keeping the latest ``written_at``, return sorted by ``subset``."""
    frames = [pd.read_parquet(dest), new] if dest.exists() else [new]  # type: ignore[reportUnknownMemberType]
    df = pd.concat(frames, ignore_index=True)  # type: ignore[reportUnknownMemberType]
    for col in dt_cols:
        df[col] = pd.to_datetime(df[col], utc=True)
    return (
        df.sort_values("written_at")
        .drop_duplicates(subset=subset, keep="last")
        .sort_values(subset)
        .reset_index(drop=True)
    )


class TrainingStore:
    """Path-based training store. `root` is a local dir (in prod, a private-repo checkout)."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def _partition(self, deployment_id: str, issue_time: datetime, kind: str) -> Path:
        # Normalize to UTC before deriving the ym= month so the write path and has_snapshot
        # always agree on the partition, even when a tz offset crosses a month boundary.
        return (
            self._root
            / kind
            / f"deployment_id={deployment_id}"
            / f"ym={_ym(_to_utc(issue_time))}"
            / "data.parquet"
        )

    def append_snapshot(
        self, snapshot: FeatureSnapshot, *, written_at: datetime | None = None
    ) -> None:
        """Append one raw snapshot row into its deployment-month file (read-modify-write)."""
        stamp = _to_utc(written_at) if written_at is not None else _to_utc(datetime.now(UTC))
        new = pd.DataFrame(
            [
                {
                    "deployment_id": snapshot.deployment_id,
                    "issue_time": pd.Timestamp(snapshot.issue_time),
                    "schema_version": snapshot.schema_version,
                    "snapshot_json": snapshot.model_dump_json(),
                    "written_at": stamp,
                }
            ],
            columns=_SNAPSHOT_COLUMNS,
        )
        dest = self._partition(snapshot.deployment_id, snapshot.issue_time, "snapshots")
        merged = _merge_and_dedupe(
            dest, new, subset=["issue_time"], dt_cols=["issue_time", "written_at"]
        )
        _atomic_write_parquet(merged, dest)

    def has_snapshot(self, deployment_id: str, issue_time: datetime) -> bool:
        """True if a snapshot for ``issue_time`` is already stored (cheap; reads one month file)."""
        ts = _to_utc(issue_time)
        dest = self._partition(deployment_id, ts, "snapshots")
        if not dest.exists():
            return False
        existing = pd.read_parquet(dest, columns=["issue_time"])  # type: ignore[reportUnknownMemberType]
        return bool((pd.to_datetime(existing["issue_time"], utc=True) == ts).any())

    def read_snapshots(
        self,
        deployment_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[FeatureSnapshot]:
        """Return snapshots for the deployment in [start, end], latest-per-issue_time, sorted."""
        start_ts, end_ts = _range_bounds(start, end)
        base = self._root / "snapshots" / f"deployment_id={deployment_id}"
        files = sorted(base.glob("ym=*/data.parquet")) if base.exists() else []
        if not files:
            return []
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)  # type: ignore[reportUnknownMemberType]
        df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
        df["written_at"] = pd.to_datetime(df["written_at"], utc=True)
        if start_ts is not None:
            df = df[df["issue_time"] >= start_ts]
        if end_ts is not None:
            df = df[df["issue_time"] <= end_ts]
        if df.empty:
            return []
        # Files are deduped at write time; this defensive dedupe is cheap insurance.
        df = df.sort_values("written_at").drop_duplicates(subset="issue_time", keep="last")
        bad = sorted(
            str(v)
            for v in df.loc[
                df["schema_version"] != SNAPSHOT_SCHEMA_VERSION, "schema_version"
            ].unique()
        )
        if bad:
            raise ValueError(
                f"training store has snapshot schema_version(s) {bad} != current "
                f"{SNAPSHOT_SCHEMA_VERSION!r}; a schema migration is required."
            )
        df = df.sort_values("issue_time")
        return [FeatureSnapshot.model_validate_json(j) for j in df["snapshot_json"]]

    def write_labels(
        self, deployment_id: str, labels: pd.DataFrame, *, written_at: datetime | None = None
    ) -> None:
        """Append label rows (one per (issue_time, lead_hour)) for a deployment.

        `labels` must carry: issue_time, lead_hour, valid_time, label_temp_c,
        label_precip_occurrence. Written later than the snapshot, once obs at valid_time land.
        """
        required = {
            "issue_time",
            "lead_hour",
            "valid_time",
            "label_temp_c",
            "label_precip_occurrence",
        }
        missing = required - set(labels.columns)
        if missing:
            raise ValueError(f"labels is missing required column(s): {sorted(missing)}")
        stamp = _to_utc(written_at) if written_at is not None else _to_utc(datetime.now(UTC))
        df = labels.copy()
        df["deployment_id"] = deployment_id
        df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
        df["written_at"] = stamp
        df = df[_LABEL_COLUMNS]
        for ym, part in df.groupby(df["issue_time"].dt.strftime("%Y%m")):  # type: ignore[reportUnknownMemberType]
            dest = (
                self._root
                / "labels"
                / f"deployment_id={deployment_id}"
                / f"ym={ym}"
                / "data.parquet"
            )
            merged = _merge_and_dedupe(
                dest,
                part,
                subset=["issue_time", "lead_hour"],
                dt_cols=["issue_time", "valid_time", "written_at"],
            )
            _atomic_write_parquet(merged, dest)

    def read_labels(
        self,
        deployment_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Return labels for the deployment in [start, end], latest-per-(issue_time, lead_hour)."""
        start_ts, end_ts = _range_bounds(start, end)
        public_cols = [c for c in _LABEL_COLUMNS if c != "written_at"]
        base = self._root / "labels" / f"deployment_id={deployment_id}"
        files = sorted(base.glob("ym=*/data.parquet")) if base.exists() else []
        if not files:
            return pd.DataFrame(columns=public_cols)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)  # type: ignore[reportUnknownMemberType]
        df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
        df["written_at"] = pd.to_datetime(df["written_at"], utc=True)
        if start_ts is not None:
            df = df[df["issue_time"] >= start_ts]
        if end_ts is not None:
            df = df[df["issue_time"] <= end_ts]
        df = df.sort_values("written_at").drop_duplicates(
            subset=["issue_time", "lead_hour"], keep="last"
        )
        return (
            df.drop(columns="written_at")
            .sort_values(["issue_time", "lead_hour"])
            .reset_index(drop=True)[public_cols]
        )
