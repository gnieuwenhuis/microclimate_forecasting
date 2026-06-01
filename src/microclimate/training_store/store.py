"""Append-only, partitioned-Parquet training store.

Persists raw FeatureSnapshots (serialized to a JSON blob) and a separate labels table under a
root directory. Dumb: no feature derivation — build_features is a read-time transform owned by
the training pipeline (ADR-0012), and TRAINING_ROW is the read-time join product. The
private-repo git sync (ADR-0009) is a CI/pipeline concern outside this module; in production
the root is a checkout of the private repo.

Layout (append-only; one Parquet file per write, atomic via temp+os.replace):
    {root}/snapshots/deployment_id={id}/ym={YYYYMM}/{uuid}.parquet
    {root}/labels/deployment_id={id}/ym={YYYYMM}/{uuid}.parquet
Reads dedupe on (issue_time[, lead_hour]) keeping the latest write (by written_at).
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


def _atomic_write_parquet(df: pd.DataFrame, partition_dir: Path) -> None:
    """Write df to a uniquely-named Parquet in partition_dir via temp file + os.replace."""
    partition_dir.mkdir(parents=True, exist_ok=True)
    final = partition_dir / f"{uuid.uuid4().hex}.parquet"
    tmp = partition_dir / f".{uuid.uuid4().hex}.tmp"
    df.to_parquet(tmp, index=False)  # type: ignore[reportUnknownMemberType]
    os.replace(tmp, final)


class TrainingStore:
    """Path-based training store. `root` is a local dir (in prod, a private-repo checkout)."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def append_snapshot(
        self, snapshot: FeatureSnapshot, *, written_at: datetime | None = None
    ) -> None:
        """Append one raw snapshot row, stamped with its schema_version and a write time."""
        stamp = written_at if written_at is not None else datetime.now(UTC)
        df = pd.DataFrame(
            [
                {
                    "deployment_id": snapshot.deployment_id,
                    "issue_time": pd.Timestamp(snapshot.issue_time),
                    "schema_version": snapshot.schema_version,
                    "snapshot_json": snapshot.model_dump_json(),
                    "written_at": pd.Timestamp(stamp),
                }
            ],
            columns=_SNAPSHOT_COLUMNS,
        )
        pdir = (
            self._root
            / "snapshots"
            / f"deployment_id={snapshot.deployment_id}"
            / f"ym={_ym(snapshot.issue_time)}"
        )
        _atomic_write_parquet(df, pdir)

    def read_snapshots(
        self,
        deployment_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[FeatureSnapshot]:
        """Return snapshots for the deployment in [start, end], latest-per-issue_time, sorted."""
        base = self._root / "snapshots" / f"deployment_id={deployment_id}"
        files = sorted(base.glob("ym=*/*.parquet")) if base.exists() else []
        if not files:
            return []
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)  # type: ignore[reportUnknownMemberType]
        df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
        df["written_at"] = pd.to_datetime(df["written_at"], utc=True)
        if start is not None:
            df = df[df["issue_time"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["issue_time"] <= pd.Timestamp(end)]
        if df.empty:
            return []
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
        stamp = written_at if written_at is not None else datetime.now(UTC)
        df = labels.copy()
        df["deployment_id"] = deployment_id
        df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
        df["written_at"] = pd.Timestamp(stamp)
        df["written_at"] = pd.to_datetime(df["written_at"], utc=True)
        df = df[_LABEL_COLUMNS]
        for ym, part in df.groupby(df["issue_time"].dt.strftime("%Y%m")):  # type: ignore[reportUnknownMemberType]
            pdir = self._root / "labels" / f"deployment_id={deployment_id}" / f"ym={ym}"
            _atomic_write_parquet(part, pdir)

    def read_labels(
        self,
        deployment_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Return labels for the deployment in [start, end], latest-per-(issue_time, lead_hour)."""
        public_cols = [c for c in _LABEL_COLUMNS if c != "written_at"]
        base = self._root / "labels" / f"deployment_id={deployment_id}"
        files = sorted(base.glob("ym=*/*.parquet")) if base.exists() else []
        if not files:
            return pd.DataFrame(columns=public_cols)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)  # type: ignore[reportUnknownMemberType]
        df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
        df["written_at"] = pd.to_datetime(df["written_at"], utc=True)
        if start is not None:
            df = df[df["issue_time"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["issue_time"] <= pd.Timestamp(end)]
        df = df.sort_values("written_at").drop_duplicates(
            subset=["issue_time", "lead_hour"], keep="last"
        )
        return (
            df.drop(columns="written_at")
            .sort_values(["issue_time", "lead_hour"])
            .reset_index(drop=True)[public_cols]
        )
