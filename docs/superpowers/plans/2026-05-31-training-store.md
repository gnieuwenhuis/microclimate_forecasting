# Training Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A per-deployment, path-based, append-only training store that persists raw `FeatureSnapshot`s (as a serialized blob) plus a separate labels table to partitioned Parquet, with a small read/write API.

**Architecture:** A `TrainingStore(root)` class doing pure Parquet I/O under a directory — two datasets (`snapshots/`, `labels/`) partitioned by `deployment_id`/year-month, append-only with atomic writes and read-time dedupe-keep-latest. The store is *dumb* (no `build_features`; `TRAINING_ROW` is assembled at read time by the training pipeline, ADR-0012). The private-repo git sync (ADR-0009) is out of scope — in production the root is a checkout of the private repo and the Action syncs it.

**Tech Stack:** Python 3.12, pandas + pyarrow (already a dependency), Pydantic (`FeatureSnapshot` round-trip), pytest, pyright strict, ruff, uv.

---

## Conventions for every task

- TDD: failing test → confirm fail → implement → confirm pass → commit.
- Full gate before each commit: `uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`. Network-marked tests deselected by default.
- pandas/pyarrow are heavily untyped under pyright strict; this codebase suppresses with `# type: ignore[reportUnknownMemberType]` per line. Run `uv run pyright` and add the **narrow** ignore pyright names on each flagged pandas line (the snippets below mark the common ones; add others exactly where pyright reports them).
- Commit on the current branch `spec/training-store` (main is PR-only); push only at Final Integration.

## File structure

**Create**
- `src/microclimate/training_store/__init__.py` — exports `TrainingStore`.
- `src/microclimate/training_store/store.py` — the implementation.
- `tests/training_store/__init__.py`, `tests/training_store/test_store.py`.
- `docs/adr/0015-training-store-shape.md`.

**Modify**
- `.importlinter` — add `microclimate.training_store` to the layered contract.
- `CONTEXT.md` — refine the "Training store" term.

---

### Task 1: Snapshots — `append_snapshot` + `read_snapshots`

**Files:**
- Create: `src/microclimate/training_store/__init__.py`, `src/microclimate/training_store/store.py`, `tests/training_store/__init__.py`, `tests/training_store/test_store.py`
- Modify: `.importlinter`

- [ ] **Step 1: Add the import-linter layer entry** so the new module is governed and `lint-imports` stays green. In `.importlinter`, in `[importlinter:contract:layers]`, insert `microclimate.training_store` between `microclimate.pipelines` and `microclimate.publication`:

```ini
layers =
    microclimate.pipelines
    microclimate.training_store
    microclimate.publication
    microclimate.models
    microclimate.evaluation
    microclimate.features
    microclimate.connectors
    microclimate.config
    microclimate.contracts
```
(The store imports only `contracts` — lowest layer — so it sits cleanly below `pipelines`; placing it above `publication` is harmless since neither imports the other.)

- [ ] **Step 2: Write the failing round-trip test**

```python
# tests/training_store/test_store.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION, FeatureSnapshot
from microclimate.training_store import TrainingStore

_T0 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def _snap(deployment_id: str = "lethbridge", issue_time: datetime = _T0) -> FeatureSnapshot:
    return FeatureSnapshot(
        deployment_id=deployment_id,
        issue_time=issue_time,
        nwp_features={"nwp_temp_c_h1": 10.0, "nwp_temp_c_h2": 11.0},
        observation_features={"obs_T1_temp_c_lag0": 9.5},
        observation_masks={"obs_T1_temp_c_lag0": True},
        static_features={"static_lat": 49.7, "static_lon": -112.77},
        temporal_features={"t0_hour_sin": 0.0, "t0_hour_cos": 1.0},
        lead_hours=(1, 2, 3),
        schema_version=SNAPSHOT_SCHEMA_VERSION,
    )


def test_append_then_read_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = TrainingStore(tmp_path)
    snap = _snap()
    store.append_snapshot(snap)
    out = store.read_snapshots("lethbridge")
    assert len(out) == 1
    assert out[0] == snap  # Pydantic value-equality across the JSON round-trip


def test_read_unknown_deployment_is_empty(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert TrainingStore(tmp_path).read_snapshots("nope") == []
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/training_store/test_store.py -v`
Expected: FAIL — `microclimate.training_store` does not exist.

- [ ] **Step 4: Implement the module.**

`src/microclimate/training_store/__init__.py`:
```python
"""Per-deployment training store (raw snapshots + labels), path-based (L≈publication)."""

from __future__ import annotations

from microclimate.training_store.store import TrainingStore

__all__ = ["TrainingStore"]
```

`src/microclimate/training_store/store.py`:
```python
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
            str(v) for v in df.loc[df["schema_version"] != SNAPSHOT_SCHEMA_VERSION, "schema_version"].unique()
        )
        if bad:
            raise ValueError(
                f"training store has snapshot schema_version(s) {bad} != current "
                f"{SNAPSHOT_SCHEMA_VERSION!r}; a schema migration is required."
            )
        df = df.sort_values("issue_time")
        return [FeatureSnapshot.model_validate_json(j) for j in df["snapshot_json"]]
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/training_store/test_store.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Add the edge-case tests** (append to `test_store.py`) and confirm they pass:

```python
def test_dedupe_keeps_latest_write_per_issue_time(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = TrainingStore(tmp_path)
    first = _snap()
    second = _snap()
    second = second.model_copy(update={"nwp_features": {"nwp_temp_c_h1": 99.0}})
    store.append_snapshot(first, written_at=_T0)
    store.append_snapshot(second, written_at=_T0 + timedelta(hours=1))  # later write wins
    out = store.read_snapshots("lethbridge")
    assert len(out) == 1
    assert out[0].nwp_features["nwp_temp_c_h1"] == 99.0


def test_start_end_filter_by_issue_time(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = TrainingStore(tmp_path)
    for i in range(0, 90, 30):  # three issue_times spanning ~3 months
        store.append_snapshot(_snap(issue_time=_T0 + timedelta(days=i)))
    mid = store.read_snapshots("lethbridge", start=_T0 + timedelta(days=20), end=_T0 + timedelta(days=40))
    assert [s.issue_time for s in mid] == [_T0 + timedelta(days=30)]


def test_schema_version_mismatch_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = TrainingStore(tmp_path)
    store.append_snapshot(_snap().model_copy(update={"schema_version": "9.9.9"}))
    with pytest.raises(ValueError, match="schema_version"):
        store.read_snapshots("lethbridge")
```

Run: `uv run pytest tests/training_store/test_store.py -v`
Expected: PASS (5 tests). (`model_copy(update={"schema_version": ...})` works because `FeatureSnapshot` stores it as a field.)

- [ ] **Step 7: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/training_store/ tests/training_store/ .importlinter
git commit -m "feat(training_store): append_snapshot + read_snapshots (raw-blob, partitioned, dedupe, version-guarded)"
```

---

### Task 2: Labels — `write_labels` + `read_labels`

**Files:**
- Modify: `src/microclimate/training_store/store.py`
- Test: `tests/training_store/test_store.py`

- [ ] **Step 1: Write the failing tests** (append to `test_store.py`)

```python
def _labels(issue_time: datetime = _T0) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(
        {
            "issue_time": pd.to_datetime([issue_time] * 3, utc=True),
            "lead_hour": [1, 2, 3],
            "valid_time": pd.to_datetime([issue_time + timedelta(hours=h) for h in (1, 2, 3)], utc=True),
            "label_temp_c": [10.0, 11.0, 12.0],
            "label_precip_occurrence": pd.array([1, 0, 1], dtype="Int64"),
        }
    )


def test_write_then_read_labels_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = TrainingStore(tmp_path)
    store.write_labels("lethbridge", _labels())
    out = store.read_labels("lethbridge")
    assert list(out["lead_hour"]) == [1, 2, 3]
    assert list(out["label_temp_c"]) == [10.0, 11.0, 12.0]
    assert list(out["label_precip_occurrence"].astype("Int64")) == [1, 0, 1]
    assert "deployment_id" in out.columns
    assert "written_at" not in out.columns  # internal bookkeeping not surfaced


def test_read_labels_unknown_deployment_is_empty(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = TrainingStore(tmp_path).read_labels("nope")
    assert out.empty
    assert "lead_hour" in out.columns


def test_labels_dedupe_keeps_latest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import pandas as pd

    store = TrainingStore(tmp_path)
    store.write_labels("lethbridge", _labels(), written_at=_T0)
    revised = _labels()
    revised["label_temp_c"] = [20.0, 21.0, 22.0]
    store.write_labels("lethbridge", revised, written_at=_T0 + timedelta(hours=1))
    out = store.read_labels("lethbridge")
    assert list(out["label_temp_c"]) == [20.0, 21.0, 22.0]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/training_store/test_store.py -k labels -v`
Expected: FAIL — `write_labels`/`read_labels` absent.

- [ ] **Step 3: Implement** (add to `store.py`; add the labels column constant near `_SNAPSHOT_COLUMNS`)

```python
_LABEL_COLUMNS = [
    "deployment_id",
    "issue_time",
    "lead_hour",
    "valid_time",
    "label_temp_c",
    "label_precip_occurrence",
    "written_at",
]
```

Add these methods to `TrainingStore`:
```python
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
        for ym, part in df.groupby(df["issue_time"].dt.strftime("%Y%m")):
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/training_store/test_store.py -v`
Expected: PASS (all snapshot + label tests). If pyright flags `groupby(...).dt`/`drop_duplicates` lines, add `# type: ignore[reportUnknownMemberType]` exactly where reported.

- [ ] **Step 5: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/training_store/store.py tests/training_store/test_store.py
git commit -m "feat(training_store): write_labels + read_labels (separate two-phase labels table)"
```

---

### Task 3: ADR-0015 + CONTEXT refinement

**Files:**
- Create: `docs/adr/0015-training-store-shape.md`
- Modify: `CONTEXT.md`

- [ ] **Step 1: Write ADR-0015** (0015 is the next free number on `main`; if the CaSPAr/other branches' ADRs land first, use the next free number):

```markdown
# 15. Training store shape: raw-snapshot blob + separate labels table, path-based

- **Status:** Accepted
- **Date:** 2026-05-31
- **Relates to:** ADR-0007 (logger → accumulating store), ADR-0009 (private raw store,
  token-written), ADR-0012 (store holds raw snapshots; build_features is read-time).

## Context

With CaSPAr (historical HRDPS forecasts) unavailable, the project pivots to logger
accumulation (ADR-0007): the hourly inference pipeline appends each HRDPS snapshot to a
training store, which becomes trainable as labels accumulate forward. This is subsystem 1 of
that pivot — the store itself. ADR-0012 fixed that the store holds *raw* snapshots (derived
features recomputed on read), but not the physical shape.

## Decision

The store is a per-deployment, append-only, partitioned-Parquet dataset under a root directory:
- **snapshots** — one row per (deployment_id, issue_time): the `FeatureSnapshot` serialized
  as a JSON blob (`snapshot_json`) + `schema_version` (stamped from `SNAPSHOT_SCHEMA_VERSION`).
  Round-trips exactly via Pydantic.
- **labels** — one row per (deployment_id, issue_time, lead_hour), written *later* once target
  obs at `valid_time` land (the two-phase write inherent to the logger).

The store is **dumb**: no `build_features`. `TRAINING_ROW` is the read-time join product
(snapshot → build_features → join labels), assembled by the training pipeline, not the store.
Partitioned by `deployment_id`/year-month; append-only with atomic writes; reads dedupe on the
key keeping the latest (`written_at`). The **private-repo git sync (ADR-0009) is out of scope**
for the store — it's pure Parquet I/O on a path; in production the path is a checkout of the
private repo and the Action syncs it with the token.

## Consequences

- Snapshots carry `SNAPSHOT_SCHEMA_VERSION`; the read-time matrix carries
  `FEATURE_SCHEMA_VERSION`; labels are version-independent — resolving the earlier
  FeatureSnapshot ↔ TRAINING_ROW ↔ labeled-matrix tension.
- A `schema_version` mismatch on read fails loudly (migration required).
- Many small hourly Parquet files accumulate; periodic compaction is future work.
- New module `microclimate/training_store`, sibling to `publication` (the two artifact
  writers: private-raw vs public-derived).
```

- [ ] **Step 2: Refine the CONTEXT.md "Training store" term** to the concrete shape. Replace the existing bullet:

```markdown
- **Training store** — the accumulating per-deployment dataset behind the logger: raw
  **snapshots** (each `FeatureSnapshot` serialized as a blob + `SNAPSHOT_SCHEMA_VERSION`) plus
  a separate **labels** table (per `issue_time`×`lead_hour`, written once obs land). Partitioned
  Parquet, append-only, path-based (a private-repo checkout in production, ADR-0009/0015). The
  store is raw-only — `TRAINING_ROW` is the read-time join (snapshot → `build_features` → labels).
```

- [ ] **Step 3: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add docs/adr/0015-training-store-shape.md CONTEXT.md
git commit -m "docs: ADR-0015 training-store shape; refine CONTEXT term"
```

---

## Final Integration

- [ ] Push and open a PR (main is PR-only):

```bash
git push -u origin spec/training-store
gh pr create --fill --base main
```

- [ ] After automated review + CI, address feedback and merge.

---

## Self-review notes

- **Spec coverage:** blob+labels datasets (Tasks 1–2) ✓; dumb store / no build_features ✓; `TrainingStore` API `append_snapshot`/`read_snapshots`/`write_labels`/`read_labels` ✓; year-month partitioning + start/end filter + dedupe-keep-latest (`written_at` tiebreaker — resolves the spec's open item) ✓; atomic writes ✓; schema_version guard + empty→[]/empty-frame ✓; new module + `.importlinter` entry ✓; ADR-0015 + CONTEXT ✓. Private-repo sync correctly out of scope.
- **Open items resolved:** dedupe tiebreaker = a `written_at` column stamped on write (injectable for tests); `read_snapshots` returns `list[FeatureSnapshot]` (issue_time lives inside each); manual directory layout + glob (no pyarrow partition-discovery API) keeps it simple and testable.
- **Type consistency:** `_SNAPSHOT_COLUMNS`/`_LABEL_COLUMNS` match the written frames and the read public columns; `append_snapshot(snapshot, *, written_at=None)` and `write_labels(deployment_id, labels, *, written_at=None)` signatures match their call sites in tests.
- **Deferred (correctly):** the inference logger (subsystem 2), training-from-store (subsystem 3), the strategy ADR (pre-model publishing, trainable threshold), and Parquet compaction.
```
