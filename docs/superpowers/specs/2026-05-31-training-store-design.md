# Training store (subsystem 1 of the logger-first pivot) — design

- **Date:** 2026-05-31
- **Status:** Approved (brainstorming) — pending implementation plan
- **Relates to:** ADR-0007 (CaSPAr seed + logger → accumulating store), ADR-0009 (public-derived
  vs private-raw store; token-written), ADR-0012 (store holds raw snapshots; `build_features`
  is a read-time transform), the `FeatureSnapshot` and `TRAINING_ROW` contracts.

## Context — the logger-first pivot

CaSPAr (the only public archive of historical HRDPS *forecasts*) appears unavailable
(Global Water Futures ended 2025; tooling repo stale since 2023; site down across all checks).
There is no substitute for historical HRDPS forecasts, and snapshots **cannot be backfilled**
(MSC Datamart serves only recent runs). So the project pivots to its own designed fallback:
the **logger** (ADR-0007) — the hourly inference pipeline persists each HRDPS snapshot it
builds (from the now-verified Datamart connector), and the deployment becomes trainable as
labeled rows accumulate forward (~48 h+ per snapshot). This is the previously-deferred
`cold_start`/logger path (ADR-0008) becoming the de-facto v1 training path even for the
`seeded` lethbridge deployment.

The pivot decomposes into a strategy/ADR decision + three builds: **(1) the training store**
(this spec), (2) the inference pipeline/logger, (3) training-from-store. The strategy
sub-decisions (what publishes before a trained model exists; the "trainable" threshold) are
folded into the logger/training specs that follow; this spec covers the foundational store.

## Purpose

A per-deployment, append-only, partitioned-Parquet **training store** that the logger appends
raw snapshots to and the training pipeline reads — the persistence layer both depend on.

## Scope

**In scope**

- A `TrainingStore` class over a root directory: `append_snapshot`, `write_labels`,
  `read_snapshots`, `read_labels`.
- Two Parquet datasets — **snapshots** (raw `FeatureSnapshot` serialized as a blob) and
  **labels** (per `(issue_time, lead_hour)`), reflecting the two-phase write timing (snapshot
  now; labels ~48 h later when target obs land).
- `SNAPSHOT_SCHEMA_VERSION` stamping + read-time validation.
- Year-month partitioning, append-only writes, read-time dedupe (keep latest), atomic writes.
- A new ADR recording the store shape; CONTEXT refinement.

**Out of scope (separate work)**

- The **private-repo git sync** (clone/commit/push via `DATA_REPO_TOKEN`) — a CI/pipeline
  concern (ADR-0009: "the Action writes to it with a token"), deferred to the logger work.
  The store is pure Parquet I/O on a path; in production that path is a checkout of the
  private repo.
- The **inference logger** (subsystem 2) and **training-from-store** orchestration
  (subsystem 3), incl. `build_features`/label-attachment composition — the store is *dumb*
  (no feature derivation, per ADR-0012).
- The full **CaSPAr-pivot strategy ADR** (pre-model publishing, trainable threshold) — lands
  with the logger/training specs.
- Compaction of the many small hourly Parquet files — noted as future work.

## Home & layering

New module `src/microclimate/training_store/` (e.g. `store.py`), **sibling to `publication`**
— ADR-0009 frames them as the two artifact writers (public-derived vs private-raw). It imports
`contracts` (`FeatureSnapshot`, `SNAPSHOT_SCHEMA_VERSION`, the `TRAINING_ROW`/label column
names) and `config` (only if a deployment type is needed; likely just `deployment_id: str`),
and is imported by `pipelines`. Adds one entry to the `.importlinter` layered contract at the
`publication` level (independent sibling).

## Data model

**snapshots** — one row per `(deployment_id, issue_time)`:

| column | type | notes |
|---|---|---|
| `deployment_id` | str | partition key |
| `issue_time` | datetime64[ns, UTC] | |
| `schema_version` | str | `SNAPSHOT_SCHEMA_VERSION` at write |
| `snapshot_json` | str | `FeatureSnapshot.model_dump_json()` — round-trips exactly |

**labels** — one row per `(deployment_id, issue_time, lead_hour)`, written later:

| column | type | notes |
|---|---|---|
| `deployment_id` | str | partition key |
| `issue_time` | datetime64[ns, UTC] | |
| `lead_hour` | int (1–48) | |
| `valid_time` | datetime64[ns, UTC] | |
| `label_temp_c` | float, nullable | |
| `label_precip_occurrence` | Int (0/1), nullable | |

The store persists raw snapshots + a labels table; **`TRAINING_ROW` is the read-time join
product** (snapshot → `build_features` → join labels), assembled by the training pipeline
(subsystem 3) — not by the store. This resolves the earlier `FeatureSnapshot` ↔ `TRAINING_ROW`
↔ labeled-feature-matrix tension: snapshots carry `SNAPSHOT_SCHEMA_VERSION`; the read-time
matrix carries `FEATURE_SCHEMA_VERSION`; labels are version-independent.

## API — `TrainingStore(root: Path)`, a dumb store

- `append_snapshot(snapshot: FeatureSnapshot) -> None` — stamps `SNAPSHOT_SCHEMA_VERSION`,
  serializes to `snapshot_json`, writes one snapshot row under the snapshot's `deployment_id`
  and `issue_time` year-month partition.
- `write_labels(deployment_id: str, labels: pd.DataFrame) -> None` — append label rows
  (columns per the labels model above) for an issue_time's leads.
- `read_snapshots(deployment_id: str, start: datetime | None = None, end: datetime | None = None) -> list[FeatureSnapshot]`
  — load the partition(s), filter by `issue_time` ∈ [start, end], dedupe on `issue_time`
  keeping the latest write, round-trip each `snapshot_json` via `FeatureSnapshot.model_validate_json`,
  and **raise** on any row whose `schema_version` ≠ `SNAPSHOT_SCHEMA_VERSION` (loud — a schema
  change needs a migration).
- `read_labels(deployment_id: str, start=None, end=None) -> pd.DataFrame` — labels in the
  range, deduped on `(issue_time, lead_hour)` keeping the latest.

## Layout, partitioning, idempotency, errors

- pyarrow dataset, partitioned `deployment_id` → `ym=YYYYMM` (from `issue_time`):
  `{root}/snapshots/deployment_id=…/ym=YYYYMM/<uuid-or-timestamp>.parquet` and `{root}/labels/…`.
- **Append-only:** each write is a new Parquet file (no rewrite of existing files).
- **Idempotency:** retried runs may duplicate an `issue_time`; **reads dedupe** on
  `(deployment_id, issue_time)` (snapshots) / `(deployment_id, issue_time, lead_hour)` (labels),
  keeping the latest by file write order (or a `written_at` stamp — decide in the plan).
- **Atomic writes:** write to a temp file in the partition dir, then `os.replace` to the final
  name — a crashed run never leaves a half-written Parquet.

## Error handling

- `append_snapshot` / `write_labels`: create partition dirs as needed; atomic write; surface
  disk errors as-is (caller decides). No silent partial writes.
- `read_snapshots`: empty/absent partition → `[]` (a not-yet-accumulated deployment is normal,
  not an error). `schema_version` mismatch → `ValueError` naming the offending version.
- `read_labels`: empty → empty DataFrame with the labels columns.

## Testing (offline, `tmp_path`)

- **Round-trip:** `append_snapshot(snap)` → `read_snapshots` returns a `FeatureSnapshot` equal
  to the original (all nested mappings/tuples/datetimes preserved).
- **Labels:** `write_labels` → `read_labels` returns the rows; nullable label dtypes preserved.
- **Filtering:** append snapshots across several months → `read_snapshots(start, end)` returns
  exactly the in-range subset (year-month partition pruning correct).
- **Dedupe:** appending the same `issue_time` twice → `read_snapshots` returns one (latest).
- **schema_version:** a row written with a different version → `read_snapshots` raises.
- **Empty:** `read_snapshots` on an unknown deployment → `[]`.
- Full gate: ruff format/check, `lint-imports` (new layer entry), pyright strict, pytest.

## Documentation updates (same PR, per CLAUDE.md)

- **New ADR** — "Training store: raw-snapshot-blob + labels table, path-based, CI-owned sync"
  (refines ADR-0007/0009/0012; next free number — currently 0015 on main, but reconcile if the
  CaSPAr/Datamart branches' ADRs land first).
- **CONTEXT.md** — refine the **Training store** term to the concrete shape (snapshots-blob +
  labels; raw, not derived; read-time `TRAINING_ROW` join).
- A short context note (in the ADR) that this is subsystem 1 of the CaSPAr-driven logger-first
  pivot; the strategy ADR follows with the logger work.

## Decomposition (for the plan)

1. `TrainingStore.append_snapshot` + `read_snapshots` round-trip (+ schema_version guard).
2. `write_labels` + `read_labels`.
3. Year-month partitioning + start/end filtering + read-time dedupe.
4. Atomic-write + empty-partition handling.
5. `.importlinter` layer entry + ADR + CONTEXT refinement.

## Open items deferred to the plan

- The dedupe "latest" tiebreaker: file mtime/write-order vs an explicit `written_at` column
  (lean: a `written_at` UTC column stamped on write — deterministic, testable).
- Whether `read_snapshots` returns `list[FeatureSnapshot]` or also surfaces `issue_time`
  alongside (lean: the list — `issue_time` is inside each snapshot).
- Exact pyarrow partitioning API (`write_to_dataset` vs manual path layout) — a plan detail.
