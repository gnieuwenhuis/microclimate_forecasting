# Training store coalescing — one file per month, read-modify-write (store revision)

- **Date:** 2026-05-31
- **Status:** Approved (brainstorming) — pending implementation plan
- **Revises:** ADR-0015 (append-only-uuid + read-time dedupe → coalesced read-modify-write).
- **Relates to:** ADR-0007/0008 (logger), ADR-0017 (public training-data branch), the
  `TrainingStore`, `pipelines.inference.run_inference`, and `.github/workflows/inference.yml`.

## Context

ADR-0015's store writes a **new `<uuid>.parquet` per append** and dedupes at read time. Run
hourly by the inference Action, that produces a new file every run — and worse, the hourly
cron re-logs the *same* 6-hourly HRDPS `issue_time` ~6× (byte-identical snapshots), so files
pile up: thousands of tiny files over time, slow to read. This revision coalesces each
deployment-month into **one `data.parquet`** (read-modify-write), dedupes at **write** time,
adds **idempotent logging** (skip an already-stored `issue_time`), and has the Action manage
the `training-data` branch as **state** (force-push a single commit), so git history doesn't
bloat from re-storing a growing binary file.

## Scope

**In scope**

- `TrainingStore`: `append_snapshot`/`write_labels` → coalesced read-modify-write into one
  `data.parquet` per partition with write-time dedupe; `read_snapshots`/`read_labels` read the
  one-file-per-month layout; new `has_snapshot`.
- `pipelines.inference.run_inference`: idempotent skip via `has_snapshot` (returns
  `ForecastDocument | None`); `main()` handles `None`.
- `.github/workflows/inference.yml`: force-push a single state commit of `./store` to
  `training-data`.
- ADR-0018 (this revision) + amendment notes on ADR-0015 and ADR-0017; CONTEXT update.

**Out of scope**

- **Concurrent multi-writer** support — the store stays single-writer (the Action is one
  serialized job); read-modify-write + force-push assume one writer. Noted, not built.
- A separate **compaction job** — coalescing *is* the compaction; ADR-0015's compaction
  future-work item is resolved here.
- Weekly windowing — `ym`→`yw` is a trivial future knob; monthly now.
- Any change to what a snapshot/label *contains* (the row schema is unchanged).

## Storage model (revised)

Partition layout unchanged except the filename is **fixed per partition**:
```
{root}/snapshots/deployment_id=<id>/ym=<YYYYMM>/data.parquet   # one file per deployment-month
{root}/labels/deployment_id=<id>/ym=<YYYYMM>/data.parquet
```
Each `data.parquet` holds **all rows for that deployment-month**, deduped. A month "seals"
naturally when the next month starts a new `ym=` dir; within a month the single file is grown
by read-modify-write.

## `TrainingStore` changes

- **`append_snapshot(snapshot, *, written_at=None)`** — compute the partition dir; read its
  `data.parquet` (empty frame if absent); append the new snapshot row; **dedupe on
  `issue_time` keeping the latest `written_at`**; sort by `issue_time`; atomic-rewrite to
  `data.parquet` (temp file + `os.replace`). (Parquet is immutable — "append" = read-modify-write.)
- **`write_labels(deployment_id, labels, *, written_at=None)`** — same read-modify-write into
  `labels/.../ym/data.parquet`, deduping on `(issue_time, lead_hour)`. Labels for one
  `issue_time` share a `ym`, so one file is touched per write (multi-`ym` inputs are grouped as
  today).
- **`read_snapshots(deployment_id, start=None, end=None)`** — glob `ym=*/data.parquet`, read +
  concat, apply the existing `[start,end]` filter and a **defensive** dedupe (cheap; the files
  are already deduped), round-trip the blobs, sort. Same return type / behavior.
- **`read_labels(...)`** — analogous; same public columns.
- **New `has_snapshot(deployment_id, issue_time) -> bool`** — read just that month's
  `data.parquet` (if any) and test membership of `issue_time`. Cheap (one small file).
- The atomic-write helper is unchanged except it writes the **fixed** `data.parquet` name (not
  a uuid). The `written_at` tiebreaker + UTC normalization + schema-version guard all carry over.

## `run_inference` — idempotent skip

```python
def run_inference(config, *, nwp, observations, store, forecast_path, issue_time) -> ForecastDocument | None:
    if store.has_snapshot(config.deployment_id, issue_time):
        # Already collected this HRDPS run — skip fetch/build/publish/append (idempotent).
        return None
    ... (unchanged: build_snapshot → build_features → baseline → assemble → write_forecast → append_snapshot) ...
    return doc
```
`main()` calls it and tolerates `None` (logs "already collected for <issue_time>"; no error).
Net effect with the hourly cron: each distinct HRDPS run is fetched/published/logged **once**;
the ~5 redundant hourly re-runs short-circuit before any network I/O.

## Action change — manage the branch as state

`.github/workflows/inference.yml` (the final commit/push step): after the inference loop,
**replace** the branch with a single fresh commit of `./store`:
```bash
cd store
git config user.name  "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
if git diff --quiet HEAD -- 2>/dev/null && [ -n "$(git rev-parse -q --verify HEAD || true)" ] && git diff --cached --quiet; then
  echo "No store changes; nothing to push."
else
  # Replace history with one commit of the current state (derived data; provenance is in the rows).
  git checkout -q --orphan _state
  git add -A
  git commit -q -m "data: store snapshot $(date -u +%FT%TZ)"
  git push -q --force origin _state:training-data
fi
```
(The exact "no-op detection" is a plan detail; the requirement: a changed store force-pushes a
single-commit `training-data`; an unchanged run pushes nothing.) The initial
`git clone --depth 1 --branch training-data` still fetches the current state to read-modify-write
against; the first-run orphan bootstrap is unchanged.

## Data flow (revised)

Action (hourly, single serialized job) → clone `training-data`→`./store` (current state) →
per deployment: `run_inference` → **`has_snapshot`? skip** : (build → baseline → publish →
`append_snapshot` **read-modify-writes** this month's `data.parquet`) → if `./store` changed,
**force-push one commit** to `training-data`. Tip: ~12 small files/deployment/year; git history:
always 1 commit.

## Error handling

- Read-modify-write reads the existing month file; a corrupt/unreadable `data.parquet` surfaces
  the read error (the run fails; next hour retries — the prior good state is still on the branch
  since the failed run never force-pushed).
- Atomic rewrite (temp + `os.replace`) — a crash mid-write never leaves a half-written
  `data.parquet`; the prior file stays intact.
- `has_snapshot` on an absent partition → `False` (normal for a new month/deployment).
- A failed `run_inference` (missing NWP) raises before any store write → no force-push that hour.

## Testing (offline, `tmp_path`)

- **Coalescing:** two `append_snapshot` calls for the same deployment-month → **one**
  `data.parquet` file containing **both** rows (`read_snapshots` returns 2, sorted by issue_time).
- **Write-time dedupe:** append the same `issue_time` twice (distinct `written_at`) → one row,
  latest wins; assert exactly one file.
- **Cross-month:** appends in two months → two files; `read_snapshots(start,end)` filters.
- **`has_snapshot`:** true after an append, false for an unknown issue_time / empty store.
- **Labels:** read-modify-write + `(issue_time, lead_hour)` dedupe; round-trip; empty → empty frame.
- **schema_version guard / UTC normalization / `start<=end`** still hold (carried over).
- **`run_inference` idempotent skip:** pre-seed the store with `issue_time`, pass a `FakeNWP`
  whose `fetch_forecast` raises if called → `run_inference` returns `None` without fetching,
  publishes nothing, logs nothing new.
- Full gate: ruff format/check, `lint-imports`, pyright strict, pytest.
- The Action force-push is infra — validated by a `workflow_dispatch` run (operator step), not pytest.

## Documentation updates (same PR, per CLAUDE.md)

- **ADR-0018** — "Training store coalesces to one file per deployment-month (read-modify-write,
  write-time dedupe); the inference Action manages the `training-data` branch as state
  (force-push a single commit); logging is idempotent." Records the revision + rationale
  (hourly re-logging + tiny-file pile-up + git binary-rewrite bloat).
- **Amend ADR-0015** with a header note: append-only-uuid + read-time dedupe + the
  compaction-future-work item are **superseded by ADR-0018** (coalesced read-modify-write).
- **Amend ADR-0017** with a note: the Action manages the branch via force-push of a single
  state commit (not accumulating append commits).
- **CONTEXT.md "Training store":** one `data.parquet` per deployment-month, coalesced
  read-modify-write, write-time dedupe; the branch is force-pushed state (ADR-0018).

## Decomposition (for the plan)

1. `TrainingStore` snapshots: coalesced `append_snapshot` + `read_snapshots` + `has_snapshot`
   (read-modify-write, write-time dedupe) + tests.
2. `TrainingStore` labels: coalesced `write_labels` + `read_labels` + tests.
3. `run_inference` idempotent skip (`-> ForecastDocument | None`) + `main()` `None` handling + test.
4. `.github/workflows/inference.yml` force-push-state.
5. ADR-0018 + ADR-0015/0017 amendments + CONTEXT.

## Open items deferred to the plan

- The Action's exact no-op detection (skip force-push when `./store` is unchanged vs the cloned
  state) — a shell detail; requirement is "changed → force-push, unchanged → nothing".
- Whether `read_*` keeps the defensive dedupe (lean: yes — cheap insurance against a future
  multi-writer or partial-write edge) or drops it now that writes dedupe (lean against dropping).
- Existing store tests written against the uuid layout (dedupe-keeps-latest, etc.) — update them
  to the coalesced layout rather than delete (the behaviors still hold; the file-count assertions
  change from "many files, deduped on read" to "one file, deduped on write").
