# 18. Training store coalesces to one file per deployment-month; branch is force-pushed state

- **Status:** Accepted
- **Date:** 2026-05-31
- **Revises:** ADR-0015 (append-only-uuid + read-time dedupe).
- **Relates to:** ADR-0007/0008 (logger), ADR-0017 (public training-data branch).

## Context

ADR-0015 wrote a new `<uuid>.parquet` per append and deduped on read. Driven hourly by the
inference Action — which also re-logs the same 6-hourly HRDPS `issue_time` ~6× — that piles
up thousands of tiny, often-duplicate files (slow reads). Coalescing into one growing file via
read-modify-write would fix reads but, on a git branch, re-storing a growing binary Parquet
each commit bloats history.

## Decision

1. **Coalesce to one `data.parquet` per `deployment_id`/`ym` (month)**, grown by
   **read-modify-write** with **write-time dedupe** (snapshots on `issue_time`; labels on
   `(issue_time, lead_hour)`), latest `written_at` wins. Reads glob `ym=*/data.parquet`.
   (Supersedes ADR-0015's append-only-uuid layout and its "compaction is future work" note —
   coalescing is the compaction.)
2. **Idempotent logging:** `run_inference` skips (no fetch/build/publish/append) when the
   `issue_time` is already stored (`TrainingStore.has_snapshot`), so each distinct HRDPS run
   is collected once.
3. **The Action manages `training-data` as state:** it force-pushes a single commit of the
   current store each run, so git history stays bounded despite read-modify-write. The branch
   is derived, forward-regenerable data; provenance is in each row (`issue_time`/`written_at`),
   not git history.

## Consequences

- ~12 small files/deployment/year (fast reads); git history ≈ one commit.
- **Single-writer only:** read-modify-write + whole-branch force-push assume one serialized
  writer (the Action is one job). Concurrent writers are out of scope (would need rework).
- Monthly window; weekly (`ym`→`yw`) is a trivial future knob.
