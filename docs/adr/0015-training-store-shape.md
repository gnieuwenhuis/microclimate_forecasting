# 15. Training store shape: raw-snapshot blob + separate labels table, path-based

- **Status:** Accepted
- **Date:** 2026-05-31

> **Revised by ADR-0018:** the store now coalesces to one `data.parquet` per deployment-month
> via read-modify-write with write-time dedupe (not append-only `<uuid>.parquet` + read-time
> dedupe), and the "compaction is future work" note is resolved (coalescing is the compaction).

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
