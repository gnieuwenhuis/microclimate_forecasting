# 12. The feature builder is a read-time transform; the training store holds raw snapshots

- **Status:** Accepted
- **Date:** 2026-05-31
- **Relates to:** ADR-0011 (snapshot is the normalization boundary), ADR-0004 (two LightGBM
  models, long-format rows), ADR-0006 (champion/challenger).

## Context

ADR-0011 made `build_snapshot` the normalization/as-of boundary holding raw canonicalized
values, and named a follow-on work item: a downstream step that produces the per-lead-hour
model-input rows and derived features. Three questions had to be settled before building it:
when the transform runs, how the derived-feature set is versioned, and whether values are
statistically scaled.

## Decision

**1. The feature builder runs at read time; the training store holds raw snapshots.**
`features.build_features(snapshot, config)` is a pure function executed at training-read time
and at inference — never at write time. The training store persists raw snapshot values (+
labels); derived features are recomputed on read. This keeps a single shared feature code
path (the reason `build_snapshot` exists), makes feature iteration cheap (retrain only — no
re-log or backfill), keeps the store feature-version-independent and smaller, and forces the
transform to be deterministic and self-contained.

**2. The derived-feature set is versioned independently** via `FEATURE_SCHEMA_VERSION`,
distinct from `SNAPSHOT_SCHEMA_VERSION`. Models record the feature version they trained on so
a champion built against a stale feature set is refused rather than silently misread.
`build_features` also rejects a snapshot whose `schema_version` it does not recognise.

**3. No statistical scaling.** LightGBM is tree-based; the builder adds derived columns and
explodes to rows but does not standardize values. "Normalization" in ADR-0011 means
canonicalization (units, variable order), already done upstream.

## Consequences

- Train/serve skew is eliminated at this layer too: the column set is deterministic from
  config, so a training-`t0` and an inference-`t0` snapshot yield identical columns (a guarded
  parity invariant).
- The transform is import-pure (no `connectors`, no I/O), enforced by an import-linter
  forbidden contract.
- A label-attachment step (join observed labels onto the feature matrix to form `TRAINING_ROW`)
  remains a separate downstream work item.
- **Resolved (ADR-0013):** `model.predict` is row-based and the pipeline owns the
  `build_features` call, passing rows to both `fit` and `predict`. A label-attachment step
  (`features.attach_labels`, pure) and a training-data assembler (`pipelines.training_data`)
  now produce the labeled feature matrix; the persisted `TRAINING_ROW` store remains deferred.
