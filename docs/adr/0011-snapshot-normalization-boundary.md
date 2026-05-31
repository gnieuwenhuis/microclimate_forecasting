# 11. build_snapshot is the normalization boundary; the as-of obs read is the skew guarantee

- **Status:** Accepted
- **Date:** 2026-05-30
- **Informed by:** PRD issue #12; the grilling session that produced it.
- **Relates to:** ADR-0007 ("one HRDPS spec"), the FeatureSnapshot contract.

## Context

`features.build_snapshot` is the single producer of `FeatureSnapshot`, used by both the
training and inference pipelines. Two design questions had to be settled before
implementation: (1) how much feature engineering it performs, and (2) how it reads
observations for past vs present `issue_time` without introducing train/serve skew or
label leakage.

## Decision

**1. build_snapshot is the normalization / IO / as-of boundary — not the feature-engineering
step.** It holds raw, canonicalized values only: NWP forecast values flattened per lead hour
(`nwp_{var}_h{lead}`), lag-windowed observations on a fixed hourly grid
(`obs_{station_id}_{var}_lag{k}`) each with a presence mask, target static values, and `t0`
cyclical encodings. Derived features (dewpoint depression, pressure tendency, advection,
per-lead-hour encodings) and the explode-to-per-lead-hour-rows transform are downstream pure
functions of the snapshot. This keeps the network-touching, skew-critical code small and
hermetically testable, and lets new derived features be added without touching connectors.

**2. The shared builder reads observations only via as-of `fetch_historical(start, end=issue_time)`
— never `fetch_live`.** Because `fetch_historical` guarantees no rows after `end`, the obs path
is byte-identical whether `issue_time` is years in the past (training) or the current hour
(inference). `fetch_live` is `now`-bounded and would leak future rows for any past `issue_time`,
so it is categorically unsafe for the shared builder; it remains in the `ObservationSource`
contract for other callers. A defensive `timestamp <= issue_time` filter backs the guarantee.
The NWP live-vs-historical choice stays the caller's job, made by injecting the right
`NWPSource`.

## Consequences

- Train/serve skew is eliminated by construction at a second point (after ADR-0007's "one HRDPS
  spec"): one obs code path, one normalization function.
- Degradation is a deliberate decision here, not in connectors: a missing NWP backbone hard-fails
  (`ForecastUnavailable`/`SourceUnavailable` propagate); a transient obs `SourceUnavailable` or an
  empty window degrades that station to NaN+masks; a `StationNotFound` hard-fails (loud config
  bug); when every obs source fails, an NWP-only snapshot is still emitted.
- A downstream "build features from the snapshot" step is now required (separate work item) to
  produce the per-lead-hour model-input rows and derived features.
