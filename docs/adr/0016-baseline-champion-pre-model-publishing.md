# 16. Baseline raw-HRDPS forecaster is the initial published champion

- **Status:** Accepted
- **Date:** 2026-05-31
- **Relates to:** ADR-0003 (server-side inference, published JSON), ADR-0004 (two models),
  ADR-0007/0008 (logger-first pivot), ADR-0014 (raw-HRDPS baseline / nwp_pop_baseline).

## Context

CaSPAr is unavailable, so there is no historical seed and no trained model at launch
(ADR-0008 logger-first pivot). The service must still go live (the project's live-hourly
constraint) and start logging snapshots forward.

## Decision

The inference pipeline publishes a **raw-HRDPS baseline forecast** as the initial champion
until a trained model is promoted: temperature is the HRDPS 2 m passthrough; PoP is the raw
occurrence call (`nwp_precip_mm ≥ config.label.precip_occurrence_threshold_mm` → 1.0/0.0,
identical to `evaluation.nwp_pop_baseline` — the floor the trained model must beat). The
published `ForecastDocument.model_versions` is `{"temp": "baseline", "pop": "baseline"}` so
clients and the (future) registry can see the forecast is un-downscaled. Each run also logs
its snapshot to the training store (ADR-0007/0015), accumulating labels forward.

## Consequences

- The service is live and verifiable from day one, before any training; the trained model
  later swaps in via champion/challenger (ADR-0006) once it beats this baseline.
- `status` is `"ok"` for a successful baseline run; the `degraded`/`stale` signals (obs-source
  failure once trained models depend on obs; run freshness) are deferred to later work.
- This first slice runs in-process to local paths; the registry/champion-loading and the
  private-repo + gh-pages git sync (the GitHub Action, ADR-0009) are separate follow-on specs.
- **Update (2026-06-03):** the *training* side is now implemented — `pipelines.training.run_training`
  runs the champion/challenger publish gate (ADR-0006) and, on promotion, publishes the champion
  model (GitHub Release asset) and `registry.json` (gh-pages), with the training store persisted on
  the public `training-data` branch (ADR-0017/0018). The **inference side still reads the baseline**;
  swapping inference to load the registry/champion remains a separate slice.
