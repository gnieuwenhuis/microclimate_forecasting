# 7. Self-accumulating training store: CaSPAr seed + inference logger

- **Status:** **Superseded by ADR-0019** (was: Accepted — store location amended by ADR-0009)
- **Date:** 2026-05-30

> **Superseded (ADR-0019, 2026-06-02):** CaSPAr is dead and Open-Meteo became the single HRDPS
> source. Both load-bearing ideas below are replaced: the **CaSPAr one-time seed** → **Open-Meteo
> Previous Runs backfill re-run at each retrain**, and the **inference logger** → **removed**
> (the inference pipeline is now stateless/publish-only). The training store survives but is fed
> by the retrain-time backfill (idempotent, additive), not by an hourly logger. The text below is
> retained for history only.

> **Amendment (ADR-0009):** the self-accumulating training store below lives in a
> **private repo** (written via a token), not a public `training-data` branch, because it
> holds raw observations whose redistribution rights are unsettled (ACIS). *(Also superseded:
> the store is public per ADR-0017/0018/0019 — its data is CC-BY-4.0/ECCC.)*

## Context

Downscaling (ADR-0001) needs historical *(forecast, observed-outcome)* pairs to train.
Historical HRDPS forecasts come from **CaSPAr**, but CaSPAr is a **queued bulk-request
research archive, not an API** — a monthly retrain cannot freshly pull from it on a cron.
Meanwhile, the inference pipeline (ADR-0003) already fetches a full feature snapshot every
hour.

## Decision

Make the inference pipeline double as a **logger**, and use CaSPAr only as a one-time seed:

- **CaSPAr = one-time historical seed** — backfill HRDPS + obs from 2017-05-22 to the
  deploy date, once.
- **Logger = ongoing source** — every hourly inference run persists the feature snapshot it
  built. A later labeling step joins realized observations once each `valid_time` passes,
  closing the loop into fully labeled training rows.

This produces a **training store** (partitioned Parquet, per deployment) that grows
automatically. The HRDPS specification from CaSPAr and from the live channels must be
identical (same variables/grid), or seed and logged data diverge.

## Consequences

- Ongoing training is fully decoupled from CaSPAr's request queue.
- A **fourth artifact home** is required: a `training-data` branch holding the Parquet
  store.
- A logged row isn't trainable until its `valid_time` + observation latency passes; the
  first ~48 h after a fresh deploy produce no new labeled data, so the CaSPAr seed carries
  the deployment until logged data accumulates.
- The training store carries a `schema_version`; a feature-contract change must account for
  previously logged data.

## Alternatives considered

- **Re-request CaSPAr each retrain** — rejected: CaSPAr's queued bulk model makes this
  impractical on a schedule.
- **Reanalysis (e.g. ERA5) as the training input** — rejected: reanalysis is a hindcast,
  not a forecast, introducing train/serve skew against live HRDPS.
