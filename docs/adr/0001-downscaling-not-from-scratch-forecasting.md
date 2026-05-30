# 1. Core task is downscaling/post-processing, not from-scratch forecasting

- **Status:** Accepted
- **Date:** 2026-05-30

## Context

The goal is a free microclimate temperature and PoP forecast. There are three candidate
framings:

1. **Downscaling / post-processing** — feed an existing official forecast (NWP) plus local
   observations into a model that learns the *local correction*.
2. **Pure-ML time-series forecasting** — predict future weather purely from station
   observation history, with no NWP input.
3. **Nowcasting** — estimate *current* conditions at an unmonitored point.

The strongest references gathered for this project (post-processing of air-temperature
forecasts; post-processing of NWP precipitation with split occurrence/amount heads) are
all framing (1). The pure-ML references are weaker and self-limit to ~0–6 h skill before
hitting a "station-only blindness" ceiling. A from-scratch model would be competing
directly against Environment Canada's free public forecast — a contest it is likely to
lose, leaving the product with no reason to exist.

## Decision

The core task is **downscaling / post-processing**. Model input = an official NWP forecast
at the target location + local observations. Model output = locally bias-corrected hourly
temperature and PoP. We do not attempt to forecast future weather from station history
alone, and we do not attempt to out-forecast a weather agency.

## Consequences

- The most important input feature is the existing NWP forecast; an NWP backbone is a hard
  dependency (see ADR-0003 — HRDPS is implied; the NWP choice is HRDPS).
- The system inherits the NWP's forecast horizon (so a useful 48 h product is possible
  despite the station-only ~6 h ceiling — the NWP carries the long-horizon skill).
- Value is measured as **skill over the raw NWP**: if the model can't beat raw HRDPS, it
  adds nothing (see the publish gate).
- Training requires historical *(forecast, observed-outcome)* pairs, which constrains the
  data sources (see ADR-0007).
- Prediction targets are limited to locations with ground-truth labels, i.e. real stations
  (see the multi-deployment/target decisions in ADR-0006).

## Alternatives considered

- **Pure-ML time-series forecasting** — rejected: research-grade risk, likely beaten by the
  free official forecast, short skill horizon.
- **Nowcasting** — rejected: useful but not "forecasting," and not the product goal.
