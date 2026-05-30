# 4. Two separate LightGBM models, not a multi-task neural net

- **Status:** Accepted
- **Date:** 2026-05-30

## Context

The initial sketch proposed a single multi-task neural network with two heads (temp
regression + PoP classification), chosen mainly because neural nets convert to TFLite.
ADR-0003 retired the TFLite requirement, removing that constraint.

Temperature and PoP are different problems: temperature is a smooth continuous regression
judged by MAE/RMSE; PoP is a zero-inflated binary-occurrence problem judged by
*calibration* (Brier score, reliability) and leaning on different features (dew-point
depression, pressure tendency). Multi-task weight-sharing mainly pays off under tight
model-size budgets, which no longer exist.

## Decision

Use **two separate LightGBM models per deployment**: a temperature **regressor** and a
precipitation-occurrence **classifier** with an explicit probability-**calibration** stage
(e.g. isotonic/Platt). Both take **`lead_hour` as a feature** and are trained on
long-format rows (one row per `(t₀, lead_hour)`), so a single model spans all 48 lead
hours.

## Consequences

- Each model is independently tunable, independently evaluable, and independently
  publishable (see ADR-0006 consequence: per-task champion/challenger).
- LightGBM handles missing values natively, pairing cleanly with the missingness masks in
  the feature snapshot.
- Calibration is a required pipeline stage, not optional — it *is* the PoP deliverable.
- Two model artifacts per deployment instead of one; mixed model vintages are possible in a
  single published forecast (acceptable — it enables independent improvement).
- Training runs on free CPU runners in seconds-to-minutes; no GPU needed.

## Alternatives considered

- **Multi-task neural net** — rejected: no longer needed (no TFLite), couples two
  dissimilar problems, harder to calibrate and evaluate per-task.
- **One model per lead hour (48 models/task)** — rejected: no structure sharing across
  horizons, 96 artifacts per deployment.
