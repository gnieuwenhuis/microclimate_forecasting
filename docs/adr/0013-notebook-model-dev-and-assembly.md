# 13. Local notebook is the model-dev surface; assembly is the shared seam

- **Status:** Accepted
- **Date:** 2026-05-31
- **Relates to:** ADR-0004 (two LightGBM models), ADR-0011 (snapshot normalization
  boundary), ADR-0012 (feature builder read-time transform), ADR-0009 (private raw store).

## Context

Model development needs a way to train the temp and PoP models locally and inspect them,
without that path forking from or bit-rotting against production. Reaching a trained model
also surfaced two feature-engineering steps ADR-0012 deferred: label attachment and
training-data assembly over a date range.

## Decision

1. **The notebook is a thin model-dev surface.** It holds no business logic — it calls
   shared, tested functions (`assemble_or_load`, the model wrappers, `evaluation.metrics`)
   and renders plots. A fast CI smoke test exercises that same assemble → fit → predict →
   metrics path on fake sources, so bitrot fails a test rather than waiting to be noticed.
2. **Training-data assembly is a shared seam** (`pipelines.training_data`). It performs the
   single training-only *future* read of target observations and labels the matrix; the
   future read is categorically absent from `build_snapshot`/`build_features` (ADR-0011).
   The same function the notebook calls will back the production training pipeline.
3. **`attach_labels` is pure and produces a labeled feature matrix** — distinct from the
   persisted `TRAINING_ROW` (raw snapshot + labels), which stays deferred with the private
   store. (`feature_matrix.py`'s docstring is corrected to stop conflating the two.)
4. **`predict` is row-based** (resolves ADR-0012's deferred open item; amends ADR-0004): the
   wrappers take feature-matrix rows and return one prediction per row; the inference
   pipeline owns `build_features` and reshapes to `{lead_hour: value}`.
5. **Evaluation uses a chronological three-way split** (`train | calib | test`): temp trains
   on `train+calib`; PoP trains on `train` and fits its isotonic calibrator on the disjoint
   `calib` slice; both are judged on `test` against the raw-HRDPS baseline, per lead hour.
6. **scikit-learn is adopted** for isotonic calibration and joblib model persistence.

## Consequences

- Deferred: the private training-store read/write path, the production training-pipeline
  orchestration / publish gate / publication, and walk-forward CV.
- Locally-trained models are throwaway (gitignored); promotion to a registry is future work.
- The notebook is authored as a jupytext percent-format `.py` (clean diffs, openable as a
  notebook); generated `.ipynb` files are gitignored.
