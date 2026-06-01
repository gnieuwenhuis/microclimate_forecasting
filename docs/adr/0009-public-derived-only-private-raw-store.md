# 9. Public surface is derived products only; raw data store is private

- **Status:** Accepted — **store-is-private decision superseded by ADR-0017**
- **Date:** 2026-05-30
- **Amends:** ADR-0003 (storage homes), ADR-0007 (training store location)

> **Amendment (ADR-0017):** the raw store is now **public** (a `training-data` branch in the
> main repo), not a private repo. ADR-0009's private-store decision was driven by ACIS, which
> ADR-0010 dropped; the store now holds only ECCC-redistributable data (attribution still
> required). The `DATA_REPO_TOKEN` / separate-private-repo machinery below is retired.

## Context

The system runs on a public GitHub repo (free unlimited Actions minutes + free Pages).
That makes every branch world-readable — including the `training-data` store, which holds
raw HRDPS forecasts and raw station observations. Redistribution rights differ by source:

- **Environment Canada / HRDPS** (GeoMet, Datamart, CaSPAr-origin) — the *ECCC Data
  Servers End-use Licence v2.1* explicitly grants a worldwide, royalty-free, perpetual
  right to copy, publish, and distribute the data for any lawful purpose, **with
  attribution** (`Data Source: Environment and Climate Change Canada`). Clearly
  redistributable.
- **ACIS** (target + neighbor observations) — terms *require* citation but contain **no
  explicit grant or prohibition** of redistribution. They gate their API against
  automation, and they backfill missing values with estimates. Republishing raw ACIS data
  in bulk is a legal gray area and against the spirit of their access controls.
- **CaSPAr** — requires free registration to access and citation of Mai et al. (2020); the
  data it serves is ECCC-licensed.

Derived products (the forecast JSON, the trained model binaries) are transformations of the
data, redistributable under all three sources' terms **with attribution**. Only the **raw
ACIS observations** in a public store create real ambiguity.

## Decision

**Publish derived products only; keep the raw data store private.**

- **Public** (main repo + `gh-pages` + Releases): source code, dashboard, the forecast JSON,
  `registry.json`, and model binaries — all code or derived works, safe under all three
  licences with attribution.
- **Private** (a separate private repo): the raw training store (logged HRDPS snapshots,
  raw observations, the CaSPAr seed). The Action writes to it with a token held in Actions
  secrets.
- **Attribution is mandatory and built in**, not optional:
  - a root **`DATA_LICENSES.md`** stating the ECCC end-use licence acknowledgment, the ACIS
    citation, and the CaSPAr citation;
  - an **`attribution` field on the published forecast JSON** (`ForecastDocument`);
  - an **attribution line rendered in the dashboard** footer.

## Consequences

- **ADR-0003 amended:** the four homes become — *public* main repo (source/dashboard),
  *public* `gh-pages` (forecast JSON + dashboard + `registry.json`), *public* Releases (model
  binaries), and a *private* repo (raw training store). Three public homes carry only
  derived/code; the raw store is the one private home.
- **ADR-0007 amended:** the training store lives in a private repo, not a public
  `training-data` branch. The inference pipeline (logger) and training pipeline read/write it
  via a token (Actions secret `DATA_REPO_TOKEN`).
- The main repo stays public, so Actions minutes stay unlimited and Pages stays free; the
  private data repo is free to host (Actions there would be metered, but the Action runs in
  the public repo and only *pushes* to the private repo).
- The public forecast JSON must never embed raw observations — it carries only derived
  predictions plus attribution. (It already does; this makes it a rule.)
- If ACIS later publishes an explicit open licence, the raw store could be made public; the
  private/public split is the conservative default until then.

## Alternatives considered

- **Publish everything with attribution** — rejected: the ECCC portion is fine, but raw
  ACIS republication is an unresolved gray area and contrary to their API gating.
- **Publish raw ECCC but withhold raw ACIS** (partition the store) — rejected as the
  default: most faithful to the licences but the most complex (split storage + source-aware
  routing in the logger). Revisit only if a public raw ECCC store becomes valuable.
