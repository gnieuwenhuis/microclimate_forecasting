# 14. HRDPS solar is accumulated J/m²; nwp_core de-accumulates to mean W/m²

- **Status:** Accepted
- **Date:** 2026-05-31
- **Relates to:** ADR-0007 (one HRDPS spec — seed/live parity), the shared nwp_core core.

## Context

`nwp_core` originally treated `solar_radiation_wm2` as an instantaneous W/m² flux
(pass-through). A spike decoding real MSC Datamart HRDPS GRIB2 (run 2026-05-31 18Z) with
eccodes showed `DSWRF` decodes as `ssrd` = **accumulated downward shortwave J/m² from run
start** — so pass-through was wrong by ~3 orders of magnitude.

## Decision

`nwp_core` de-accumulates solar exactly like precip — `solar(h) − solar(h−1)` — then divides
by 3600 s to yield the **mean W/m² over the hour**, clamped ≥ 0. This is a shared-core change
applying to **both** HRDPS connectors (Datamart live + CaSPAr seed). The Datamart connector is
verified against real data here; the CaSPAr connector inherits the behavior and confirms its
own solar encoding when its sample lands (subsystem-A gate).

## Consequences

- Solar values are now physically correct mean hourly fluxes.
- Requires `h−1` in the dataset (already required for precip de-accumulation — no new constraint).
- If a future HRDPS source encodes solar instantaneously, this becomes per-connector config;
  deferred (YAGNI) until proven.
