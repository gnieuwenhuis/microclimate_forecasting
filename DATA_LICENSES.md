# Data Licenses & Attribution

This project uses third-party weather data under the terms below. Redistribution rights
differ by source, which is why **only derived products** (the forecast JSON and trained
model binaries) are published publicly, while the **raw data store is kept private**
(see `docs/adr/0009-public-derived-only-private-raw-store.md`).

Every public, data-bearing artifact (the forecast JSON's `attribution` field, the dashboard
footer) must carry the acknowledgments below.

## Open-Meteo — HRDPS forecasts via open API (CC-BY 4.0)

- **Source:** Open-Meteo.com public forecast API (`api.open-meteo.com`), serving HRDPS
  (and other NWP models) under the Creative Commons Attribution 4.0 International licence.
- **Licence:** CC-BY 4.0 — https://creativecommons.org/licenses/by/4.0/
- **Required attribution (verbatim string used in all public artifacts):**
  > Weather data by Open-Meteo.com (CC-BY 4.0)
- **Modification note:** data is normalized, unit-converted, and resampled to a single
  target grid cell before storage and use.
- **Redistribution:** permitted with attribution. The `training-data` branch (a
  redistribution of CC-BY-4.0 data) carries its own attribution notice
  (`scripts/training_store_attribution.txt`).
- Reference: https://open-meteo.com / https://github.com/open-meteo/open-meteo

## Environment and Climate Change Canada (ECCC / MSC) — HRDPS, station observations

- **Source:** HRDPS via MSC GeoMet / Datamart (inference) and via CaSPAr (historical seed);
  Environment Canada station observations (e.g. Lethbridge Airport, Climate ID 3033875).
- **Licence:** *Environment and Climate Change Canada Data Servers End-use Licence*
  (v2.1, September 2022) — worldwide, royalty-free, perpetual, commercial use permitted.
  Explicitly grants the right to copy, modify, publish, and distribute the data for any
  lawful purpose.
- **Required attribution:**
  > Data Source: Environment and Climate Change Canada
- **Redistribution:** permitted (raw and derived), with attribution.
- Reference: https://eccc-msc.github.io/open-data/licence/readme_en/

## ACIS — Alberta Climate Information Service (Agriculture and Irrigation)

> **Not used in v1.** ACIS was dropped after spike #3 showed it has no ungated *live-hourly*
> feed; the `lethbridge` deployment now uses ECCC observations only (ADR-0010). This section
> is retained for the deferred path (ACIS as a future daily-feature source).

- **Source:** ACIS station observations (target + neighbors), e.g. Lethbridge Demo Farm
  IMCIN (#9835), Picture Butte LITE (#710547), Iron Springs IMCIN (#9883), Blood Tribe
  IMCIN (#9747).
- **Terms:** use must be acknowledged/cited. The terms **do not explicitly grant or
  prohibit redistribution**, and ACIS gates its API against automation. Data may include
  AGI-supplied estimates in place of missing/invalid values.
- **Required citation:**
  > Data provided by Agriculture and Irrigation, Alberta Climate Information Service (ACIS)
  > https://acis.alberta.ca (month and year when data was retrieved)
- **Redistribution:** **not republished publicly** by this project (raw ACIS data stays in
  the private store) — conservative stance given the unsettled terms. Derived products are
  published with the citation above.
- Reference: https://acis.alberta.ca/data-disclaimer.jsp

## CaSPAr — Canadian Surface Prediction Archive (University of Waterloo)

> **Not used — superseded by ADR-0019.** Historical HRDPS data is now sourced via the
> Open-Meteo API (see above); CaSPAr is no longer used as a data source and its citation
> is dropped from all public artifacts. This section is retained for the record.

- **Source:** historical HRDPS seed (2017-05-22 onward). CaSPAr is a convenience archive of
  ECCC numerical forecasts; the underlying data is ECCC-licensed (above).
- **Terms:** free registration required to access; citation requested.
- **Required citation (no longer used):**
  > Mai et al. (2020). The Canadian Surface Prediction Archive (CaSPAr): A Platform to
  > Enhance Environmental Modeling in Canada and Globally. *Bulletin of the American
  > Meteorological Society*, 101(3), E341–E356.
- **Redistribution:** the underlying data rides on the ECCC licence; cite both ECCC and
  CaSPAr. Raw archive data is held in the private store.
- Reference: https://caspar-data.ca

---

*Disclaimer:* all source data is provided "as is" with no warranties. This project's
forecasts are derived products and carry no warranty.
