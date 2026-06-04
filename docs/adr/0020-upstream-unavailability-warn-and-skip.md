# 20. Upstream unavailability is not failure: bounded retry, then warn-and-skip

- **Status:** Accepted
- **Date:** 2026-06-04
- **Amends:** ADR-0019 (stateless hourly inference — this records its failure policy)

## Context

On 2026-06-04 an Open-Meteo 502 killed the 06:24 UTC inference Action run with a raw
`SourceUnavailable` traceback. The design already declared a missed hourly run fail-safe —
capturing each HRDPS 6-hourly cycle needs only one success per ~6 h window, and
`_latest_hrdps_issue_time` documents "the next hourly Action run retries". But the
implementation contradicted the design in three ways: retries gave up after ~3.5 s
(urllib3 `Retry(total=3, backoff_factor=0.5)`); the expected-unavailability exception
escaped as an unhandled traceback, turning the run red and emailing the operator; and the
workflow's `set -euo pipefail` deployment loop aborted every deployment after the failing
one, skipping publication entirely.

## Decision

Upstream unavailability is **expected weather**, not failure.

1. **Bounded explicit retry in `connectors/http.py`.** One backoff loop (no urllib3
   `Retry` layer): 6 attempts with 5/10/20/40/80 s sleeps — ~2.5 min budget per request.
   Transient = HTTP 5xx, connection error, timeout; 4xx and other request errors fail
   immediately. The exhaustion `SourceUnavailable` names the URL, attempt count, and
   elapsed retry time. The budget is per request; a snapshot's handful of requests can at
   worst spend a few sequential budgets, acceptable inside an hourly cadence.
2. **Warn-and-skip in the inference entrypoint.** `pipelines/inference.py` `main()` —
   and only `main()`; `run_inference` stays exception-transparent — catches exactly
   `SourceUnavailable` and `ForecastUnavailable` (the chosen HRDPS run not published
   yet), prints a warning naming the deployment and the connector's message, emits a
   `::warning::` annotation under GitHub Actions, and exits 0. The forecast JSON is
   simply not rewritten; the dashboard's viewer-relative status pill already surfaces
   staleness from `last_updated`. Everything else — `StationNotFound`, schema errors,
   `KeyError` — propagates and turns the run red: those are bugs and the failure email
   is wanted.

Per-deployment isolation falls out for free: each deployment is a separate process in the
workflow loop, and a graceful skip exits 0, so the loop continues and the publish step
still runs for whatever succeeded. `inference.yml` is unchanged.

## Consequences

- A transient upstream blip costs at most one hourly slot per affected deployment, with a
  green run and a visible warning annotation instead of a red run and an email.
- Batch contexts (`backfill.py`, training) inherit the longer per-request retry budget —
  slower to fail during a real outage — but keep loud failure, which is correct for
  supervised runs.
- Real outages lasting a full HRDPS cycle (~6 h) now surface only through dashboard
  staleness, not failure emails. If silent staleness ever proves insufficient, add an
  explicit staleness alert — do not revert to failing the run.

## Alternatives rejected

- **Tuned urllib3 `Retry` + shell exit-code routing** (`EX_TEMPFAIL` handling in the
  workflow loop): cannot report elapsed retry time, and encodes failure policy in
  `set -e` shell interplay rather than one Python site.
- **Republish the previous forecast with `status: "stale"`:** makes the stateless
  pipeline (ADR-0019) read its own output, and duplicates staleness signaling the
  dashboard already derives from `last_updated`.
