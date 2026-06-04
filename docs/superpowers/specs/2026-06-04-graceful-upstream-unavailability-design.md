# Graceful upstream unavailability in the inference pipeline

**Date:** 2026-06-04
**Status:** Approved
**Trigger:** The 2026-06-04 06:24 UTC inference Action run died red with a raw
`SourceUnavailable` traceback when Open-Meteo returned 502 Bad Gateway, even though the
system is designed to tolerate a missed hourly run (one success per ~6 h HRDPS cycle
window suffices).

## Problem

Two gaps compound when an upstream (Open-Meteo, ECCC) has a transient outage:

1. **Retries give up too fast.** `connectors/http.py` uses urllib3
   `Retry(total=3, backoff_factor=0.5)` — all retries complete within ~3.5 s. Any
   outage longer than a few seconds fails the request.
2. **Expected unavailability is treated as failure.** `SourceUnavailable` /
   `ForecastUnavailable` escape `pipelines/inference.py` as an unhandled traceback:
   the Action goes red, the operator gets a failure email, and — because the workflow's
   deployment loop runs under `set -euo pipefail` — every deployment after the failing
   one is skipped and nothing is published.

The intended behavior (already documented in `_latest_hrdps_issue_time`'s docstring and
`inference.yml`'s schedule comment): a missed run is fail-safe — the forecast simply
remains non-updated and the next hourly run retries.

## Decision

Upstream unavailability is **expected weather**, not failure. Two changes:

### 1. Explicit retry loop in `connectors/http.py` (~2.5 min budget)

Replace the urllib3 `Retry` status/connect retry machinery with a single explicit
backoff loop in `_do_get`:

- **Schedule:** attempt, then sleep 5 s / 10 s / 20 s / 40 s / 80 s before each retry —
  6 attempts over ~155 s of backoff (~2.6 min wall clock plus request time).
- **Retry on:** HTTP 5xx, connection errors, timeouts (the transient class).
- **Do not retry on:** 4xx (won't heal) or other request errors — raise immediately.
- **Exhaustion message:** the final `SourceUnavailable` includes the URL, the number of
  attempts, and the total elapsed retry time, e.g.
  `Gave up after 6 attempts over 158s fetching '<url>': 502 Server Error ...` —
  this is the payload the pipeline-level warning surfaces.
- The schedule lives in a module constant (`_BACKOFF_SCHEDULE`); sleeping goes through a
  module-level reference monkeypatchable in tests. urllib3 `Retry` is removed so there is
  exactly one retry layer (no multiplicative retries).
- Note: the budget is **per request**. A snapshot issues a handful of requests (one NWP +
  one per station), so a total outage costs at most a few sequential budgets — acceptable
  inside an hourly cadence; GitHub's job timeout is the backstop.

### 2. Graceful skip in the `inference.py` entrypoint

In `main()` only (not `run_inference`, which stays exception-transparent for library
callers and tests), wrap the `run_inference` call:

- **Catch exactly** `SourceUnavailable` and `ForecastUnavailable` — both mean "upstream
  can't serve us right now" (the latter: the chosen HRDPS run isn't published yet).
- On catch: print a clear warning naming the deployment and carrying the connector's
  message (which now includes what request failed and how long it was retried), then
  **exit 0**. When running under GitHub Actions (`GITHUB_ACTIONS` env set), also emit a
  `::warning::` annotation so it surfaces in the run summary without failing the run.
- The forecast JSON for that deployment is simply not rewritten — the previously
  published document remains, and the dashboard's viewer-relative status pill already
  surfaces staleness to viewers (#35).
- **Everything else still fails loudly**: `StationNotFound` (config bug), schema errors,
  `KeyError`s, etc. propagate and turn the run red — those are bugs, and the failure
  email is wanted.

No `inference.yml` change is needed: with the failing deployment exiting 0, the
`set -euo pipefail` loop naturally continues to the next deployment and the publish step
still runs for whatever succeeded (per-deployment isolation falls out for free).

## Alternatives considered

- **Tuned urllib3 `Retry` + shell-level exit-code handling** (e.g. exit 75 treated as
  warn-and-continue in the workflow loop): urllib3 can't report elapsed retry time in the
  error, and routing failure policy through exit codes and `set -e` interplay is more
  machinery in a worse language. Rejected.
- **Republish the previous forecast with `status: "stale"`** on upstream failure: makes
  the stateless pipeline (ADR-0019) read its own output, and duplicates staleness
  signaling the dashboard already derives from `last_updated`. Rejected.

## Blast radius

- `backfill.py` and `training.yml` also surface `SourceUnavailable`; they are batch/
  supervised contexts where a loud failure is correct — **out of scope, unchanged**.
  They do inherit the longer per-request retry budget from `http.py`, which is harmless
  (slower to fail during a real outage, more likely to succeed during a blip).
- ADR: this is a failure-policy decision about the hourly product — record it as a new
  ADR (`0020-upstream-unavailability-warn-and-skip`) in the same PR (per CLAUDE.md).
- CONTEXT.md: no new domain terms; the `ok`/`stale`/`degraded` status vocabulary is
  untouched (a skipped run publishes nothing).

## Testing

- **http.py:** with sleeping monkeypatched and a fake session — (a) retries 502 then
  succeeds, returning the body; (b) exhausts the schedule and raises `SourceUnavailable`
  whose message contains the URL, attempt count, and elapsed time; (c) a 404 raises
  immediately with no retry/sleep; (d) connection errors retry like 5xx.
- **inference entrypoint:** with `run_inference` monkeypatched to raise
  `SourceUnavailable` / `ForecastUnavailable`, `main()` exits 0 and prints the warning;
  with an unrelated exception, `main()` propagates it (non-zero).
