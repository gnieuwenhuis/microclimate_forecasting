# Graceful Upstream Unavailability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A transient upstream outage (e.g. Open-Meteo 502) is retried for ~2.5 min and, if still failing, logs a clear warning and exits 0 — the forecast stays non-updated and the Action stays green; real bugs still fail red.

**Architecture:** Replace the urllib3 `Retry` in `connectors/http.py` with one explicit backoff loop (6 attempts over ~155 s of sleeps) whose exhaustion error carries URL/attempts/elapsed. In `pipelines/inference.py` `main()` only, catch `SourceUnavailable`/`ForecastUnavailable`, print a warning (plus a `::warning::` annotation under GitHub Actions), and return — exit 0. Record the policy as ADR-0020. Spec: `docs/superpowers/specs/2026-06-04-graceful-upstream-unavailability-design.md`.

**Tech Stack:** Python 3.12, `requests`, pytest (`monkeypatch`/`MagicMock`/`capsys`), uv. Gates: `uv run ruff check .`, `uv run ruff format --check .`, `uv run lint-imports`, `uv run pyright`, `uv run pytest`.

**Project conventions that bind this work:**
- `main` rejects direct pushes — integrate via PR.
- Layered imports enforced by import-linter: `http.py` is in `connectors` (may import `contracts`/`config`); `pipelines` may import everything below.
- pyright runs **strict** — annotate everything, including tests.
- Network-marked tests are deselected by default; everything here is offline (mock `_SESSION.get`).

---

### Task 1: Branch setup — commit spec + plan

The spec and plan exist only as untracked files in the main checkout at
`/Users/gregn/Documents/microclimate_forecasting`. If you are in a fresh worktree they will
be missing — copy them in.

**Files:**
- Add: `docs/superpowers/specs/2026-06-04-graceful-upstream-unavailability-design.md`
- Add: `docs/superpowers/plans/2026-06-04-graceful-upstream-unavailability.md`

- [ ] **Step 1: Ensure both docs exist in the working tree**

```bash
mkdir -p docs/superpowers/specs docs/superpowers/plans
[ -f docs/superpowers/specs/2026-06-04-graceful-upstream-unavailability-design.md ] || \
  cp /Users/gregn/Documents/microclimate_forecasting/docs/superpowers/specs/2026-06-04-graceful-upstream-unavailability-design.md docs/superpowers/specs/
[ -f docs/superpowers/plans/2026-06-04-graceful-upstream-unavailability.md ] || \
  cp /Users/gregn/Documents/microclimate_forecasting/docs/superpowers/plans/2026-06-04-graceful-upstream-unavailability.md docs/superpowers/plans/
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-04-graceful-upstream-unavailability-design.md \
        docs/superpowers/plans/2026-06-04-graceful-upstream-unavailability.md
git commit -m "docs: spec + plan for graceful upstream unavailability"
```

---

### Task 2: Explicit retry loop in `connectors/http.py`

**Files:**
- Modify: `src/microclimate/connectors/http.py` (full rewrite below)
- Test: `tests/connectors/test_http.py`

**Behavior being built:** `_do_get` attempts the GET up to 6 times. Transient failures
(HTTP 5xx, connection error, timeout) sleep 5/10/20/40/80 s between attempts, then raise
`SourceUnavailable` with URL + attempt count + elapsed time. Non-transient failures (4xx,
any other `RequestException`) raise `SourceUnavailable` immediately, no sleep. Sleeping
goes through module-level `_sleep` so tests patch it away.

- [ ] **Step 1: Add the autouse no-sleep fixture and new failing tests**

In `tests/connectors/test_http.py`, add right after the `_make_response` helper (the
existing transient-failure tests will start retrying once the loop lands — without this
autouse fixture they would really sleep ~155 s each):

```python
@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch backoff sleeping away — no test in this module may sleep for real."""
    monkeypatch.setattr("microclimate.connectors.http._sleep", lambda _s: None)
```

**Delete** `test_retry_adapter_is_mounted` (the urllib3 `Retry` adapter is being removed)
and append this new section at the end of the file:

```python
# ---------------------------------------------------------------------------
# Explicit backoff loop (ADR-0020)
# ---------------------------------------------------------------------------


def test_transient_502_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 502 is retried; the next attempt's body is returned."""
    mock_get = MagicMock(
        side_effect=[_make_response("bad", status_code=502), _make_response("recovered")]
    )
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    assert http_get("https://example.com/data") == "recovered"
    assert mock_get.call_count == 2


def test_backoff_sleeps_follow_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """A persistent 502 sleeps through the whole schedule, one attempt per slot + 1."""
    import microclimate.connectors.http as http_mod

    sleeps: list[float] = []
    monkeypatch.setattr("microclimate.connectors.http._sleep", sleeps.append)
    mock_get = MagicMock(return_value=_make_response("bad", status_code=502))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")

    assert sleeps == list(http_mod._BACKOFF_SCHEDULE)  # type: ignore[reportPrivateUsage]
    assert mock_get.call_count == len(http_mod._BACKOFF_SCHEDULE) + 1  # type: ignore[reportPrivateUsage]


def test_exhaustion_message_names_url_attempts_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The give-up error carries the URL, attempt count, elapsed time, and the cause."""
    import re

    # _make_response builds a message-less HTTPError (str(exc) == ""), so attach a
    # realistic message here — the assertion below needs the cause text to survive.
    resp = _make_response("bad", status_code=502)
    resp.raise_for_status.side_effect = requests.HTTPError(
        "502 Server Error: Bad Gateway",
        response=resp,  # type: ignore[call-arg]
    )
    mock_get = MagicMock(return_value=resp)
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable) as exc_info:
        http_get("https://example.com/data")

    msg = str(exc_info.value)
    assert "https://example.com/data" in msg
    assert "6 attempts" in msg
    assert re.search(r"over \d+s", msg), msg
    assert "502" in msg


def test_http_404_fails_immediately_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """4xx won't heal: exactly one attempt, no sleeping."""
    sleeps: list[float] = []
    monkeypatch.setattr("microclimate.connectors.http._sleep", sleeps.append)
    mock_get = MagicMock(return_value=_make_response("nope", status_code=404))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")

    assert mock_get.call_count == 1
    assert sleeps == []


def test_connection_error_retries_like_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connection errors are transient: the full schedule is consumed before giving up."""
    import microclimate.connectors.http as http_mod

    mock_get = MagicMock(side_effect=requests.ConnectionError("refused"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")

    assert mock_get.call_count == len(http_mod._BACKOFF_SCHEDULE) + 1  # type: ignore[reportPrivateUsage]


def test_backoff_schedule_is_bounded() -> None:
    """Total backoff stays within the ~2-3 min budget (spec) — bounded, not unbounded."""
    import microclimate.connectors.http as http_mod

    total = sum(http_mod._BACKOFF_SCHEDULE)  # type: ignore[reportPrivateUsage]
    assert 60 <= total <= 300
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
uv run pytest tests/connectors/test_http.py -v
```

Expected: the six new tests FAIL (`AttributeError: ... has no attribute '_sleep'` from the
autouse fixture, since `http.py` doesn't define `_sleep` yet). Pre-existing tests will also
error for the same reason — that's fine at this step.

- [ ] **Step 3: Rewrite `src/microclimate/connectors/http.py`**

Replace the entire file with:

```python
"""Shared HTTP GET helper for L2 connectors (timeouts, explicit bounded backoff, descriptive UA)."""

from __future__ import annotations

import time
from collections.abc import Mapping

import requests

from microclimate.connectors.base import SourceUnavailable

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_USER_AGENT: str = (
    "microclimate-forecasting (+https://github.com/gnieuwenhuis/microclimate_forecasting)"
)

# Explicit (connect_timeout, read_timeout) in seconds.
_TIMEOUT: tuple[float, float] = (10.0, 30.0)

# Sleeps between attempts on transient failures (HTTP 5xx, connection error, timeout):
# 6 attempts over ~155 s of backoff (~2.6 min budget). Rides out short upstream blips
# (e.g. an Open-Meteo 502); for longer outages the next hourly Action run is the
# fallback and the caller warns-and-skips (ADR-0020).
_BACKOFF_SCHEDULE: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0)

# Module-level indirection so tests can patch sleeping away.
_sleep = time.sleep

# ---------------------------------------------------------------------------
# Module-level session (created once, shared across calls)
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = _USER_AGENT


# ---------------------------------------------------------------------------
# Internal shared request helper
# ---------------------------------------------------------------------------


def _is_transient(exc: requests.RequestException) -> bool:
    """True when the failure class can heal on retry (HTTP 5xx, connection, timeout)."""
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code >= 500
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


def _do_get(
    url: str, *, params: Mapping[str, str | int | float] | None = None
) -> requests.Response:
    """Perform an HTTP GET with bounded backoff on transient failures; return the Response.

    Transient failures (HTTP 5xx, connection errors, timeouts) are retried through
    ``_BACKOFF_SCHEDULE``; anything else (4xx, malformed requests) fails immediately.

    Raises:
        SourceUnavailable: On a non-transient failure, or once retries are exhausted —
            the exhaustion message names the URL, attempt count, and elapsed retry time.
    """
    start = time.monotonic()
    for attempt in range(1, len(_BACKOFF_SCHEDULE) + 2):
        try:
            response = _SESSION.get(url, params=params, timeout=_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            if not _is_transient(exc):
                kind = "HTTP error" if isinstance(exc, requests.HTTPError) else "request failed"
                raise SourceUnavailable(f"{kind} fetching {url!r}: {exc}") from exc
            if attempt > len(_BACKOFF_SCHEDULE):
                elapsed = time.monotonic() - start
                raise SourceUnavailable(
                    f"gave up after {attempt} attempts over {elapsed:.0f}s "
                    f"fetching {url!r}: {exc}"
                ) from exc
            _sleep(_BACKOFF_SCHEDULE[attempt - 1])
            continue
        return response
    raise AssertionError("unreachable: loop exits via return or raise")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def http_get(url: str, *, params: Mapping[str, str | int | float] | None = None) -> str:
    """Perform an HTTP GET and return the response body as text.

    Args:
        url:    Absolute URL to fetch.
        params: Optional query parameters to append to the URL.

    Returns:
        The response body as a decoded string.

    Raises:
        SourceUnavailable: On any network failure, timeout, or non-2xx HTTP status
            (transient failures are retried first — see ``_do_get``).
    """
    return _do_get(url, params=params).text


def http_get_bytes(url: str, *, params: Mapping[str, str | int | float] | None = None) -> bytes:
    """Perform an HTTP GET and return the response body as bytes.

    Useful for binary formats such as GRIB2.

    Args:
        url:    Absolute URL to fetch.
        params: Optional query parameters to append to the URL.

    Returns:
        The response body as raw bytes (``response.content``).

    Raises:
        SourceUnavailable: On any network failure, timeout, or non-2xx HTTP status
            (transient failures are retried first — see ``_do_get``).
    """
    return _do_get(url, params=params).content
```

Notes on what changed vs. the old file:
- `HTTPAdapter`/`Retry` imports, the `_RETRY` constant, and both `_SESSION.mount(...)`
  calls are **gone** — exactly one retry layer remains.
- The old per-exception-type `except` arms collapse into one `requests.RequestException`
  arm routed by `_is_transient`. `requests.RetryError` (still asserted by an existing
  test) is a `RequestException` → non-transient → immediate `SourceUnavailable`,
  preserving that test's behavior.

- [ ] **Step 4: Run the connector tests to verify they pass**

```bash
uv run pytest tests/connectors/test_http.py -v
```

Expected: ALL pass — the six new tests plus every pre-existing test (the old transient
tests now exercise retry-then-exhaust under the autouse no-sleep fixture).

- [ ] **Step 5: Run the wider suite to catch fallout in connector callers**

```bash
uv run pytest tests/connectors tests/features tests/pipelines -q
```

Expected: PASS. (`openmeteo`/`envcanada` tests stub at the `http_get` fetcher level, so the
loop change is invisible to them; this run confirms it.)

- [ ] **Step 6: Commit**

```bash
git add src/microclimate/connectors/http.py tests/connectors/test_http.py
git commit -m "feat(connectors): explicit ~2.5 min backoff for transient HTTP failures"
```

---

### Task 3: Graceful warn-and-skip in the inference entrypoint

**Files:**
- Modify: `src/microclimate/pipelines/inference.py` (imports + `main()` only — `run_inference` is untouched)
- Test: `tests/pipelines/test_pipelines_cli.py`

**Behavior being built:** `main()` catches exactly `SourceUnavailable` and
`ForecastUnavailable` from `run_inference`, prints a warning naming the deployment and
carrying the connector's message, emits a `::warning::` annotation when `GITHUB_ACTIONS`
is set, and returns normally (exit 0). Any other exception propagates.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/pipelines/test_pipelines_cli.py` with (the one
existing test is kept verbatim; a wiring helper and four tests are added):

```python
from __future__ import annotations

from pathlib import Path
from typing import NoReturn
from unittest.mock import MagicMock

import pytest

from microclimate.connectors.base import ForecastUnavailable, SourceUnavailable
from microclimate.pipelines import inference


def test_inference_cli_requires_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog"])  # no --deployment
    with pytest.raises(SystemExit):
        inference.main()


# ---------------------------------------------------------------------------
# Upstream unavailability is not failure (ADR-0020): warn and exit 0
# ---------------------------------------------------------------------------


def _wire_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point main() at a mocked config/sources so only run_inference's outcome matters."""
    monkeypatch.setattr("sys.argv", ["prog", "--deployment", "testdep"])
    monkeypatch.setenv("FORECAST_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    config = MagicMock()
    config.output.forecast_json = "forecast.json"
    config.target.connector_key = "obs_key"
    config.neighbors = []
    config.nwp.live_connector = "nwp_key"
    monkeypatch.setattr(inference, "load_deployment", MagicMock(return_value=config))
    monkeypatch.setattr(inference, "get_source", MagicMock())


def test_cli_source_unavailable_exits_zero_with_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SourceUnavailable → no exception, warning names the deployment and the cause."""
    _wire_cli(monkeypatch, tmp_path)
    cause = "gave up after 6 attempts over 155s fetching 'https://api.example': 502"

    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise SourceUnavailable(cause)

    monkeypatch.setattr(inference, "run_inference", boom)

    inference.main()  # returning (no SystemExit, no exception) IS exit 0

    out = capsys.readouterr().out
    assert "testdep" in out
    assert cause in out
    assert "upstream unavailable" in out


def test_cli_forecast_unavailable_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ForecastUnavailable (HRDPS run not published yet) gets the same graceful skip."""
    _wire_cli(monkeypatch, tmp_path)

    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise ForecastUnavailable("no leads available for issue_time")

    monkeypatch.setattr(inference, "run_inference", boom)

    inference.main()

    assert "upstream unavailable" in capsys.readouterr().out


def test_cli_emits_github_warning_annotation_under_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under GITHUB_ACTIONS a ::warning:: annotation is emitted; locally it is not."""
    _wire_cli(monkeypatch, tmp_path)

    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise SourceUnavailable("502")

    monkeypatch.setattr(inference, "run_inference", boom)

    inference.main()
    assert "::warning" not in capsys.readouterr().out  # local: no annotation noise

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    inference.main()
    assert "::warning" in capsys.readouterr().out


def test_cli_bug_exceptions_still_propagate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anything outside the expected-unavailability pair fails loudly (red run wanted)."""
    _wire_cli(monkeypatch, tmp_path)

    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise KeyError("missing feature column")

    monkeypatch.setattr(inference, "run_inference", boom)

    with pytest.raises(KeyError):
        inference.main()
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
uv run pytest tests/pipelines/test_pipelines_cli.py -v
```

Expected: the three graceful-path tests FAIL with `SourceUnavailable`/`ForecastUnavailable`
escaping `main()`; `test_cli_bug_exceptions_still_propagate` and the original
`test_inference_cli_requires_deployment` PASS.

- [ ] **Step 3: Implement the catch in `main()`**

In `src/microclimate/pipelines/inference.py`, extend the existing import from
`microclimate.connectors.base`:

```python
from microclimate.connectors.base import (
    ForecastUnavailable,
    NWPSource,
    ObservationSource,
    SourceUnavailable,
)
```

Then in `main()`, wrap the `run_inference(...)` call (currently the last statement) in:

```python
    try:
        run_inference(
            config,
            nwp=nwp,
            observations=observations,
            forecast_path=root / config.output.forecast_json,
            issue_time=issue_time,
            registry_path=registry_path,
            work_dir=work_dir,
        )
    except (SourceUnavailable, ForecastUnavailable) as exc:
        # ADR-0020: upstream unavailability is expected weather, not failure. Warn, leave
        # the published forecast non-updated, and exit 0 so the hourly Action stays green
        # and the next run retries. Anything else is a bug and must propagate loudly.
        message = (
            f"inference: upstream unavailable for deployment '{args.deployment}'; "
            f"forecast left non-updated, next hourly run retries "
            f"({type(exc).__name__}: {exc})"
        )
        print(message)
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::warning title=inference skipped ({args.deployment})::{message}")
```

(`os` is already imported in this module.)

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/pipelines/test_pipelines_cli.py tests/pipelines/test_inference.py -v
```

Expected: ALL pass (`test_inference.py` exercises `run_inference` directly, which is
untouched and stays exception-transparent).

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/pipelines/inference.py tests/pipelines/test_pipelines_cli.py
git commit -m "feat(pipelines): warn-and-skip on upstream unavailability in inference CLI"
```

---

### Task 4: ADR-0020

**Files:**
- Create: `docs/adr/0020-upstream-unavailability-warn-and-skip.md`

- [ ] **Step 1: Write the ADR**

Create `docs/adr/0020-upstream-unavailability-warn-and-skip.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/adr/0020-upstream-unavailability-warn-and-skip.md
git commit -m "docs(adr): ADR-0020 upstream unavailability warn-and-skip"
```

---

### Task 5: Full verification gates

No code changes — run every CI gate locally and fix anything that surfaces (if a fix is
needed, amend the relevant task's commit pattern: fix, re-run gates, commit as
`fix: <what>`).

- [ ] **Step 1: Lint, format, layer contract, types**

```bash
uv run ruff check . && uv run ruff format --check . && uv run lint-imports && uv run pyright
```

Expected: all clean. (Likely trip points: ruff `UP038` may prefer
`isinstance(exc, requests.ConnectionError | requests.Timeout)` — apply whatever ruff
dictates; pyright strict in tests — the provided tests are annotated, keep them so.)

- [ ] **Step 2: Full test suite**

```bash
uv run pytest
```

Expected: PASS, with network-marked tests deselected (default). Confirm no test slowed
dramatically — a real 155 s sleep sneaking into a test means a missing `_sleep` patch.

- [ ] **Step 3: Sanity-check README "Project status"**

```bash
grep -n -i "retry\|unavailab" README.md || true
```

Expected: nothing stale to update (this change completes no stubbed subsystem). If the
status section mentions inference failure behavior, align the wording; otherwise no edit.

---

### Task 6: Integrate via PR

`main` rejects direct pushes — everything lands through a PR.

- [ ] **Step 1: Push the branch and open the PR**

```bash
git push -u origin HEAD
gh pr create \
  --title "Graceful upstream unavailability: bounded retry + warn-and-skip (ADR-0020)" \
  --body "$(cat <<'EOF'
## Summary
- `connectors/http.py`: replace the ~3.5 s urllib3 `Retry` with one explicit backoff loop — 6 attempts over ~2.5 min on transient failures (5xx / connection / timeout); exhaustion error names URL, attempts, elapsed. 4xx fail immediately.
- `pipelines/inference.py` `main()`: catch `SourceUnavailable`/`ForecastUnavailable`, warn (plus `::warning::` annotation on GitHub Actions), exit 0 — forecast stays non-updated, next hourly run retries, the deployment loop continues. Real bugs still fail red.
- ADR-0020 records the policy; spec + plan in `docs/superpowers/`.

Trigger: 2026-06-04 06:24 UTC run died red on an Open-Meteo 502.

## Test plan
- [ ] `uv run pytest` (new: backoff schedule/exhaustion-message/404-immediate tests; CLI graceful-exit tests)
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run lint-imports && uv run pyright`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Wait for CI green, then merge** (user preference: merge after checks pass; external actions like merging need the user's explicit OK — ask before merging).
