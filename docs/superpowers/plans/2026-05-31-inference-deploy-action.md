# Inference Deployment Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Note:** Task 2 (workflow YAML) is infra — it is validated by *running the Action*, not by pytest.

**Goal:** Make the hourly inference GitHub Action actually collect data — run `run_inference` per deployment and persist the snapshots to a public `training-data` branch via `GITHUB_TOKEN`.

**Architecture:** A small testable `main()` fix targets the latest available HRDPS run (HRDPS is 6-hourly, not hourly). The `.github/workflows/inference.yml` workflow (single job) installs eccodes, checks out/bootstraps the `training-data` branch into `./store`, runs inference for each deployment writing the store there, and commits+pushes new snapshots. ADR-0017 records the store-is-public decision (ACIS dropped → ECCC-redistributable), amending ADR-0009.

**Tech Stack:** GitHub Actions (ubuntu-latest), `uv`, eccodes (apt) for cfgrib, git (`GITHUB_TOKEN`). One Python helper + test; the rest is workflow YAML + docs.

---

## Conventions

- Task 1 is TDD (Python). Tasks 2–3 are YAML/docs — no unit tests; Task 2 is validated by a `workflow_dispatch` run (an operator step).
- Full gate before each commit: `uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`.
- Commit on branch `spec/inference-deploy-action` (main is PR-only); push only at Final Integration.

## File structure

**Modify**
- `src/microclimate/pipelines/inference.py` — add `_latest_hrdps_issue_time`; `main()` uses it.
- `tests/pipelines/test_inference.py` — unit test for the helper.
- `.github/workflows/inference.yml` — rewrite (single job, eccodes, store checkout/bootstrap, loop, commit+push).
- `README.md`, `CONTEXT.md`, `docs/adr/0009-public-derived-only-private-raw-store.md` (amendment note).

**Create**
- `docs/adr/0017-training-store-public.md`.

---

### Task 1: `main()` targets the latest available HRDPS run

**Why:** HRDPS runs at 00/06/12/18 UTC and Datamart publishes each ~3–4 h after init. `main()`'s current `issue_time = now` floored to the hour would request a non-existent run (404) most hours. Target the most-recently-published 6-hourly cycle instead.

**Files:**
- Modify: `src/microclimate/pipelines/inference.py`
- Test: `tests/pipelines/test_inference.py`

- [ ] **Step 1: Write the failing test** (append to `tests/pipelines/test_inference.py`)

```python
def test_latest_hrdps_issue_time_floors_to_published_6h_cycle() -> None:
    from microclimate.pipelines.inference import _latest_hrdps_issue_time

    # 14:00Z minus ~4h publish lag = 10:00Z → floor to the 06Z run
    assert _latest_hrdps_issue_time(datetime(2026, 6, 1, 14, 0, tzinfo=UTC)) == datetime(
        2026, 6, 1, 6, 0, tzinfo=UTC
    )
    # 16:30Z − 4h = 12:30Z → 12Z run
    assert _latest_hrdps_issue_time(datetime(2026, 6, 1, 16, 30, tzinfo=UTC)) == datetime(
        2026, 6, 1, 12, 0, tzinfo=UTC
    )
    # 03:00Z − 4h = previous day 23:00Z → 18Z run (day rollover)
    assert _latest_hrdps_issue_time(datetime(2026, 6, 1, 3, 0, tzinfo=UTC)) == datetime(
        2026, 5, 31, 18, 0, tzinfo=UTC
    )
    # naive input is treated as UTC
    assert _latest_hrdps_issue_time(datetime(2026, 6, 1, 14, 0)) == datetime(
        2026, 6, 1, 6, 0, tzinfo=UTC
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/pipelines/test_inference.py::test_latest_hrdps_issue_time_floors_to_published_6h_cycle -v`
Expected: FAIL — `_latest_hrdps_issue_time` absent.

- [ ] **Step 3: Implement** in `src/microclimate/pipelines/inference.py`.

Add near the top (after `_ATTRIBUTION`), and a `timedelta` import:
```python
from datetime import UTC, datetime, timedelta  # add timedelta to the existing import

_HRDPS_PUBLISH_LAG = timedelta(hours=4)  # Datamart publishes each run ~3-4 h after init


def _latest_hrdps_issue_time(now: datetime) -> datetime:
    """Most recent HRDPS run init time (00/06/12/18 UTC) likely published by ``now``.

    HRDPS runs four times daily; Datamart publishes each run ~3-4 h after its init time.
    Subtracting the publish lag then flooring to the 6-hourly cycle yields a run that should
    be available. If a chosen run is still unpublished, ``build_snapshot`` raises
    ``ForecastUnavailable`` and the next hourly Action run retries; the training store dedupes
    a re-logged ``issue_time`` (ADR-0015), so hourly re-runs are idempotent.
    """
    t = (now.astimezone(UTC) if now.tzinfo is not None else now.replace(tzinfo=UTC)) - _HRDPS_PUBLISH_LAG
    run_hour = (t.hour // 6) * 6
    return t.replace(hour=run_hour, minute=0, second=0, microsecond=0)
```

In `main()`, replace:
```python
    issue_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
```
with:
```python
    issue_time = _latest_hrdps_issue_time(datetime.now(UTC))
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/pipelines/test_inference.py -v` → PASS (all inference tests).

- [ ] **Step 5: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/pipelines/inference.py tests/pipelines/test_inference.py
git commit -m "fix(pipelines): inference main() targets the latest published HRDPS 6-hourly run"
```

---

### Task 2: Rewrite the inference workflow (data collection)

**Files:**
- Modify: `.github/workflows/inference.yml`

> **This task has no unit tests** — it is workflow YAML, validated by a `workflow_dispatch` run (see Final Integration). The implementer just authors the file and confirms it is valid YAML.

- [ ] **Step 1: Replace `.github/workflows/inference.yml`** with:

```yaml
name: inference
on:
  schedule:
    - cron: "0 * * * *"
  workflow_dispatch:

# Serialize runs so two hourly runs never push the training-data branch at once.
concurrency:
  group: inference
  cancel-in-progress: false

# Allow the built-in GITHUB_TOKEN to push the training-data branch.
permissions:
  contents: write

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install eccodes (cfgrib GRIB2 decode for HRDPS Datamart)
        run: sudo apt-get update && sudo apt-get install -y --no-install-recommends libeccodes0

      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - name: Verify cfgrib can load eccodes
        run: uv run python -c "import cfgrib; print('cfgrib', cfgrib.__version__)"

      - name: Check out (or bootstrap) the public training-data store
        env:
          STORE_URL: https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git
        run: |
          set -euo pipefail
          if git clone --depth 1 --branch training-data "$STORE_URL" store 2>/dev/null; then
            echo "Cloned existing training-data branch."
          else
            echo "training-data branch absent; bootstrapping an empty orphan branch."
            rm -rf store && mkdir store
            git -C store init -q
            git -C store remote add origin "$STORE_URL"
            git -C store checkout -q --orphan training-data
          fi

      - name: Run inference for each deployment (append snapshots to the store)
        run: |
          set -euo pipefail
          for f in config/deployments/*.yml; do
            id="$(basename "$f" .yml)"
            echo "::group::inference $id"
            TRAINING_STORE_ROOT=store uv run python -m microclimate.pipelines.inference --deployment "$id"
            echo "::endgroup::"
          done

      - name: Commit & push new snapshots to training-data
        run: |
          set -euo pipefail
          cd store
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add -A
          if git diff --cached --quiet; then
            echo "No new store files; nothing to commit."
          else
            git commit -q -m "data: snapshots $(date -u +%FT%TZ)"
            git push origin HEAD:training-data
          fi
```

- [ ] **Step 2: Confirm the YAML is valid** (no Python to run):

Run: `uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/inference.yml')); print('valid yaml')"`
Expected: `valid yaml`. (Optionally `actionlint .github/workflows/inference.yml` if available.)

- [ ] **Step 3: Full gate (unchanged tests) + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add .github/workflows/inference.yml
git commit -m "ci(inference): hourly Action collects snapshots to the public training-data branch"
```

---

### Task 3: ADR-0017 + ADR-0009 amendment + README + CONTEXT + runbook

**Files:**
- Create: `docs/adr/0017-training-store-public.md`
- Modify: `docs/adr/0009-public-derived-only-private-raw-store.md`, `README.md`, `CONTEXT.md`

- [ ] **Step 1: Write ADR-0017** (0017 is the next free number; confirm with `ls docs/adr/`):

```markdown
# 17. Training store is public (`training-data` branch); private store retired

- **Status:** Accepted
- **Date:** 2026-05-31
- **Amends:** ADR-0009 (which made the raw store private).
- **Relates to:** ADR-0007 (training-data branch), ADR-0010 (ACIS dropped), ADR-0015 (store).

## Context

ADR-0009 made the raw training store **private** (separate repo, `DATA_REPO_TOKEN`) **solely
because of ACIS's unsettled redistribution rights**. ADR-0010 then dropped ACIS and retargeted
to ECCC (`envcanada` observations + HRDPS), both **redistributable with attribution**. The
store therefore now holds only redistributable data — triggering ADR-0009's own escape hatch
("the raw store could be made public … if the ambiguity resolves").

## Decision

The training store is **public**, committed to a `training-data` branch in the main repo by
the hourly inference Action via the built-in `GITHUB_TOKEN`. The separate private repo and
`DATA_REPO_TOKEN` are **retired** — no external setup is required to collect data.
**Attribution remains mandatory** (the published forecast carries it; the store is raw ECCC
data, redistributable with attribution).

## Consequences

- Zero-setup hourly data collection (no private repo, no PAT/secret).
- The `training-data` branch grows by one commit/hour; periodic compaction/prune is future
  work (consistent with ADR-0015's small-file note).
- Training (subsystem 3) reads the `training-data` branch.
- ADR-0007's "fourth artifact home" reverts to the public `training-data` branch it originally
  described; ADR-0009's private-store consequence is superseded for the ECCC-only deployment.
```

- [ ] **Step 2: Amend ADR-0009** — add a header note directly under its `- **Date:**` line (mirroring how ADR-0009 itself amended ADR-0003/0007):

```markdown
> **Amendment (ADR-0017):** the raw store is now **public** (a `training-data` branch in the
> main repo), not a private repo. ADR-0009's private-store decision was driven by ACIS, which
> ADR-0010 dropped; the store now holds only ECCC-redistributable data. The `DATA_REPO_TOKEN`
> / separate-private-repo machinery below is retired.
```

- [ ] **Step 3: Update `README.md` Project status** — note the hourly inference Action now collects snapshots to the public `training-data` branch (read the current section; add this; keep the gh-pages forecast publish + registry/champion-loading in the not-yet list).

- [ ] **Step 4: Update `CONTEXT.md` "Training store" term** — change "path-based (a private-repo checkout in production, ADR-0009/0015)" to "path-based; persisted to a **public `training-data` branch** committed by the hourly inference Action (ADR-0017)".

- [ ] **Step 5: Add a runbook** — append a short "Running the inference Action" subsection to `README.md` (or a comment block atop `inference.yml`): how to trigger (`workflow_dispatch`), where data lands (`training-data` branch, `snapshots/deployment_id=…/ym=…` (the `store/` prefix is the runner path; the branch root is the store checkout)), the first-run orphan-branch bootstrap, and that the repo needs Actions **read/write** permission (the workflow declares `permissions: contents: write`, which suffices on a public repo with default settings).

- [ ] **Step 6: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add docs/adr/0017-training-store-public.md docs/adr/0009-public-derived-only-private-raw-store.md README.md CONTEXT.md
git commit -m "docs: ADR-0017 public training store (amends ADR-0009); README/CONTEXT + runbook"
```

---

## Final Integration

- [ ] Push and open a PR (main is PR-only):

```bash
git push -u origin spec/inference-deploy-action
gh pr create --fill --base main
```

- [ ] After automated review + CI pass and the PR merges, **validate the Action (operator step / gate):**
  1. In the repo's Actions tab, run the **inference** workflow via `workflow_dispatch` (or wait for the top-of-hour cron).
  2. Confirm the job is green (eccodes/cfgrib import OK; live Datamart + ECCC fetch OK).
  3. Confirm a commit appears on the **`training-data`** branch containing
     `snapshots/deployment_id=lethbridge/ym=YYYYMM/*.parquet`.
  4. Re-run once and confirm it appends (and the store dedupe keeps the latest re-logged `issue_time`).
  - If the run 404s on HRDPS, the chosen run wasn't published yet — the next hour retries; if it persists, tune `_HRDPS_PUBLISH_LAG`.

---

## Self-review notes

- **Spec coverage:** workflow persists snapshots to the public `training-data` branch (Task 2) ✓; ADR-0017 + ADR-0009 amendment (Task 3) ✓; README/CONTEXT/runbook (Task 3) ✓; single-job loop avoids the matrix push-race ✓; eccodes install + cfgrib check ✓; `GITHUB_TOKEN`, `permissions`, `concurrency` ✓; gh-pages publish / registry correctly out of scope.
- **Spec refinement (necessary, surfaced at plan time):** the spec said "no new Python" with `issue_time = now` floored to the hour, but HRDPS is 6-hourly — that would 404 most runs. Task 1 adds the small, tested `_latest_hrdps_issue_time` helper so the Action targets a published run. This is the one Python change; it's testable and self-contained.
- **Validation reality:** Task 2 is YAML — no pytest. The real proof is the post-merge `workflow_dispatch` (Final Integration), which needs live Datamart/ECCC from the runner + eccodes. The `network`-marked Datamart test already proved the connector locally; this confirms it in CI.
- **Open items resolved:** orphan-branch bootstrap = clone-or-`checkout --orphan` shell guard; the `discover` job is dropped in favor of an inline `for` loop (single job, race-free); eccodes package = `libeccodes0` (the cfgrib-import step is the check).
- **Type/name consistency:** `_latest_hrdps_issue_time(now)` matches its `main()` call site and test; `TRAINING_STORE_ROOT=store` matches `main()`'s env read.
