# Training Store Coalescing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. **Task 4 (workflow YAML) is infra — validated by running the Action, not pytest.**

**Goal:** Coalesce the training store to one `data.parquet` per deployment-month (read-modify-write, write-time dedupe), make inference logging idempotent, and have the Action force-push the `training-data` branch as a single state commit — eliminating tiny-file pile-up, hourly re-log duplication, and git-history bloat.

**Architecture:** `TrainingStore` gains a shared read-modify-write merge helper; `append_snapshot`/`write_labels` read the partition's `data.parquet`, merge+dedupe, atomic-rewrite. `read_*` glob the one-file-per-month layout. New `has_snapshot` powers an idempotent skip in `run_inference`. The Action orphan-commits + force-pushes the store state. Revises ADR-0015 (ADR-0018).

**Tech Stack:** pandas + pyarrow, Pydantic, pytest, pyright strict, ruff, uv, GitHub Actions/git.

---

## Conventions

- Tasks 1–3 are TDD (Python). Task 4 is YAML (validated by a `workflow_dispatch` run). Task 5 is docs.
- Full gate before each commit: `uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`.
- Commit on branch `spec/store-coalesce` (main is PR-only); push only at Final Integration.
- The existing `tests/training_store/test_store.py` and `tests/pipelines/test_inference.py` cases use only the public API and stay green (no file-count/glob assertions); this plan **adds** tests, it doesn't rewrite them.
- **Migration note:** the store has no real data yet (the Action hasn't collected). The new reads glob `ym=*/data.parquet`; any legacy `<uuid>.parquet` files (none in practice) are simply not read — no migration step needed.

## File structure

**Modify**
- `src/microclimate/training_store/store.py` — read-modify-write coalescing + `has_snapshot`.
- `tests/training_store/test_store.py` — add coalescing/`has_snapshot` tests.
- `src/microclimate/pipelines/inference.py` — `run_inference` idempotent skip (`-> … | None`); `main()` handles `None`.
- `tests/pipelines/test_inference.py` — add the skip test.
- `.github/workflows/inference.yml` — force-push state.
- `docs/adr/0015-*.md`, `docs/adr/0017-*.md` (amendment notes), `CONTEXT.md`.

**Create**
- `docs/adr/0018-training-store-coalescing.md`.

---

### Task 1: Coalesce snapshots (read-modify-write) + `has_snapshot`

**Files:** Modify `src/microclimate/training_store/store.py`; Test `tests/training_store/test_store.py`

- [ ] **Step 1: Add failing tests** (append to `tests/training_store/test_store.py`):

```python
def test_one_data_file_per_month_with_both_rows(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    store.append_snapshot(_snap(issue_time=_T0))
    store.append_snapshot(_snap(issue_time=_T0 + timedelta(days=2)))  # same month (June)
    ym_dir = tmp_path / "snapshots" / "deployment_id=lethbridge" / "ym=202606"
    assert [p.name for p in ym_dir.glob("*.parquet")] == ["data.parquet"]  # exactly one file
    assert len(store.read_snapshots("lethbridge")) == 2


def test_has_snapshot(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    assert store.has_snapshot("lethbridge", _T0) is False
    store.append_snapshot(_snap(issue_time=_T0))
    assert store.has_snapshot("lethbridge", _T0) is True
    assert store.has_snapshot("lethbridge", _T0 + timedelta(hours=1)) is False
    assert store.has_snapshot("other", _T0) is False
```

(`_snap`/`_T0` already exist in this file. `_T0` is 2026-06-01 → `ym=202606`.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/training_store/test_store.py -k "one_data_file or has_snapshot" -v` → FAIL (uuid filenames; `has_snapshot` absent).

- [ ] **Step 3: Implement.** In `store.py`:

(a) Update the module docstring's "Layout" block:
```python
Layout (one coalesced Parquet per partition; grown by read-modify-write, atomic via
temp+os.replace; deduped at write time):
    {root}/snapshots/deployment_id={id}/ym={YYYYMM}/data.parquet
    {root}/labels/deployment_id={id}/ym={YYYYMM}/data.parquet
The training-data branch is force-pushed as a single state commit (ADR-0017/0018).
```

(b) Replace `_atomic_write_parquet` to write a **fixed destination file**, and add a shared
read-modify-write merge helper (place after `_range_bounds`):
```python
def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Atomically write df to the Parquet file ``dest`` (temp in the same dir + os.replace)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{uuid.uuid4().hex}.tmp"
    df.to_parquet(tmp, index=False)  # type: ignore[reportUnknownMemberType]
    os.replace(tmp, dest)


def _merge_and_dedupe(
    dest: Path, new: pd.DataFrame, *, subset: list[str], dt_cols: list[str]
) -> pd.DataFrame:
    """Read the existing partition file (if any), append ``new``, normalise ``dt_cols`` to UTC,
    dedupe on ``subset`` keeping the latest ``written_at``, return sorted by ``subset``."""
    frames = [pd.read_parquet(dest), new] if dest.exists() else [new]  # type: ignore[reportUnknownMemberType]
    df = pd.concat(frames, ignore_index=True)  # type: ignore[reportUnknownMemberType]
    for col in dt_cols:
        df[col] = pd.to_datetime(df[col], utc=True)
    return (
        df.sort_values("written_at")
        .drop_duplicates(subset=subset, keep="last")
        .sort_values(subset)
        .reset_index(drop=True)
    )
```

(c) Add a partition-path helper and rewrite `append_snapshot` + `read_snapshots` + add
`has_snapshot` (replace lines 86–145):
```python
    def _partition(self, deployment_id: str, issue_time: datetime, kind: str) -> Path:
        return (
            self._root
            / kind
            / f"deployment_id={deployment_id}"
            / f"ym={_ym(issue_time)}"
            / "data.parquet"
        )

    def append_snapshot(
        self, snapshot: FeatureSnapshot, *, written_at: datetime | None = None
    ) -> None:
        """Append one raw snapshot row into its deployment-month file (read-modify-write)."""
        stamp = _to_utc(written_at) if written_at is not None else _to_utc(datetime.now(UTC))
        new = pd.DataFrame(
            [
                {
                    "deployment_id": snapshot.deployment_id,
                    "issue_time": pd.Timestamp(snapshot.issue_time),
                    "schema_version": snapshot.schema_version,
                    "snapshot_json": snapshot.model_dump_json(),
                    "written_at": stamp,
                }
            ],
            columns=_SNAPSHOT_COLUMNS,
        )
        dest = self._partition(snapshot.deployment_id, snapshot.issue_time, "snapshots")
        merged = _merge_and_dedupe(
            dest, new, subset=["issue_time"], dt_cols=["issue_time", "written_at"]
        )
        _atomic_write_parquet(merged, dest)

    def has_snapshot(self, deployment_id: str, issue_time: datetime) -> bool:
        """True if a snapshot for ``issue_time`` is already stored (cheap; reads one month file)."""
        ts = _to_utc(issue_time)
        dest = self._partition(deployment_id, ts, "snapshots")
        if not dest.exists():
            return False
        existing = pd.read_parquet(dest, columns=["issue_time"])  # type: ignore[reportUnknownMemberType]
        return bool((pd.to_datetime(existing["issue_time"], utc=True) == ts).any())

    def read_snapshots(
        self,
        deployment_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[FeatureSnapshot]:
        """Return snapshots for the deployment in [start, end], latest-per-issue_time, sorted."""
        start_ts, end_ts = _range_bounds(start, end)
        base = self._root / "snapshots" / f"deployment_id={deployment_id}"
        files = sorted(base.glob("ym=*/data.parquet")) if base.exists() else []
        if not files:
            return []
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)  # type: ignore[reportUnknownMemberType]
        df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
        df["written_at"] = pd.to_datetime(df["written_at"], utc=True)
        if start_ts is not None:
            df = df[df["issue_time"] >= start_ts]
        if end_ts is not None:
            df = df[df["issue_time"] <= end_ts]
        if df.empty:
            return []
        # Files are deduped at write time; this defensive dedupe is cheap insurance.
        df = df.sort_values("written_at").drop_duplicates(subset="issue_time", keep="last")
        bad = sorted(
            str(v)
            for v in df.loc[
                df["schema_version"] != SNAPSHOT_SCHEMA_VERSION, "schema_version"
            ].unique()
        )
        if bad:
            raise ValueError(
                f"training store has snapshot schema_version(s) {bad} != current "
                f"{SNAPSHOT_SCHEMA_VERSION!r}; a schema migration is required."
            )
        df = df.sort_values("issue_time")
        return [FeatureSnapshot.model_validate_json(j) for j in df["snapshot_json"]]
```

- [ ] **Step 4: Run** — `uv run pytest tests/training_store/test_store.py -v` → all PASS (existing + 2 new). `uv run pyright` → 0 (add narrow ignores only where pyright flags new pandas lines).

- [ ] **Step 5: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/training_store/store.py tests/training_store/test_store.py
git commit -m "feat(training_store): coalesce snapshots to one data.parquet/month (read-modify-write) + has_snapshot"
```

---

### Task 2: Coalesce labels (read-modify-write)

**Files:** Modify `src/microclimate/training_store/store.py`; Test `tests/training_store/test_store.py`

- [ ] **Step 1: Add a failing test**:

```python
def test_one_label_data_file_per_month(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    store.write_labels("lethbridge", _labels(_T0), written_at=_T0)
    store.write_labels("lethbridge", _labels(_T0 + timedelta(days=1)), written_at=_T0)  # same month
    ym_dir = tmp_path / "labels" / "deployment_id=lethbridge" / "ym=202606"
    assert [p.name for p in ym_dir.glob("*.parquet")] == ["data.parquet"]
    out = store.read_labels("lethbridge")
    assert len(out) == 6  # two issue_times × 3 leads
```

(`_labels`/`_T0` already exist.)

- [ ] **Step 2: Run to verify failure** — FAIL (uuid filenames → two files; or KeyError if asserting one).

- [ ] **Step 3: Implement** — rewrite `write_labels`'s write loop to read-modify-write each `ym`
  partition's `data.parquet`, and change `read_labels`'s glob to `ym=*/data.parquet`:

In `write_labels`, replace the final `for ym, part …` loop:
```python
        for ym, part in df.groupby(df["issue_time"].dt.strftime("%Y%m")):  # type: ignore[reportUnknownMemberType]
            dest = self._root / "labels" / f"deployment_id={deployment_id}" / f"ym={ym}" / "data.parquet"
            merged = _merge_and_dedupe(
                dest,
                part,
                subset=["issue_time", "lead_hour"],
                dt_cols=["issue_time", "valid_time", "written_at"],
            )
            _atomic_write_parquet(merged, dest)
```

In `read_labels`, change the glob line:
```python
        files = sorted(base.glob("ym=*/data.parquet")) if base.exists() else []
```
(everything else in `read_labels` is unchanged.)

- [ ] **Step 4: Run** — `uv run pytest tests/training_store/test_store.py -v` → all PASS. `uv run pyright` → 0.

- [ ] **Step 5: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/training_store/store.py tests/training_store/test_store.py
git commit -m "feat(training_store): coalesce labels to one data.parquet/month (read-modify-write)"
```

---

### Task 3: `run_inference` idempotent skip

**Files:** Modify `src/microclimate/pipelines/inference.py`; Test `tests/pipelines/test_inference.py`

- [ ] **Step 1: Add a failing test** (append to `tests/pipelines/test_inference.py`):

```python
def test_run_inference_skips_when_already_collected(tmp_path: Path) -> None:
    from microclimate.connectors.base import SourceUnavailable

    config = make_config(horizon_hours=3, lag_hours=2)
    leads = [1, 2, 3]
    ts = [_T0 - timedelta(hours=h) for h in (2, 1, 0)]
    observations = {"fake": FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})}
    store = TrainingStore(tmp_path / "store")

    # First run logs the snapshot for _T0.
    doc1 = run_inference(
        config, nwp=FakeNWP(make_forecast_frame(_T0, leads)), observations=observations,
        store=store, forecast_path=tmp_path / "f1.json", issue_time=_T0,
    )
    assert doc1 is not None

    # Second run for the SAME issue_time must skip — no fetch (nwp raises if called), no publish.
    boom = FakeNWP(exc=SourceUnavailable("run_inference must not fetch when already collected"))
    f2 = tmp_path / "f2.json"
    doc2 = run_inference(
        config, nwp=boom, observations=observations, store=store, forecast_path=f2, issue_time=_T0
    )
    assert doc2 is None
    assert not f2.exists()
```

- [ ] **Step 2: Run to verify failure** — FAIL (no skip → `boom` raises `SourceUnavailable`, or `doc2` isn't `None`).

- [ ] **Step 3: Implement** in `src/microclimate/pipelines/inference.py`:

Change the `run_inference` return type and add the early skip at the top of the body:
```python
def run_inference(
    config: DeploymentConfig,
    *,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    store: TrainingStore,
    forecast_path: Path,
    issue_time: datetime,
) -> ForecastDocument | None:
    """Build a snapshot → baseline forecast → write JSON → log the snapshot. Returns the doc,
    or None if this issue_time is already in the store (idempotent skip — no fetch/publish)."""
    if store.has_snapshot(config.deployment_id, issue_time):
        return None
    snapshot = build_snapshot(config, issue_time, nwp, observations)
    ... (rest unchanged) ...
    return doc
```

In `main()`, tolerate `None`:
```python
    doc = run_inference(
        config,
        nwp=nwp,
        observations=observations,
        store=store,
        forecast_path=Path(config.output.forecast_json),
        issue_time=issue_time,
    )
    if doc is None:
        print(f"Already collected snapshot for issue_time={issue_time.isoformat()}; skipped.")
```

- [ ] **Step 4: Run** — `uv run pytest tests/pipelines/test_inference.py -v` → PASS (incl. the existing tests, which pass a fresh store so `has_snapshot` is False and they proceed). `uv run pyright` → 0.

- [ ] **Step 5: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/pipelines/inference.py tests/pipelines/test_inference.py
git commit -m "feat(pipelines): run_inference skips already-collected issue_times (idempotent logging)"
```

---

### Task 4: Action force-pushes the store state

**Files:** Modify `.github/workflows/inference.yml`

> Infra — no unit test; validated by a `workflow_dispatch` run (Final Integration).

- [ ] **Step 1: Replace the final "Commit & push" step** with a force-push-state step:

```yaml
      - name: Force-push the store state to training-data
        run: |
          set -euo pipefail
          cd store
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          if [ -z "$(git status --porcelain)" ]; then
            echo "No store changes; nothing to push."
          else
            # The training-data branch is derived state (provenance is in the rows), so replace
            # its history with a single commit of the current store — keeps git size bounded
            # despite read-modify-write rewrites (ADR-0018).
            git checkout -q --orphan _state
            git add -A
            git commit -q -m "data: store snapshot $(date -u +%FT%TZ)"
            git push -q --force origin _state:training-data
          fi
```

Also update the workflow's header comment if it implies accumulating append commits (it
describes where data lands — keep that; note the branch is force-pushed state).

- [ ] **Step 2: Validate YAML** — `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/inference.yml')); print('valid yaml')"` → `valid yaml`.

- [ ] **Step 3: Full gate (unchanged tests) + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add .github/workflows/inference.yml
git commit -m "ci(inference): force-push the training-data branch as a single state commit (ADR-0018)"
```

---

### Task 5: ADR-0018 + ADR-0015/0017 amendments + CONTEXT

**Files:** Create `docs/adr/0018-training-store-coalescing.md`; Modify `docs/adr/0015-*.md`, `docs/adr/0017-*.md`, `CONTEXT.md`

- [ ] **Step 1: Write ADR-0018** (0018 is the next free number; confirm `ls docs/adr/`):

```markdown
# 18. Training store coalesces to one file per deployment-month; branch is force-pushed state

- **Status:** Accepted
- **Date:** 2026-05-31
- **Revises:** ADR-0015 (append-only-uuid + read-time dedupe).
- **Relates to:** ADR-0007/0008 (logger), ADR-0017 (public training-data branch).

## Context

ADR-0015 wrote a new `<uuid>.parquet` per append and deduped on read. Driven hourly by the
inference Action — which also re-logs the same 6-hourly HRDPS `issue_time` ~6× — that piles
up thousands of tiny, often-duplicate files (slow reads). Coalescing into one growing file via
read-modify-write would fix reads but, on a git branch, re-storing a growing binary Parquet
each commit bloats history.

## Decision

1. **Coalesce to one `data.parquet` per `deployment_id`/`ym` (month)**, grown by
   **read-modify-write** with **write-time dedupe** (snapshots on `issue_time`; labels on
   `(issue_time, lead_hour)`), latest `written_at` wins. Reads glob `ym=*/data.parquet`.
   (Supersedes ADR-0015's append-only-uuid layout and its "compaction is future work" note —
   coalescing is the compaction.)
2. **Idempotent logging:** `run_inference` skips (no fetch/build/publish/append) when the
   `issue_time` is already stored (`TrainingStore.has_snapshot`), so each distinct HRDPS run
   is collected once.
3. **The Action manages `training-data` as state:** it force-pushes a single commit of the
   current store each run, so git history stays bounded despite read-modify-write. The branch
   is derived, forward-regenerable data; provenance is in each row (`issue_time`/`written_at`),
   not git history.

## Consequences

- ~12 small files/deployment/year (fast reads); git history ≈ one commit.
- **Single-writer only:** read-modify-write + whole-branch force-push assume one serialized
  writer (the Action is one job). Concurrent writers are out of scope (would need rework).
- Monthly window; weekly (`ym`→`yw`) is a trivial future knob.
```

- [ ] **Step 2: Amend ADR-0015** — header note under its `- **Date:**`:
```markdown
> **Revised by ADR-0018:** the store now coalesces to one `data.parquet` per deployment-month
> via read-modify-write with write-time dedupe (not append-only `<uuid>.parquet` + read-time
> dedupe), and the "compaction is future work" note is resolved (coalescing is the compaction).
```

- [ ] **Step 3: Amend ADR-0017** — add a consequence/note: the inference Action manages the
  `training-data` branch as **state** — it force-pushes a single commit of the current store
  each run (not accumulating append commits), keeping git size bounded (ADR-0018).

- [ ] **Step 4: Update CONTEXT.md "Training store"** — change "Partitioned Parquet, append-only"
  to: one **`data.parquet` per deployment-month** (coalesced **read-modify-write**, write-time
  dedupe); the `training-data` branch is **force-pushed state** (ADR-0018).

- [ ] **Step 5: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add docs/adr/0018-training-store-coalescing.md docs/adr/0015-*.md docs/adr/0017-*.md CONTEXT.md
git commit -m "docs: ADR-0018 store coalescing + force-push state; amend ADR-0015/0017 + CONTEXT"
```

---

## Final Integration

- [ ] Push + PR: `git push -u origin spec/store-coalesce && gh pr create --fill --base main`.
- [ ] After merge, re-validate the Action via `workflow_dispatch`: confirm `training-data` has a
  **single commit** whose tree holds `snapshots/deployment_id=lethbridge/ym=YYYYMM/data.parquet`
  (one file), and a second run **appends a row to that same file** (force-pushes a new single
  commit) rather than adding a new file — and a same-`issue_time` re-run pushes nothing (skip).

---

## Self-review notes

- **Spec coverage:** coalesced snapshots (Task 1) ✓; coalesced labels (Task 2) ✓; `has_snapshot` + idempotent `run_inference` returning `… | None` (Tasks 1, 3) ✓; force-push state (Task 4) ✓; ADR-0018 + ADR-0015/0017 amendments + CONTEXT (Task 5) ✓. Single-writer + monthly + weekly-knob noted in ADR-0018.
- **Existing tests stay green:** they use the public API only (round-trip, dedupe-keeps-latest, filter, schema-guard, labels) — all hold under coalescing; this plan only adds the file-count/`has_snapshot`/skip tests.
- **Type/name consistency:** `_merge_and_dedupe(dest, new, *, subset, dt_cols)` used by both `append_snapshot` and `write_labels`; `_atomic_write_parquet(df, dest)` now takes a file path; `has_snapshot(deployment_id, issue_time)` matches its `run_inference` call site; `run_inference -> ForecastDocument | None` matches `main()` (which now handles `None`).
- **Open items resolved:** `read_*` keeps the defensive dedupe; the Action no-op check is `git status --porcelain` empty → skip; legacy uuid files need no migration (store is empty).
