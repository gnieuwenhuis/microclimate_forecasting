## Source of truth for decisions & domain language

**`docs/adr/` is the authoritative record of architectural decisions.** Explore that folder
and read the ADRs that touch the area before changing anything structural — the filenames are
descriptive, so browse them as needed rather than relying on a list here. ADRs are not
historical artifacts — they are living: **when you make, change, or supersede an architectural
decision, add or update an ADR in the same PR** so code never drifts from the documented
decision. Don't silently contradict an ADR; if your work calls one into question, surface it
and reopen the ADR (use `/grill-with-docs` to produce or revise one).

**`CONTEXT.md` is the single-context domain glossary.** Use its terms *exactly* in code,
issues, tests, ADRs, and plans — don't drift to synonyms. **When you introduce or rename a
domain concept, update `CONTEXT.md` in the same PR.** If a concept you need isn't there,
that's a signal: either you're inventing language the project doesn't use (reconsider) or
there's a real gap (resolve it via `/grill-with-docs` and add it).

## Architecture

Single Python package `src/microclimate`, layered low→high and enforced by `import-linter`
(`.importlinter`): `contracts` → `config` → `connectors` → `features` → `evaluation` /
`models` (independent siblings) → `publication` → `pipelines`. Higher layers may import lower;
never the reverse. Typed boundaries are Pydantic (config, forecast JSON) and Pandera
(dataframes); `features.build_snapshot` is the **single** feature path shared by training and
inference (ADR-0011), guarded by an architecture test.

**Build state:** the project is built bottom-up, so some upper-layer subsystems are still
stubs that raise `NotImplementedError` — `grep -rl NotImplementedError src/microclimate` to
see exactly what's outstanding before assuming a capability exists. (`acis` is retained but
unused per ADR-0010.) **Keep the README's "Project status" section current — update it as
subsystems are finished or new work is discovered.**

## Commands

```bash
uv sync                      # install deps + dev group
uv run ruff check .          # lint
uv run ruff format --check . # format check (drop --check to apply)
uv run lint-imports          # enforce the layer contract
uv run pyright               # strict type check
uv run pytest                # tests; network-marked tests are deselected by default
uv run pytest -m network     # run the tests that hit real external services
```

CI (`.github/workflows/ci.yml`) runs all of the above on every push/PR.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (gnieuwenhuis/microclimate_forecasting) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: CONTEXT.md + docs/adr/ at the repo root. See `docs/agents/domain.md`.
