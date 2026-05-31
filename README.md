# Microclimate Forecasting

Free, zero-maintenance hourly **temperature** and **probability-of-precipitation** forecasts
for a local station, by downscaling Environment Canada's HRDPS. Designed around Lethbridge,
Alberta; deployable for any microclimate by config.

- **Domain glossary (source of truth for terms):** [CONTEXT.md](CONTEXT.md)
- **Architectural decisions (source of truth):** [docs/adr/](docs/adr/)
- **Data licenses & attribution:** [DATA_LICENSES.md](DATA_LICENSES.md)
- **Scaffolding spec:** [docs/superpowers/specs/2026-05-30-scaffolding-spec.md](docs/superpowers/specs/2026-05-30-scaffolding-spec.md)

`docs/adr/` records *why* the architecture is the way it is, and `CONTEXT.md` defines the
vocabulary the whole project uses. Both are kept current as the project evolves — a change
that alters a decision or introduces a concept updates them in the same PR.

The architecture is enforced mechanically: typed boundaries (Pydantic/Pandera), connector
ABCs, a single feature-snapshot builder, source-eligibility validation, and an
`import-linter` layer contract — all gated in CI.

## Project status

Pre-1.0, built bottom-up. The **data backbone is implemented**: typed contracts, validated
deployment config, the connector framework with live + historical sources (Environment
Canada observations, HRDPS from Datamart and CaSPAr), and `features.build_snapshot` — the
single as-of feature path shared by training and inference (ADR-0011).

**Not yet implemented** (currently stubs): the temp/PoP models, evaluation + publish gate,
forecast-JSON / registry publication, and the inference/training pipeline CLIs.

## Develop

```bash
uv sync                      # install deps + dev group
uv run ruff check . && uv run ruff format --check .
uv run lint-imports          # enforce the layered architecture
uv run pyright               # strict type check
uv run pytest                # network-marked tests are deselected by default
```

These mirror the CI gate (`.github/workflows/ci.yml`). The package lives in
`src/microclimate`, layered low→high (`contracts` → `config` → `connectors` → `features` →
`evaluation`/`models` → `publication` → `pipelines`); the layer order is enforced by
`import-linter`. Deployment configs live in `config/deployments/`; the thin-client dashboard
in `dashboard/`.
