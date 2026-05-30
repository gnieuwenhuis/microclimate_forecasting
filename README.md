# Microclimate Forecasting

Free, zero-maintenance hourly **temperature** and **probability-of-precipitation** forecasts
for a local station, by downscaling Environment Canada's HRDPS. Designed around Lethbridge,
Alberta; deployable for any microclimate by config.

- **Domain glossary:** [CONTEXT.md](CONTEXT.md)
- **Decisions:** [docs/adr/](docs/adr/)
- **Data licenses & attribution:** [DATA_LICENSES.md](DATA_LICENSES.md)
- **Scaffolding spec:** [docs/superpowers/specs/2026-05-30-scaffolding-spec.md](docs/superpowers/specs/2026-05-30-scaffolding-spec.md)

The architecture is enforced mechanically: typed boundaries (Pydantic/Pandera), connector
ABCs, a single feature-snapshot builder, source-eligibility validation, and an
`import-linter` layer contract — all gated in CI.
