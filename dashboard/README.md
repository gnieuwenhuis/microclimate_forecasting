# Dashboard (thin client)

Static files served from the `gh-pages` branch. Reads `forecasts/<deployment_id>.json` from
the same origin and renders it. Targets forecast `schema_version` **1** and shows the JSON's
`attribution` strings in the footer (ADR-0009). No build step, no secrets, no raw data.
