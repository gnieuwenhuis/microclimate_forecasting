# Spike 0004 — Open-Meteo HRDPS access (2026-06-02)

Empirical probe (no API key) confirming the ADR-0019 source decision.

- `api.open-meteo.com/v1/forecast` and `historical-forecast-api.open-meteo.com/v1/forecast`
  both return GEM HRDPS Continental (model=`gem_hrdps_continental`) with no key — free
  non-commercial access confirmed.
- Deep history confirmed: 2024-06 returns full data on the historical host.
- **Key finding (ADR-0019 §1b):** the deep archive is a STITCHED SHORT-LEAD series.
  `temperature_2m_previous_day1` returns data (~24h lead) but `temperature_2m_previous_day2`
  and beyond are NULL for HRDPS — so deep full-1..48-lead-per-run history is NOT available
  for free. v1 trains on the short-lead seed and accepts the lead-time skew.
- Response shape: `{"hourly": {"time": [...], "<var>": [...], ...}}`, times are UTC-naive
  ISO `YYYY-MM-DDTHH:MM` when `timezone=GMT`.
- Fixtures captured: `tests/connectors/fixtures/openmeteo_forecast.json`,
  `openmeteo_historical.json`.
