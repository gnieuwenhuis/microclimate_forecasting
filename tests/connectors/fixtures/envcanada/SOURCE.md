# Fixture provenance

These CSV files are tiny cropped samples from the ECCC bulk hourly CSV endpoint
(`https://climate.weather.gc.ca/climate_data/bulk_data_e.html`, `timeframe=1`),
station 49268, Climate ID 3033875, station name "LETHBRIDGE" (YQL), Alberta, Canada.

`hourly_window.csv` — seven rows spanning 2026-04-27 03:00–09:00 LST.  Row 06:00
has several fields blanked to exercise per-row absent-value masking.

`live_partial.csv` — two real rows (2026-05-29 21:00–22:00 LST) plus three trailing
rows that are intentionally truncated (no measurement fields), simulating the
not-yet-reported future hours that appear in a live current-month response.  These
truncated rows must be dropped by the connector.

`dewpoint_derive.csv` — one row from 2023-06-01 12:00 LST with the "Dew Point Temp
(°C)" cell intentionally blanked so that the connector must derive dewpoint from
T=15.0 °C and RH=75% via the Magnus-Tetens formula (expected ≈ 10.6 °C).

Data Source: Environment and Climate Change Canada
