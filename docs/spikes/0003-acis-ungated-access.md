# Spike 0003 — ACIS ungated access + Picture Butte reachability

- **Issue:** [#3](../../../../issues/3) (Spike: ACIS ungated access + Picture Butte reachability)
- **Parent:** [#1](../../../../issues/1) (ACIS connector slice)
- **Date:** 2026-05-30
- **Status:** Complete. Spike answered; **parent #1 (ACIS connector) is BLOCKED pending an architecture decision** (see *Options / decision required*).
- **Evidence:** all curl probes, HTTP statuses, byte counts, and raw rows behind this report were captured headless (no cookies, no token, no referer; UA `microclimate-forecasting-spike (research)`) on 2026-05-30 and are recorded in the scratch file `SPIKE_NOTES.md` (untracked) at the repo root of this worktree.

---

## TL;DR — bottom line up front

1. **There IS an ungated ACIS endpoint serving multi-year history for our stations — but it is DAILY, not hourly.** This **refutes the premise inherited from issue #1** ("ungated IMCIN climate-file downloads serve multi-year *hourly* history"). The ungated ACIS "AIMM climate file" is a **daily aggregate** (one row per calendar day; no hour/time column).
2. **There is NO ungated source of *live hourly* observations for these stations — anywhere.** Ungated hourly *history* does exist off-ACIS (ECCC/MSC "AGDM" twins via the same bulk-CSV mechanism the `envcanada` connector already uses), but those hourly feeds are **dead** — they stopped reporting in 2023–2024. ACIS's only live-hourly route is the session-/captcha-gated Data Viewer API the project forbids.
3. **Ungated *live* data for these stations exists only at DAILY cadence** (ACIS AIMM, ~1-day lag).
4. **Picture Butte #710547 is reachable** via the same ungated ACIS AIMM daily file (contradicting the prior config assumption). **Iron Springs #9883 is stale** (last data 2023-05-18; the 2026 file is header-only).

Net: the connector cannot be built to its original "ungated hourly" premise. The findings are sufficient to **unblock the investigation**, but they force a deliberate architecture choice (daily vs. hourly target; ACIS vs. ECCC; live vs. history) that this spike documents and leaves to issue #1 / a new design issue.

Terminology in this doc follows `CONTEXT.md`: *dual-feed source*, *historical coverage* (`deep`), *fetch_live*, *target* / *neighbor*, *connector*.

---

## The ungated endpoint

Discovered from the JavaScript on the AIMM climate-files page (`https://acis.alberta.ca/acis/imcin/aimm-climate-files.jsp`, redirects http→https, HTTP 200). The per-station per-year download tiles call two JS functions that build two **byte-identical** ungated URLs:

- **By IMCIN short-name (2-digit year suffix):**
  `https://acis.alberta.ca/acis/api/v1/imcin/aimm/data/{ShortName}{YY}.txt`
- **By numeric station id (4-digit year):**
  `https://acis.alberta.ca/acis/api/v1/imcin/aimm/station/{id}/climate-data/{YYYY}`

Both return `Content-Type: text/csv`, and were verified to return identical bytes for the same station/year. **Ungated** confirmed: HTTP 200 headless with no auth header, no cookie, no token, no referer. The server *sets* a `JSESSIONID` cookie but does **not** require one (no `WWW-Authenticate`; re-requests with no cookie still 200).

```
$ curl -sS --no-keepalive -A "microclimate-forecasting-spike (research)" \
    -o /dev/null -w "aimm status=%{http_code}\n" \
    "https://acis.alberta.ca/acis/api/v1/imcin/aimm/data/Lethbridge26.txt"
aimm status=200
```

---

## Acceptance criteria — answered

### AC1 — Ungated endpoint(s) + URL pattern + format serving multi-year history for #9835 / #9883 / #9747

**Verdict: ANSWERED — multi-year history CONFIRMED ungated, but cadence is DAILY, not hourly.** The original AC asked for *hourly* history; no ungated endpoint serves hourly for these stations (see *Deeper dig*). The AIMM daily file is the ungated multi-year history that exists.

- **Endpoint / pattern / format:** the two AIMM URLs above; CSV, header row + **one row per calendar day**, no time-of-day field.
- **Header (identical across all stations/years):** `YEAR, MONTH, DAY, TMAXC, TMINC, WINDKM, PRECMM, RHMAX, RHMIN, SRKJD`
- **Multi-year depth (CONFIRMED `deep`):**
  - #9835 `Lethbridge`: years **2005–2026**. `Lethbridge17.txt` → 200, 23,481 bytes, 365 daily rows (full 2017); `Lethbridge05.txt` → 200, 18,354 bytes (2005-04-01 → 2005-12-31).
  - #9747 `Btap` (Blood Tribe): years **2005–2026** (2026 file → 200, 9,596 bytes).
  - #9883 `IronSprings`: years **2005–2023 only** — no 2024/2025/2026 tile; the by-id 2026 request returns a header-only file (see AC3 / Iron Springs caveat).

Evidence — current-year Lethbridge (#9835):

```
$ curl -sSL ... "https://acis.alberta.ca/acis/api/v1/imcin/aimm/data/Lethbridge26.txt"
status=200 type=text/csv bytes=9590   (149 lines incl. header => 148 days, Jan 1..May 29 2026)
YEAR, MONTH, DAY, TMAXC, TMINC, WINDKM, PRECMM, RHMAX, RHMIN, SRKJD
2026, 1, 1, -3.50, -9.04, 172.60, 0.00, 94.30, 85.10, 3451.55
...
2026, 5, 29, 28.09, 15.45, 429.59, 0.10, 80.80, 26.49, 24287.20
```

### AC2 — Refresh cadence of the current-year file (the basis for `fetch_live` staleness)

**Verdict: ANSWERED — daily file, ~1-day lag.** On 2026-05-30, the latest row in every live 2026 AIMM file was **2026-05-29** (yesterday):

```
Lethbridge26.txt    last row: 2026, 5, 29, ...
Btap26.txt          last row: 2026, 5, 29, ...
PictureButte26.txt  last row: 2026, 5, 29, ...
```

**Implication for the dual-feed contract:** a `fetch_live` reading this endpoint gets a new **daily** row at roughly a 1-day lag (yesterday's completed day, available today). It **cannot** supply intra-day / hourly current conditions. The most recent obs `fetch_live` can return is the previous calendar day, at daily resolution.

### AC3 — id→IMCIN-short-name map + IMCIN column→variable mapping

**Verdict: ANSWERED.** Short-name map confirmed from the page JS and cross-checked by downloading both URL forms (identical bytes).

| Role | id | IMCIN short-name | Verdict |
|---|---|---|---|
| target | 9835 | `Lethbridge` | CONFIRMED |
| neighbor | 9747 | `Btap` | CONFIRMED |
| neighbor | 9883 | `IronSprings` | CONFIRMED name — **STALE** (last data 2023-05-18; 2026 file header-only, 67 bytes) |
| neighbor | 710547 | `PictureButte` (LITE) | CONFIRMED reachable — see AC4 |

By-id cross-check (proves id↔short-name; byte-identical to by-name):

```
.../aimm/station/9835/climate-data/2026   -> 200, 9590 bytes (== Lethbridge26.txt)
.../aimm/station/9747/climate-data/2026    -> 200, 9596 bytes (== Btap26.txt)
.../aimm/station/9883/climate-data/2026    -> 200, 67 bytes   (header only, no data rows)
.../aimm/station/710547/climate-data/2026  -> 200, 9597 bytes (== PictureButte26.txt)
```
The by-id form's `Content-Disposition` reveals real station names (e.g. `LethbridgeDemoFarmIMCIN...`), corroborating the id↔name binding.

**IMCIN column → canonical-variable mapping (DAILY):**

| File column | Meaning | Canonical var | Present? |
|---|---|---|---|
| YEAR, MONTH, DAY | calendar date (no hour) | timestamp | date only — **no hour** |
| TMAXC | daily max air temp °C | temp | PARTIAL (daily max only) |
| TMINC | daily min air temp °C | temp | PARTIAL (daily min only) |
| WINDKM | daily wind run, km/day | wind speed | PARTIAL (run, not m/s; no direction) |
| PRECMM | daily total precip, mm | precip | PRESENT (daily total) |
| RHMAX | daily max relative humidity % | relative humidity | PARTIAL (daily max only) |
| RHMIN | daily min relative humidity % | relative humidity | PARTIAL (daily min only) |
| SRKJD | daily solar radiation, kJ/m²/day | solar radiation | PRESENT (daily total) |

**Coverage of the 8 canonical variables:** temp PARTIAL (daily max/min only) · dewpoint ABSENT · relative humidity PARTIAL (daily max/min only) · surface pressure ABSENT · precip PRESENT (daily total) · solar radiation PRESENT (daily total) · cloud cover ABSENT · wind PARTIAL (daily run km only, no direction, not a speed).

Even setting aside the hourly problem, the daily AIMM file cleanly provides only precip and solar; three more variables only as daily max/min; and is missing dewpoint, surface pressure, and cloud cover entirely.

### AC4 — Picture Butte #710547 resolved

**Verdict: ANSWERED — CONFIRMED reachable.** Picture Butte (a "LITE", non-IMCIN station) **does** serve an ungated AIMM file, contradicting the prior config assumption that LITE stations have no IMCIN/AIMM file.

```
$ ... PictureButte26.txt                    -> 200, type=text/csv, 9597 bytes
$ .../aimm/station/710547/climate-data/2026  -> 200, 9597 bytes (byte-identical)
YEAR, MONTH, DAY, TMAXC, TMINC, WINDKM, PRECMM, RHMAX, RHMIN, SRKJD
2026, 1, 1, -4.56, -10.03, 146.60, 0.00, 93.00, 82.90, 2565.80
...last row... 2026, 5, 29, 26.99, 15.68, 402.98, 0.00, 76.81, 31.56, 24607.72
```

- Same daily-only schema, same ~1-day lag (fresh through 2026-05-29).
- **Available years: 2022–2026** (shorter history than the IMCIN stations).
- **Caveat:** 2022 carries `-9999.00` sentinels for WINDKM/SRKJD early in the file (gaps).
- **No ECCC AGDM hourly twin exists** for Picture Butte (so no ungated hourly anywhere for it — see *Deeper dig*).
- Config updated: `config/deployments/lethbridge.yml` now records Picture Butte as reachable-ungated-daily and drops the resolved "precip gauge unconfirmed / gated ACIS API" reachability uncertainty. Coordinates/elevation remain unconfirmed and are still flagged.

### AC5 — Findings sufficient to unblock the ACIS connector slice

**Verdict: ANSWERED.** This report + the appendix give a builder exact URL templates, the daily schema, the id↔short-name↔ECCC-StationID table, and the variable mapping. However, the findings **invalidate the slice's premise** (ungated hourly): they unblock by exposing a decision that must be made before code is written. **Issue #1's ACIS connector work is therefore BLOCKED pending the architecture decision** in *Options / decision required*.

---

## Deeper dig: hourly avenues

### ECCC/MSC "AGDM" twins — ungated HOURLY history, but DEAD

The ACIS ag stations are mirrored into ECCC's national climate archive as "...AGDM" stations (same lat/lon, same data) and served via the **same ungated bulk-CSV endpoint the `envcanada` connector already uses**:

`https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={ID}&Year={Y}&Month={M}&Day=1&timeframe=1`

CONFIRMED ungated (no cookie/captcha/referer; `application/force-download`, 200) and **truly hourly** (one row per hour, `Time (LST)` column):

```
$ curl -sSL ... "...bulk_data_e.html?format=csv&stationID=42726&Year=2024&Month=3&Day=1&timeframe=1&submit=Download+Data"
status=200 type=application/force-download size=135983
header: ... Date/Time (LST), Year, Month, Day, Time (LST), Flag, Temp (°C), ..., Dew Point Temp (°C), ...,
        Rel Hum (%), ..., Precip. Amount (mm), ..., Wind Dir (10s deg), ..., Wind Spd (km/h), ..., Stn Press (kPa), ...
row: "-112.75","49.68","LETHBRIDGE DEMO FARM AGDM","3033897","2024-03-01 00:00",...
```

| ACIS station | ECCC name | Climate ID | ECCC StationID | HLY range | Last POPULATED hourly obs |
|---|---|---|---|---|---|
| #9835 Lethbridge Demo Farm | LETHBRIDGE DEMO FARM AGDM | 3033897 | **42726** | 2004–2024 | **2024-04-02 11:00 LST** |
| #9747 Blood Tribe / `Btap` | BLOOD TRIBE AGDM | 3030720 | **42703** | 2004–2024 | 2024-05-10 12:00 LST |
| #9883 Iron Springs | IRON SPRINGS AGDM | 3033498 | **42728** | 2004–2023 | 2023-05-15 09:00 LST |
| #710547 Picture Butte | (no AGDM twin) | — | — | — | — |

- **Variable coverage (hourly): 6 of 8 usable** — temp, dewpoint, RH, precip, wind speed, wind direction. Station-pressure column exists but is **always empty** (flagged M). Solar and cloud-cover columns do not exist.
- **STALENESS (decisive): these hourly feeds are DEAD.** Requests for 2025/2026 months return well-formed CSVs with **every value empty** (e.g. stationID 42726, 2026-05 → 200, 744 rows, 0 populated Temp rows). ECCC inventory "Last Year" corroborates (2024 / 2024 / 2023).
- Nearest still-recent ECCC hourly neighbor probed (VAUXHALL CDA CS, climID 3036681, stnID 10889, ~50 km NE) ran hourly through ~Jan 2025 and is also now stale (2026 empty). No nearby AGDM/ag station is currently live with ungated hourly.

So ECCC gives a rich ~2004→2023/2024 hourly **history** (drop-in to the existing `envcanada` bulk-CSV mechanism by StationID), but **cannot supply current/live hourly** for these stations.

### The gated ACIS Data Viewer API — the only ACIS hourly route, forbidden

The viewer (`/acis/weather-data-viewer.jsp` → `station-data-viewer.js`) exposes the real (non-AIMM) data endpoints:

```
/acis/api/v1/legacy/weather-data/timeseries
/acis/api/v1/legacy/weather-data/static-table
/acis/api/v1/legacy/weather-data/highcharts-data
```

All three, probed headless with full hourly params, return **HTTP 403** (`application/json`, 94 bytes): *"Please access our data using the web interface with a modern, standards-compliant web-browser."* The viewer JS includes a `captchaManager` with `preflightCheck()` / `validateSubmit()` — an explicit CAPTCHA + browser-fingerprint gate. The older `/acis/weather-data/graph` endpoint renders a PNG and returns 500 headless. Every ungated hourly variation (extra path segment, `?interval=hourly`) either 404s or silently serves the **daily** AIMM file. This is the session-gated path the project forbids for unattended jobs.

### Dead-end portals / contact path

- **Alberta Open Data** (`open.alberta.ca/.../acis`) just links back to the ACIS viewer — no downloadable hourly station files, no separate API.
- **open.canada.ca** "Precipitation at Selected Alberta Weather Stations" = two XLSX of *annual* precip vs 1961–2014 normals. Not station obs.
- **ACIS township / interpolated / spatial** historical viewer = *interpolated daily* grid points (township centres, back to 1961) — same daily ceiling, not station obs.
- **ACIS bulk-extract contact path** — the only stated route to volume data is a vetted data-extract request (phone Trevor Wallace, (780) 980-7587). Could yield a one-off historical hourly extract or a sanctioned feed, but it is **manual, approval-gated, and NOT automatable** for an unattended cron job.

---

## Options / decision required

The original slice premise (ungated **hourly** ACIS) is invalid. Below are the credible pivots with evidence-based tradeoffs. **Per the issue owner's decision, this spike does not choose one** — the architecture pivot is an explicit decision for issue #1 / a new design issue. Until then, **issue #1's ACIS connector work is BLOCKED.**

| # | Option | Live cadence | Training history | Vars | Key tradeoff / risk |
|---|---|---|---|---|---|
| A | **Keep ACIS daily-live + ECCC AGDM hourly history** | ACIS AIMM daily, ~1-day lag | ECCC AGDM hourly 2004→2023/2024 | daily: precip/solar + daily max/min temp/RH; hourly hist: 6/8 | Train/serve **cadence skew** (hourly history vs daily live); AGDM logger frozen at 2024 so no fresh hourly labels; ACIS daily live cannot feed an *hourly* label/feature. Hardest to reconcile with the hourly forecast goal. |
| B | **Retarget to a station still live on ungated ECCC hourly** | ECCC hourly (live) | ECCC hourly (deep) | 6–7/8 | Cleanest dual-feed (single mechanism, hourly both feeds). But the chosen Demo Farm microclimate is lost; nearest probed candidates (e.g. Vauxhall) are also stale — needs a fresh search for a currently-live hourly station near Henderson Lake. |
| C | **Daily product** (re-scope the forecast to daily) | ACIS AIMM daily | ACIS AIMM daily (deep, 2005→) | precip/solar + daily max/min temp/RH | Contradicts `CONTEXT.md` ("hourly temperature and PoP"); large product re-scope; but fully ungated, live, and deep on one source. |
| D | **ECCC-only** (drop ACIS; target an ECCC station) | ECCC live | ECCC deep | per-station | Simplest source story; removes ACIS attribution/connector entirely; depends on finding a live-hourly ECCC station for the target microclimate (same risk as B). |

Cross-cutting facts that constrain all options:
- The *dual-feed* contract requires **both** a deep historical feed and a live feed for the *same* physical measurement at the *same* cadence; no single ungated source currently satisfies this **hourly** for the target station.
- Picture Butte (#710547) has ungated daily but **no hourly anywhere**; under any hourly option it is daily-only or must be dropped as an hourly neighbor.
- Iron Springs (#9883) is dead on both ACIS (2023-05-18) and ECCC AGDM (2023) — **history-only**, not usable for live, under every option.

---

## Appendix — reusable specifics for whoever builds the connector

### URL templates

```
# ACIS AIMM daily climate file (ungated, text/csv) — pick either form, byte-identical:
https://acis.alberta.ca/acis/api/v1/imcin/aimm/data/{ShortName}{YY}.txt          # YY = last 2 digits of year
https://acis.alberta.ca/acis/api/v1/imcin/aimm/station/{id}/climate-data/{YYYY}

# ECCC/MSC bulk hourly CSV (ungated; same mechanism as the envcanada connector) — HISTORY ONLY (dead since 2023/2024):
https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={ID}&Year={Y}&Month={M}&Day=1&timeframe=1
```

### AIMM daily file schema (10 columns, one row per calendar day)

```
YEAR, MONTH, DAY, TMAXC, TMINC, WINDKM, PRECMM, RHMAX, RHMIN, SRKJD
```
`TMAXC`/`TMINC` = daily max/min air temp °C · `WINDKM` = daily wind run km/day · `PRECMM` = daily total precip mm · `RHMAX`/`RHMIN` = daily max/min RH % · `SRKJD` = daily solar radiation kJ/m²/day. Missing-value sentinel observed: `-9999.00` (e.g. Picture Butte 2022 WINDKM/SRKJD). No hour/time column.

### Station identity table

| ACIS id | ACIS IMCIN short-name | Type | ACIS years (ungated daily) | ECCC StationID (hourly) | ECCC Climate ID | ECCC hourly status |
|---|---|---|---|---|---|---|
| 9835 | `Lethbridge` (Lethbridge Demo Farm) | IMCIN | 2005–2026 (live) | 42726 | 3033897 | dead — last 2024-04-02 |
| 9747 | `Btap` (Blood Tribe Ag Project) | IMCIN | 2005–2026 (live) | 42703 | 3030720 | dead — last 2024-05-10 |
| 9883 | `IronSprings` | IMCIN | 2005–2023 (STALE) | 42728 | 3033498 | dead — last 2023-05-15 |
| 710547 | `PictureButte` | LITE | 2022–2026 (live) | — | — | no AGDM twin |

### Variable-mapping summary (canonical 8)

| Canonical var | ACIS AIMM daily | ECCC AGDM hourly (history) |
|---|---|---|
| temp | daily max/min only (TMAXC/TMINC) | hourly (Temp °C) |
| dewpoint | absent | hourly (Dew Point Temp °C) |
| relative humidity | daily max/min only (RHMAX/RHMIN) | hourly (Rel Hum %) |
| surface pressure | absent | column present but always empty (M) |
| precip | daily total (PRECMM) | hourly (Precip. Amount mm) |
| solar radiation | daily total (SRKJD) | absent |
| cloud cover | absent | absent |
| wind | daily run km only, no direction (WINDKM) | hourly speed + direction (Wind Spd km/h, Wind Dir 10s deg) |

### Attribution

Any ACIS-derived artifact carries: `Data provided by Agriculture and Irrigation, Alberta Climate Information Service (ACIS)`. ECCC-derived: `Data Source: Environment and Climate Change Canada` (per `CONTEXT.md` / ADR-0009).
