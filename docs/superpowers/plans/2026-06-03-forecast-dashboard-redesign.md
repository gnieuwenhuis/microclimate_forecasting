# Forecast Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bland gh-pages README with a redesigned forecast dashboard (A-header + C-body hybrid) and make it actually deploy, fixing the never-published and schema-rejection defects.

**Architecture:** Plain static thin client — `dashboard/{index.html,styles.css,app.js}` — hand-rolled inline SVG, auto light/dark via CSS custom properties, local time, `?deployment=`-aware. The hourly `inference.yml` copies the dashboard into the gh-pages worktree (+`.nojekyll`) alongside the forecast JSON. Two cheap Python tests guard the schema-major match and the deploy step; visual checks are an out-of-band Playwright pass (no JS/browser runner in CI).

**Tech Stack:** Static HTML/CSS/vanilla JS (no build step, no deps — ADR-0009); Python `pytest` for the regression guards; GitHub Actions + Pages + the existing gh-pages publish flow.

**Design source:** `docs/superpowers/specs/2026-06-03-forecast-dashboard-redesign-design.md`. The throwaway prototype `dashboard/_prototype.html` is the visual reference (variant A header + variant C body); it is deleted in the final task.

---

### Task 1: Project docs + `styles.css`

**Files:**
- Commit (already on disk): `docs/superpowers/specs/2026-06-03-forecast-dashboard-redesign-design.md`, `docs/superpowers/plans/2026-06-03-forecast-dashboard-redesign.md`
- Create: `dashboard/styles.css`

This is a static stylesheet — no unit test (verified visually in Task 5). It defines light vars, a dark override, and every class `app.js` (Task 3) renders.

- [ ] **Step 1: Create `dashboard/styles.css`**

```css
:root {
  --bg:#f6f7f9; --panel:#fff; --ink:#14181d; --muted:#5b6672; --line:#e3e7ec;
  --temp:#ef5b3b; --pop:#2f7dd1; --grid:#eef1f4; --zero:#9aa6b2; --accent:#2563eb;
  --ok:#1a9e58; --ok-bg:#e3f5ea; --stale:#c98a16; --stale-bg:#fdf3df;
  --degraded:#d6452f; --degraded-bg:#fbe6e2;
  --shadow:0 1px 3px rgba(20,24,29,.08), 0 8px 24px rgba(20,24,29,.05); --radius:14px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#0e1217; --panel:#161b22; --ink:#e7ecf1; --muted:#8b97a4; --line:#232a33;
    --temp:#ff7a5c; --pop:#5aa9ef; --grid:#1d242c; --zero:#4a5663; --accent:#5b9bff;
    --ok:#45d486; --ok-bg:#11321f; --stale:#f0c156; --stale-bg:#322a12;
    --degraded:#ff7a63; --degraded-bg:#3a1c16;
    --shadow:0 1px 3px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
  }
}
* { box-sizing:border-box; }
html, body { margin:0; }
body {
  background:var(--bg); color:var(--ink);
  font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing:antialiased;
}
#app { max-width:920px; margin:0 auto; padding:24px 20px 48px; }
h1 { margin:0; font-weight:650; letter-spacing:-.01em; font-size:30px; }
.muted { color:var(--muted); }
.sub { font-size:13px; margin-top:2px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); margin-bottom:16px; }
.head { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; margin-bottom:14px; }
.now { display:flex; align-items:baseline; gap:18px; margin:4px 0 20px; }
.now .big { font-size:64px; font-weight:700; letter-spacing:-.03em; line-height:.9; }
.now .big.temp { color:var(--temp); }
.now-sub { font-size:14px; }
.now-sub b { font-size:22px; font-weight:700; color:var(--pop); }
.chart { padding:16px; }
.legend { font-size:12px; margin-bottom:6px; }
.legend .temp { color:var(--temp); }
.legend .pop { color:var(--pop); }
.meta { padding:16px; display:flex; flex-wrap:wrap; gap:24px; align-items:flex-start; font-size:13px; }
.meta dl { display:grid; grid-template-columns:auto auto; gap:6px 14px; margin:0; }
.meta dt { color:var(--muted); }
.meta dd { margin:0; font-variant-numeric:tabular-nums; }
.models { display:flex; flex-direction:column; gap:6px; }
.mtask { display:inline-block; min-width:32px; color:var(--muted); }
.pill { display:inline-flex; align-items:center; gap:6px; font-size:12.5px; font-weight:650; padding:4px 10px; border-radius:999px; cursor:help; text-transform:capitalize; }
.pill .dot { width:7px; height:7px; border-radius:50%; background:currentColor; }
.pill.ok { color:var(--ok); background:var(--ok-bg); }
.pill.stale { color:var(--stale); background:var(--stale-bg); }
.pill.degraded { color:var(--degraded); background:var(--degraded-bg); }
.badge { display:inline-block; font-size:11.5px; padding:3px 8px; border-radius:7px; border:1px solid var(--line); color:var(--muted); }
.badge.champ { color:var(--accent); border-color:color-mix(in srgb, var(--accent) 40%, var(--line)); }
.table-head { padding:12px 16px; font-weight:600; border-bottom:1px solid var(--line); }
.table-wrap { max-height:520px; overflow-y:auto; }
table.hourly { width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }
table.hourly th { text-align:left; color:var(--muted); font-weight:600; font-size:11.5px; text-transform:uppercase; letter-spacing:.03em; padding:8px 12px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--panel); }
table.hourly td { padding:7px 12px; border-bottom:1px solid var(--grid); }
table.hourly td.t { color:var(--temp); font-weight:600; }
.attrib { font-size:12px; color:var(--muted); line-height:1.6; margin-top:6px; }
.msg { padding:60px 20px; text-align:center; color:var(--muted); }
@media (max-width:520px) { h1 { font-size:24px; } .now .big { font-size:52px; } }
```

- [ ] **Step 2: Commit the docs + stylesheet**

```bash
git add docs/superpowers/specs/2026-06-03-forecast-dashboard-redesign-design.md \
        docs/superpowers/plans/2026-06-03-forecast-dashboard-redesign.md \
        dashboard/styles.css
git commit -m "docs+style: dashboard redesign spec/plan + stylesheet"
```

---

### Task 2: `index.html` skeleton

**Files:**
- Modify (full rewrite): `dashboard/index.html`

- [ ] **Step 1: Replace `dashboard/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Microclimate Forecast</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main id="app"><div class="msg">Loading forecast…</div></main>
    <script src="app.js"></script>
  </body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/index.html
git commit -m "feat: dashboard index skeleton (styles + #app mount)"
```

---

### Task 3: `app.js` rewrite + schema-major guard (TDD)

**Files:**
- Create: `tests/dashboard/__init__.py`
- Create: `tests/dashboard/test_dashboard.py`
- Modify (full rewrite): `dashboard/app.js`

The schema-major guard is the regression test for the "rejects 1.0.0" defect. Write it first (red against the current `app.js`, which uses `SCHEMA_VERSION = "1"`), then the rewrite introduces `SUPPORTED_SCHEMA_MAJOR` and makes it green.

- [ ] **Step 1: Create the test package init**

```bash
mkdir -p tests/dashboard
printf '' > tests/dashboard/__init__.py
```

- [ ] **Step 2: Write the failing schema-major guard**

Create `tests/dashboard/test_dashboard.py`:

```python
"""Regression guards for the static dashboard thin client (no JS runner in CI)."""

import re
from pathlib import Path

from microclimate.contracts.forecast import FORECAST_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_JS = REPO_ROOT / "dashboard" / "app.js"


def test_app_js_supported_major_matches_contract():
    text = APP_JS.read_text()
    match = re.search(r'SUPPORTED_SCHEMA_MAJOR\s*=\s*"(\d+)"', text)
    assert match, "dashboard/app.js must define SUPPORTED_SCHEMA_MAJOR"
    assert match.group(1) == FORECAST_SCHEMA_VERSION.split(".")[0]
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/dashboard/test_dashboard.py -v`
Expected: FAIL — current `app.js` has no `SUPPORTED_SCHEMA_MAJOR` (the assertion "must define SUPPORTED_SCHEMA_MAJOR" fires).

- [ ] **Step 4: Rewrite `dashboard/app.js`**

Replace the entire file with:

```javascript
// Thin client (ADR-0009): read the published forecast JSON and render it.
// No raw data, no secrets, no build step. Hand-rolled SVG, local time, auto light/dark.
const SUPPORTED_SCHEMA_MAJOR = "1";
const params = new URLSearchParams(location.search);
const DEPLOYMENT = (params.get("deployment") || "lethbridge").toLowerCase();

// ---- formatters (local time) ----
const fmtHour = (iso) => new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
const fmtFull = (iso) => new Date(iso).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" });
const fmtT = (c) => `${Math.round(c)}°`;
const fmtP = (p) => `${Math.round(p * 100)}%`;
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

function ago(iso) {
  const mins = Math.round((Date.now() - new Date(iso)) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const h = Math.round(mins / 60);
  if (h < 24) return `${h} h ago`;
  return `${Math.round(h / 24)} d ago`;
}

const STATUS_MEAN = {
  ok: "Fresh — full horizon served by the latest run.",
  stale: "Published, but shorter than the target horizon (far leads weren't yet available).",
  degraded: "An expected champion model was unusable; a baseline served instead.",
};
function statusPill(s) {
  const cls = STATUS_MEAN[s] ? s : "stale";
  return `<span class="pill ${cls}" title="${STATUS_MEAN[s] || ""}"><span class="dot"></span>${s}</span>`;
}

const scaler = (min, max, lo, hi) => (v) => lo + (hi - lo) * (max === min ? 0.5 : (v - min) / (max - min));

// ---- dual-axis SVG: temperature line + PoP bars (variant C) ----
function dualSVG(series, { w = 1000, h = 320, pad = { l: 38, r: 40, t: 18, b: 28 } } = {}) {
  const temps = series.map((s) => s.temp_c);
  let lo = Math.min(...temps), hi = Math.max(...temps);
  const span = Math.max(hi - lo, 1);
  lo -= span * 0.12;
  hi += span * 0.12;
  const x = scaler(0, series.length - 1, pad.l, w - pad.r);
  const yT = scaler(lo, hi, h - pad.b, pad.t);
  const yP = scaler(0, 1, h - pad.b, pad.t);
  const bw = (w - pad.l - pad.r) / series.length;
  const bars = series
    .map((s, i) => {
      const bx = x(i) - bw * 0.35;
      const bh = h - pad.b - yP(s.pop);
      return `<rect x="${bx.toFixed(1)}" y="${yP(s.pop).toFixed(1)}" width="${(bw * 0.7).toFixed(1)}" height="${Math.max(bh, 0).toFixed(1)}" fill="var(--pop)" opacity=".30"/>`;
    })
    .join("");
  const pts = series.map((s, i) => [x(i), yT(s.temp_c)]);
  const dLine = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const tv = lo + (hi - lo) * (i / 4);
    grid += `<line x1="${pad.l}" y1="${yT(tv)}" x2="${w - pad.r}" y2="${yT(tv)}" stroke="var(--grid)"/>`;
    grid += `<text x="${pad.l - 6}" y="${yT(tv) + 3}" text-anchor="end" font-size="10" fill="var(--temp)">${Math.round(tv)}°</text>`;
  }
  [0, 0.5, 1].forEach((p) => {
    grid += `<text x="${w - pad.r + 6}" y="${yP(p) + 3}" font-size="10" fill="var(--pop)">${(p * 100) | 0}%</text>`;
  });
  if (lo < 0 && hi > 0) {
    grid += `<line x1="${pad.l}" y1="${yT(0)}" x2="${w - pad.r}" y2="${yT(0)}" stroke="var(--zero)" stroke-dasharray="3 3"/>`;
  }
  let xl = "";
  for (let i = 0; i < series.length; i += 6) {
    xl += `<text x="${x(i)}" y="${h - 8}" text-anchor="middle" font-size="10" fill="var(--muted)">${fmtHour(series[i].valid_time)}</text>`;
  }
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" preserveAspectRatio="none" style="height:${h}px">${grid}${bars}<path d="${dLine}" fill="none" stroke="var(--temp)" stroke-width="2.5" stroke-linejoin="round"/>${xl}</svg>`;
}

// ---- full view (A-header + C body) ----
function view(doc) {
  const s = doc.series;
  const now = s.find((x) => x.lead_hour === 1) || s[0];
  const rows = s
    .map(
      (x) => `<tr>
      <td class="muted">+${x.lead_hour}</td>
      <td>${fmtFull(x.valid_time)}</td>
      <td class="t">${x.temp_c.toFixed(1)}°</td>
      <td>${fmtP(x.pop)}</td>
    </tr>`,
    )
    .join("");
  const models = Object.entries(doc.model_versions)
    .map(([t, v]) => `<div><span class="mtask">${t}</span> <span class="badge ${v !== "baseline" ? "champ" : ""}">${v}</span></div>`)
    .join("");
  return `
  <header class="head">
    <div>
      <h1>${cap(doc.deployment_id)}</h1>
      <div class="muted sub">Issued ${fmtFull(doc.issue_time)} · updated ${ago(doc.last_updated)}</div>
    </div>
    ${statusPill(doc.status)}
  </header>
  <section class="now">
    <div class="big temp">${fmtT(now.temp_c)}</div>
    <div class="now-sub muted">in ${now.lead_hour} h<br><b>${fmtP(now.pop)}</b> chance of precip</div>
  </section>
  <section class="panel chart">
    <div class="legend muted"><span class="temp">— temperature</span> &nbsp; <span class="pop">▮ precip probability</span></div>
    ${dualSVG(s)}
  </section>
  <section class="panel meta">
    <dl>
      <dt>Status</dt><dd>${statusPill(doc.status)}</dd>
      <dt>Issued</dt><dd>${fmtFull(doc.issue_time)}</dd>
      <dt>Updated</dt><dd>${ago(doc.last_updated)}</dd>
      <dt>Horizon</dt><dd>${s.length} / 48 h</dd>
      <dt>Schema</dt><dd>${doc.schema_version}</dd>
    </dl>
    <div class="models">${models}</div>
  </section>
  <section class="panel">
    <div class="table-head">Hourly values (${s.length})</div>
    <div class="table-wrap"><table class="hourly">
      <thead><tr><th>Lead</th><th>Valid (local)</th><th>Temp</th><th>Precip</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </section>
  <footer class="attrib">${doc.attribution.map((a) => `<div>${a}</div>`).join("")}</footer>`;
}

function message(html) {
  document.getElementById("app").innerHTML = `<div class="msg">${html}</div>`;
}

async function load() {
  try {
    const res = await fetch(`forecasts/${DEPLOYMENT}.json`, { cache: "no-store" });
    if (!res.ok) {
      message("Forecast unavailable.");
      return;
    }
    const doc = await res.json();
    if ((doc.schema_version || "").split(".")[0] !== SUPPORTED_SCHEMA_MAJOR) {
      message(`This dashboard doesn't support forecast schema ${doc.schema_version}.`);
      return;
    }
    if (!Array.isArray(doc.series) || doc.series.length === 0) {
      message("No forecast steps available.");
      return;
    }
    document.getElementById("app").innerHTML = view(doc);
  } catch (e) {
    message("Forecast unavailable.");
    console.error(e);
  }
}

load();
```

- [ ] **Step 5: Run the guard to verify it passes**

Run: `uv run pytest tests/dashboard/test_dashboard.py -v`
Expected: PASS — `SUPPORTED_SCHEMA_MAJOR = "1"` matches `FORECAST_SCHEMA_VERSION.split(".")[0]`.

- [ ] **Step 6: Commit**

```bash
git add tests/dashboard/__init__.py tests/dashboard/test_dashboard.py dashboard/app.js
git commit -m "feat: redesigned dashboard client + schema-major regression guard"
```

---

### Task 4: Deploy the dashboard + deploy guard (TDD)

**Files:**
- Modify: `tests/dashboard/test_dashboard.py` (add a test)
- Modify: `.github/workflows/inference.yml` (publish step)

- [ ] **Step 1: Add the failing deploy guard**

Append to `tests/dashboard/test_dashboard.py`:

```python
INFERENCE_YML = REPO_ROOT / ".github" / "workflows" / "inference.yml"


def test_inference_workflow_publishes_dashboard():
    text = INFERENCE_YML.read_text()
    for asset in ("dashboard/index.html", "dashboard/app.js", "dashboard/styles.css"):
        assert asset in text, f"inference.yml must copy {asset} into the gh-pages worktree"
    assert ".nojekyll" in text, "inference.yml must create .nojekyll on gh-pages"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/dashboard/test_dashboard.py::test_inference_workflow_publishes_dashboard -v`
Expected: FAIL — `inference.yml` doesn't copy the dashboard yet.

- [ ] **Step 3: Add the publish step to `inference.yml`**

In the "Publish forecast JSON to gh-pages" step, the `run:` block currently starts with `cd gp`. Change the start of that block from:

```yaml
        run: |
          cd gp
          git init -q 2>/dev/null || true
```

to:

```yaml
        run: |
          # Publish the static dashboard alongside the forecast JSON so gh-pages serves the
          # dashboard (not the rendered README). .nojekyll disables Jekyll on Pages.
          cp dashboard/index.html dashboard/app.js dashboard/styles.css gp/
          touch gp/.nojekyll
          cd gp
          git init -q 2>/dev/null || true
```

(Leave the rest of the step — `git add -A`, the commit, and the non-force push logic — unchanged; `git add -A` picks up the copied files.)

- [ ] **Step 4: Run the guard to verify it passes**

Run: `uv run pytest tests/dashboard/test_dashboard.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add tests/dashboard/test_dashboard.py .github/workflows/inference.yml
git commit -m "feat: publish dashboard to gh-pages from inference.yml (+deploy guard)"
```

---

### Task 5: Visual verification (out-of-band Playwright pass)

No CI browser runner exists, so this is a manual confirmation of the rendered output against all three `status` values. Not committed.

**Files:**
- Temporary (do not commit): `dashboard/forecasts/lethbridge.json` and two scratch fixtures

- [ ] **Step 1: Generate ok / stale / degraded fixtures from the live forecast**

```bash
python3 - <<'PY'
import json, copy, pathlib
live = json.load(open("dashboard/_prototype_forecast.json"))
out = pathlib.Path("dashboard/forecasts"); out.mkdir(parents=True, exist_ok=True)
# stale (the live case, 42 leads) -> default deployment id
json.dump(live, open(out / "lethbridge.json", "w"))
# ok: pretend full coverage + champions
ok = copy.deepcopy(live); ok["status"] = "ok"
json.dump(ok, open(out / "ok.json", "w"))
# degraded: baseline fell back
deg = copy.deepcopy(live); deg["status"] = "degraded"
deg["model_versions"] = {"temp": "baseline", "pop": "baseline"}
json.dump(deg, open(out / "degraded.json", "w"))
print("wrote", list(p.name for p in out.iterdir()))
PY
```

- [ ] **Step 2: Serve the dashboard**

```bash
( cd dashboard && python3 -m http.server 8731 --bind 127.0.0.1 ) &
```

- [ ] **Step 3: Render each state via Playwright (light, dark, mobile)**

Using the Playwright MCP: navigate to each URL, emulate `colorScheme` light/dark (via `browser_run_code_unsafe` → `page.emulateMedia({colorScheme})`), resize to 390×840 for mobile, and screenshot. Confirm by eye:
- `http://127.0.0.1:8731/index.html` — stale pill (amber), `42 / 48 h`, champion badges, now-hero at +1 h.
- `http://127.0.0.1:8731/index.html?deployment=ok` — ok pill (green).
- `http://127.0.0.1:8731/index.html?deployment=degraded` — degraded pill (red), both models show `baseline`.
- `http://127.0.0.1:8731/index.html?deployment=nope` — "Forecast unavailable." (404 path).

Checklist: light + dark both legible; chart line + bars + 0 °C line render; table shows PoP as `%` only (no bar); mobile single-column reflow OK.

- [ ] **Step 4: Tear down the scratch fixtures and server**

```bash
rm -rf dashboard/forecasts
kill %1 2>/dev/null || true
```

(`dashboard/forecasts/` is a scratch render target — it must NOT be committed; the real forecasts live only on gh-pages.)

---

### Task 6: Docs, cleanup, full gate

**Files:**
- Modify: `dashboard/README.md`
- Modify: `README.md` (root "Project status")
- Delete: `dashboard/_prototype.html`, `dashboard/_prototype_forecast.json`, `dashboard/_PROTOTYPE_NOTES.md`, root `proto-A-light.png`, `proto-A-dark.png`, `proto-A-mobile.png`, `proto-B-light.png`, `proto-C-light.png`, `proto-C-dark.png`

- [ ] **Step 1: Rewrite `dashboard/README.md`**

```markdown
# Dashboard (thin client)

Static files (`index.html`, `styles.css`, `app.js`) served from the `gh-pages` branch root and
published there by `.github/workflows/inference.yml` (hourly), alongside `forecasts/<id>.json`.

Reads `forecasts/${deployment}.json` from the same origin (`?deployment=` selects it, default
`lethbridge`) and renders: an A-style header (deployment + status pill + the next-1-hour "now"
hero), a dual-axis temperature/PoP chart (hand-rolled inline SVG), a run-metadata + champion
`model_versions` panel, and a scrollable hourly table. Local time only; auto light/dark via
`prefers-color-scheme`.

Accepts any forecast whose `schema_version` major is **1** (`SUPPORTED_SCHEMA_MAJOR`). No build
step, no secrets, no raw observations — only the derived forecast document (ADR-0009).
```

- [ ] **Step 2: Update the root README "Project status"**

Find the "Project status" section in `README.md` and add a line noting the dashboard is built and auto-deployed. Example wording to insert:

```markdown
- **Dashboard:** redesigned static thin client (`dashboard/`), auto-deployed to gh-pages by the
  hourly inference workflow; renders the published forecast (chart + hourly table + status/model
  metadata).
```

(Match the surrounding bullet/section formatting in the actual file.)

- [ ] **Step 3: Delete the prototype artifacts**

```bash
git rm dashboard/_prototype.html dashboard/_prototype_forecast.json dashboard/_PROTOTYPE_NOTES.md
rm -f proto-A-light.png proto-A-dark.png proto-A-mobile.png proto-B-light.png proto-C-light.png proto-C-dark.png
```

(The `proto-*.png` were never committed — `rm -f` is enough; `git rm` only the tracked prototype files.)

- [ ] **Step 4: Run the full gate**

Run:
```bash
uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
```
Expected: all green. (No `src/` change, so `lint-imports`/`pyright` are unaffected; `pytest` includes the two new dashboard guards.)

- [ ] **Step 5: Commit**

```bash
git add README.md dashboard/README.md
git add -A dashboard
git commit -m "docs: dashboard README + project status; remove prototype"
```

---

## Self-Review

**Spec coverage:**
- Defect 1 (never deployed) → Task 4. ✓
- Defect 2 (schema rejection) → Task 3 (`SUPPORTED_SCHEMA_MAJOR`, major-match in `load()`). ✓
- Locked decisions (SVG, light/dark, local time, `?deployment=`, major-match) → Tasks 1–3. ✓
- Layout A-header + C-body (now-hero, dual-axis chart, metadata/model panel, hourly table pop-%-only) → Task 3 `view()`. ✓ (No horizon cards — matches the spec's "header = name + status + now-hero" decision.)
- `.nojekyll` + publish from inference.yml only → Task 4. ✓
- Error/states (loading, fetch fail, schema mismatch, empty series, status pills) → Task 2 (loading) + Task 3 (`load()`/`message()`/`statusPill`). ✓
- Testing (two Python guards + visual pass) → Tasks 3, 4, 5. ✓
- Docs (dashboard README, root status) + prototype cleanup → Task 6. ✓
- Full gate green → Task 6 Step 4. ✓

**Placeholder scan:** No TBD/TODO; all code blocks complete; the only "match the actual file" note is the root-README bullet whose exact neighbours can't be known until edit time — wording supplied.

**Type/name consistency:** `SUPPORTED_SCHEMA_MAJOR` used identically in `app.js` and the guard regex; CSS class names (`.head`, `.now`, `.now-sub`, `.chart`, `.legend`, `.meta`, `.models`, `.mtask`, `.table-head`, `.table-wrap`, `table.hourly`, `.attrib`, `.msg`, `.pill`, `.badge`, `.panel`) all defined in Task 1 and emitted by Task 3 `view()`. `REPO_ROOT`/`APP_JS`/`INFERENCE_YML` defined once, reused. ✓
