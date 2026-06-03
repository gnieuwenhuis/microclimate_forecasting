// Thin client (ADR-0009): read the published forecast JSON and render it.
// No raw data, no secrets, no build step. Hand-rolled SVG, local time, auto light/dark.
const SUPPORTED_SCHEMA_MAJOR = "1";
const params = new URLSearchParams(location.search);
const DEPLOYMENT = ((params.get("deployment") || "lethbridge").toLowerCase().replace(/[^a-z0-9_-]/g, "")) || "lethbridge";

// The dashboard shows a sliding window of the next WINDOW_HOURS, computed from the browser clock
// at render time. The published JSON holds the full 48 h run; filtering client-side means the
// window stays correct ("next 12 h from now") without re-publishing every hour.
const WINDOW_HOURS = 12;

// ---- formatters (local time) ----
const fmtHour = (iso) => new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
const fmtFull = (iso) => new Date(iso).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" });
const fmtDay = (iso) => new Date(iso).toLocaleDateString([], { weekday: "short" });
const dayKey = (iso) => new Date(iso).toLocaleDateString();
const fmtT = (c) => `${Math.round(c)}°`;
const fmtP = (p) => `${Math.round(p * 100)}%`;
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function ago(iso) {
  const mins = Math.round((Date.now() - new Date(iso)) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const h = Math.round(mins / 60);
  if (h < 24) return `${h} h ago`;
  return `${Math.round(h / 24)} d ago`;
}

// The next WINDOW_HOURS of the forecast from now (sliding, client-side). Drops already-elapsed
// leads so the view starts at the current hour rather than the model cycle's init time.
function windowSeries(series) {
  const now = Date.now();
  const end = now + WINDOW_HOURS * 3600000;
  return series.filter((step) => {
    const t = Date.parse(step.valid_time);
    return t >= now && t < end; // [now, now+WINDOW_HOURS): exclusive end caps at WINDOW_HOURS steps
  });
}

// A run older than this (≈3 missed hourly updates) is flagged "Delayed".
const FRESH_MAX_MIN = 180;

// Viewer-relative status: what the person looking at the next-WINDOW_HOURS view cares about —
// is it fresh and does it cover the window? This is deliberately NOT the published run-quality
// `status` (whose "stale" means horizon-truncated below 48 h, irrelevant to a 12 h view).
function displayStatus(doc) {
  const ageMin = (Date.now() - Date.parse(doc.last_updated)) / 60000;
  const lastLead = Date.parse(doc.series[doc.series.length - 1].valid_time);
  const coversWindow = lastLead >= Date.now() + WINDOW_HOURS * 3600000;
  if (doc.status === "degraded") {
    return { cls: "degraded", label: "Degraded", title: "A model fell back to baseline — an expected champion was unusable." };
  }
  if (ageMin > FRESH_MAX_MIN) {
    return { cls: "stale", label: "Delayed", title: `Last updated ${ago(doc.last_updated)}; the hourly update may be behind.` };
  }
  if (!coversWindow) {
    return { cls: "stale", label: "Delayed", title: `The forecast doesn't extend a full ${WINDOW_HOURS} h ahead.` };
  }
  return { cls: "ok", label: "Live", title: `Fresh — updated ${ago(doc.last_updated)}, covering the next ${WINDOW_HOURS} h.` };
}
function pill(d) {
  return `<span class="pill ${d.cls}" title="${d.title}"><span class="dot"></span>${d.label}</span>`;
}

const scaler = (min, max, lo, hi) => (v) => lo + (hi - lo) * (max === min ? 0.5 : (v - min) / (max - min));

// ---- dual-axis SVG: temperature line + PoP bars, with day-transition markers ----
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
  // Day-transition markers: a faint vertical rule + weekday label where the local date rolls over.
  let days = "";
  for (let i = 1; i < series.length; i++) {
    if (dayKey(series[i].valid_time) !== dayKey(series[i - 1].valid_time)) {
      const dx = x(i);
      days += `<line x1="${dx.toFixed(1)}" y1="${pad.t}" x2="${dx.toFixed(1)}" y2="${h - pad.b}" stroke="var(--zero)" stroke-dasharray="2 3" opacity=".75"/>`;
      days += `<text x="${(dx + 4).toFixed(1)}" y="${pad.t + 10}" font-size="10" font-weight="600" fill="var(--muted)">${fmtDay(series[i].valid_time)}</text>`;
    }
  }
  // x-axis time labels: ~6 across regardless of window length.
  const xstep = Math.max(1, Math.ceil(series.length / 6));
  let xl = "";
  for (let i = 0; i < series.length; i += xstep) {
    xl += `<text x="${x(i)}" y="${h - 8}" text-anchor="middle" font-size="10" fill="var(--muted)">${fmtHour(series[i].valid_time)}</text>`;
  }
  // viewBox width is set to the rendered pixel width by drawChart() so the mapping is 1:1 and
  // text/strokes are never distorted (the old preserveAspectRatio="none" squashed labels on mobile).
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" style="height:${h}px; display:block">${grid}${days}${bars}<path d="${dLine}" fill="none" stroke="var(--temp)" stroke-width="2.5" stroke-linejoin="round"/>${xl}</svg>`;
}

// ---- full view (A-header + C body). `win` is the windowed series (next WINDOW_HOURS). ----
function view(doc, win) {
  const next = win[0]; // soonest upcoming hour
  const disp = displayStatus(doc);
  const rows = win
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
    .map(([t, v]) => `<div><span class="mtask">${esc(t)}</span> <span class="badge ${v !== "baseline" ? "champ" : ""}">${esc(v)}</span></div>`)
    .join("");
  return `
  <header class="head">
    <div>
      <h1>${esc(cap(doc.deployment_id))}</h1>
      <div class="muted sub">Issued ${fmtFull(doc.issue_time)} · updated ${ago(doc.last_updated)}</div>
    </div>
    ${pill(disp)}
  </header>
  <section class="now">
    <div class="big temp">${fmtT(next.temp_c)}</div>
    <div class="now-sub muted">Forecast for ${fmtHour(next.valid_time)}<br><b>${fmtP(next.pop)}</b> chance of precip</div>
  </section>
  <section class="panel chart">
    <div class="legend muted"><span class="temp">— temperature</span> &nbsp; <span class="pop">▮ precip probability</span></div>
    <div class="chartbox"></div>
  </section>
  <section class="panel meta">
    <dl>
      <dt>Status</dt><dd>${pill(disp)}</dd>
      <dt>Issued</dt><dd>${fmtFull(doc.issue_time)}</dd>
      <dt>Updated</dt><dd>${ago(doc.last_updated)}</dd>
      <dt>Showing</dt><dd>next ${WINDOW_HOURS} h</dd>
      <dt>Run horizon</dt><dd>${doc.series.length} / 48 h</dd>
      <dt>Schema</dt><dd>${esc(doc.schema_version)}</dd>
    </dl>
    <div class="models">${models}</div>
  </section>
  <section class="panel">
    <div class="table-head">Next ${WINDOW_HOURS} hours (${win.length})</div>
    <div class="table-wrap"><table class="hourly">
      <thead><tr><th>Lead</th><th>Valid (local)</th><th>Temp</th><th>Precip</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </section>
  <footer class="attrib">${doc.attribution.map((a) => `<div>${esc(a)}</div>`).join("")}</footer>`;
}

function message(html) {
  document.getElementById("app").innerHTML = `<div class="msg">${html}</div>`;
}

// Draw the chart sized to the real container width so the SVG maps 1:1 to pixels (no horizontal
// squash of the axis labels on narrow screens). Re-runs on resize/orientation change.
let _win = null;
function drawChart() {
  const box = document.querySelector(".chartbox");
  if (!box || !_win) return;
  const w = Math.round(box.clientWidth) || 920; // measured px → viewBox is 1:1 (no distortion)
  box.innerHTML = dualSVG(_win, { w });
}
addEventListener("resize", drawChart);

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
    const win = windowSeries(doc.series);
    if (win.length === 0) {
      message("No upcoming forecast hours — the latest run is stale.");
      return;
    }
    _win = win;
    document.getElementById("app").innerHTML = view(doc, win);
    drawChart();
  } catch (e) {
    message("Forecast unavailable.");
    console.error(e);
  }
}

load();
