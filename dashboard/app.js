// Thin client (ADR-0009): read the published forecast JSON and render it.
// No raw data, no secrets, no build step. Hand-rolled SVG, local time, auto light/dark.
const SUPPORTED_SCHEMA_MAJOR = "1";
const params = new URLSearchParams(location.search);
const DEPLOYMENT = ((params.get("deployment") || "lethbridge").toLowerCase().replace(/[^a-z0-9_-]/g, "")) || "lethbridge";

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
