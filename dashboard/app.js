// Thin client: read the published forecast JSON and render it. No raw data, no secrets.
const DEPLOYMENT_ID = "lethbridge";
const SCHEMA_VERSION = "1";

async function load() {
  const res = await fetch(`forecasts/${DEPLOYMENT_ID}.json`, { cache: "no-store" });
  if (!res.ok) {
    document.getElementById("status").textContent = "Forecast unavailable.";
    return;
  }
  const doc = await res.json();
  if (doc.schema_version !== SCHEMA_VERSION) {
    document.getElementById("status").textContent =
      `Unsupported schema_version ${doc.schema_version}.`;
    return;
  }
  document.getElementById("status").textContent =
    `${doc.status} — updated ${doc.last_updated}`;
  document.getElementById("series").textContent = JSON.stringify(doc.series, null, 2);
  document.getElementById("attribution").textContent = (doc.attribution || []).join(" · ");
}

load();
