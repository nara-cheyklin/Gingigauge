// Handles export (download & print) of the annotated output image and results.
// Call initExport(imageSrc, resultData) once the output is ready to display.

function initExport(imageSrc, resultData) {
  const exportContainer = document.getElementById("exportContainer");
  if (!exportContainer) return;

  exportContainer.style.display = "block";

  const outputImage = document.getElementById("outputImage");
  if (outputImage) outputImage.src = imageSrc;

  if (resultData) {
    const teeth = resultData.teeth || [];

    // Summary: categorise each tooth individually
    const healthy = [], atRisk = [], recession = [];
    for (const t of teeth) {
      if (t.inferred_tooth_id && typeof t.kgw_mm === "number") {
        if (t.kgw_mm >= 3.5) healthy.push(t.inferred_tooth_id);
        else if (t.kgw_mm >= 2.0) atRisk.push(t.inferred_tooth_id);
        else recession.push(t.inferred_tooth_id);
      }
    }
    const el = (id) => document.getElementById(id);
    if (el("healthyTeeth"))   el("healthyTeeth").textContent   = healthy.length   ? healthy.join(", ")   : "—";
    if (el("atRiskTeeth"))    el("atRiskTeeth").textContent    = atRisk.length    ? atRisk.join(", ")    : "—";
    if (el("recessionTeeth")) el("recessionTeeth").textContent = recession.length ? recession.join(", ") : "—";

    // Per-tooth table
    const teethContainer = document.getElementById("teethTableContainer");
    const teethBody = document.getElementById("teethTableBody");

    if (teethBody && teeth.length > 0) {
      teethBody.innerHTML = buildDentalChartHTML(teeth, resultData.interpretation);
      if (teethContainer) teethContainer.classList.remove("hidden");
    } else if (teethContainer) {
      teethContainer.classList.add("hidden");
    }
  }

  document.getElementById("downloadBtn").onclick = function () {
    downloadImage(imageSrc);
  };

  document.getElementById("printBtn").onclick = function () {
    printResult(imageSrc, resultData);
  };
}

function downloadImage(imageSrc) {
  const link = document.createElement("a");
  link.href = imageSrc;
  link.download = "kgw_result_" + getTimestamp() + ".jpg";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function printResult(imageSrc, resultData) {
  const printWindow = window.open("", "_blank");

  const teeth = (resultData && resultData.teeth) || [];

  let summaryRows = "";
  if (resultData) {
    const healthy = [], atRisk = [], recession = [];
    for (const t of teeth) {
      if (t.inferred_tooth_id && typeof t.kgw_mm === "number") {
        if (t.kgw_mm >= 3.5) healthy.push(t.inferred_tooth_id);
        else if (t.kgw_mm >= 2.0) atRisk.push(t.inferred_tooth_id);
        else recession.push(t.inferred_tooth_id);
      }
    }
    summaryRows = `
      <tr>
        <th style="color:#15803d">Healthy</th>
        <td style="color:#15803d;font-family:monospace">${healthy.length ? healthy.join(", ") : "—"}</td>
      </tr>
      <tr>
        <th style="color:#f97316">At Risk</th>
        <td style="color:#f97316;font-family:monospace">${atRisk.length ? atRisk.join(", ") : "—"}</td>
      </tr>
      <tr>
        <th style="color:#dc2626">Recession</th>
        <td style="color:#dc2626;font-family:monospace">${recession.length ? recession.join(", ") : "—"}</td>
      </tr>`;
  }

  const teethSection = teeth.length > 0
    ? `<h2 style="font-size:15px;margin:24px 0 8px">Per-Tooth Measurements</h2>` +
      buildDentalChartPrint(teeth, resultData && resultData.interpretation)
    : "";

  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>GINGIGAUGE - KGW Result</title>
      <style>
        body { font-family: Arial, sans-serif; padding: 30px; color: #111; }
        h1 { font-size: 22px; margin-bottom: 4px; }
        .subtitle { color: #555; font-size: 13px; margin-bottom: 20px; }
        img { max-width: 100%; max-height: 400px; display: block; margin-bottom: 20px; border: 1px solid #ccc; }
        table { border-collapse: collapse; width: 100%; max-width: 520px; margin-bottom: 8px; }
        th, td { border: 1px solid #ddd; padding: 7px 12px; text-align: left; font-size: 13px; }
        th { background: #f4f4f4; font-weight: 600; }
        .footer { margin-top: 30px; font-size: 11px; color: #888; }
      </style>
    </head>
    <body>
      <h1>GINGIGAUGE</h1>
      <div class="subtitle">Deep-learning Based Measurement of KGW</div>
      <img src="${imageSrc}" alt="KGW Output" />
      <table>${summaryRows}</table>
      ${teethSection}
      <div class="footer">
        A collaborative project between the Faculty of Engineering and the Faculty of Dentistry, Chulalongkorn University
        &nbsp;&nbsp;|&nbsp;&nbsp; Printed: ${new Date().toLocaleString()}
      </div>
    </body>
    </html>
  `);

  printWindow.document.close();
  printWindow.focus();
  printWindow.onload = function () {
    printWindow.print();
    printWindow.close();
  };
}

function getTimestamp() {
  const d = new Date();
  return d.getFullYear() +
    String(d.getMonth() + 1).padStart(2, "0") +
    String(d.getDate()).padStart(2, "0") + "_" +
    String(d.getHours()).padStart(2, "0") +
    String(d.getMinutes()).padStart(2, "0") +
    String(d.getSeconds()).padStart(2, "0");
}

// ---- dental chart helpers ---------------------------------------------------

const _JAWS = [
  { label: "Upper Jaw", left: ["18","17","16","15","14","13","12","11"], right: ["21","22","23","24","25","26","27","28"] },
  { label: "Lower Jaw", left: ["48","47","46","45","44","43","42","41"], right: ["31","32","33","34","35","36","37","38"] },
];

function _kgwLookup(teeth) {
  const m = {};
  for (const t of teeth) if (t.inferred_tooth_id) m[String(t.inferred_tooth_id)] = t.kgw_mm;
  return m;
}

function buildDentalChartHTML(teeth, interpretation) {
  const lookup = _kgwLookup(teeth);

  const thCls = 'class="font-semibold text-slate-600 bg-slate-50 border border-slate-200 text-center" style="min-width:26px;padding:2px 4px;font-size:10px"';
  const tdCls = 'class="border border-slate-100 text-center" style="padding:2px 4px;font-size:10px"';
  const sepTh = 'class="bg-black border-0 p-0" style="width:3px"';
  const sepTd = 'class="bg-black border-0 p-0" style="width:3px"';

  const tables = _JAWS.map(({ label, left, right }) => {
    const ids = [...left, null, ...right];
    const idRow  = ids.map(id => id === null ? `<th ${sepTh}></th>` : `<th ${thCls}>${id}</th>`).join("");
    const kgwRow = ids.map(id => {
      if (id === null) return `<td ${sepTd}></td>`;
      const v = lookup[id];
      const color = _toothColorHex(v);
      return typeof v === "number"
        ? `<td ${tdCls}><span style="color:${color};font-family:monospace;font-weight:700;font-size:10px">${v.toFixed(2)}</span></td>`
        : `<td ${tdCls}><span style="color:#9ca3af;font-size:10px">—</span></td>`;
    }).join("");
    return `
      <div class="mb-5">
        <p class="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-1">${label}</p>
        <div class="overflow-x-auto rounded-lg border border-slate-200 bg-white shadow-sm">
          <table class="border-collapse text-center">${
            `<thead><tr>${idRow}</tr></thead><tbody><tr>${kgwRow}</tr></tbody>`
          }</table>
        </div>
      </div>`;
  }).join("");

  const legend = `
    <div class="flex flex-wrap gap-4 mt-1 text-xs" style="color:#475569">
      <span><span style="font-weight:600;color:#15803d">Green</span> — Healthy (≥ 3.5 mm)</span>
      <span><span style="font-weight:600;color:#f97316">Orange</span> — At Risk (2.0 – 3.5 mm)</span>
      <span><span style="font-weight:600;color:#dc2626">Red</span> — Recession (&lt; 2.0 mm)</span>
    </div>`;

  return tables + legend;
}

function _toothColorHex(kgw) {
  if (typeof kgw !== "number") return "#bbb";
  if (kgw < 2.0) return "#dc2626";
  if (kgw < 3.5) return "#f97316";
  return "#15803d";
}

function buildDentalChartPrint(teeth, interpretation) {
  const lookup = _kgwLookup(teeth);

  const cellBase = "border:1px solid #ddd;padding:2px 4px;text-align:center;font-size:10px;min-width:26px;white-space:nowrap";
  const thStyle  = `${cellBase};background:#f4f4f4;font-weight:600`;
  const sepStyle = "width:3px;background:#000;border:none;padding:0";

  return _JAWS.map(({ label, left, right }) => {
    const ids = [...left, null, ...right];
    const idRow  = ids.map(id => id === null ? `<th style="${sepStyle}"></th>` : `<th style="${thStyle}">${id}</th>`).join("");
    const kgwRow = ids.map(id => {
      if (id === null) return `<td style="${sepStyle}"></td>`;
      const v = lookup[id];
      const color = _toothColorHex(v);
      return `<td style="${cellBase}">${
        typeof v === "number"
          ? `<span style="color:${color};font-family:monospace;font-weight:600">${v.toFixed(2)}</span>`
          : `<span style="color:#bbb">—</span>`
      }</td>`;
    }).join("");
    return `
      <p style="font-size:11px;font-weight:600;color:#888;text-transform:uppercase;margin:16px 0 4px">${label}</p>
      <div style="overflow-x:auto">
        <table style="border-collapse:collapse;text-align:center;width:auto;max-width:none">
          <thead><tr>${idRow}</tr></thead>
          <tbody><tr>${kgwRow}</tr></tbody>
        </table>
      </div>`;
  }).join("") + `
    <p style="font-size:11px;color:#555;margin-top:10px">
      <span style="color:#15803d;font-weight:600">Green</span> — Healthy (&ge; 3.5 mm)
      &nbsp;&nbsp;
      <span style="color:#f97316;font-weight:600">Orange</span> — At Risk (2.0 – 3.5 mm)
      &nbsp;&nbsp;
      <span style="color:#dc2626;font-weight:600">Red</span> — Recession (&lt; 2.0 mm)
    </p>`;
}

