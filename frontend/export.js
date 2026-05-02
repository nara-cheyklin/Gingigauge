// Handles export (download & print) of the annotated output image and results.
// Call initExport(imageSrc, resultData) once the output is ready to display.

function initExport(imageSrc, resultData) {
  const exportContainer = document.getElementById("exportContainer");
  if (!exportContainer) return;

  exportContainer.style.display = "block";

  const outputImage = document.getElementById("outputImage");
  if (outputImage) {
    outputImage.src = imageSrc;
  }

  if (resultData) {
    const kgwValue = document.getElementById("kgwValue");
    const confidenceValue = document.getElementById("confidenceValue");
    const interpretationValue = document.getElementById("interpretationValue");

    if (kgwValue) kgwValue.textContent = formatKgw(resultData.kgw_mm);
    if (confidenceValue) confidenceValue.textContent = (resultData.confidence * 100).toFixed(1) + "%";
    if (interpretationValue) interpretationValue.textContent = resultData.interpretation;
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

  const rows = resultData ? `
    <tr><th>KGW Measurement</th><td>${formatKgw(resultData.kgw_mm)}</td></tr>
    <tr><th>Confidence</th><td>${(resultData.confidence * 100).toFixed(1)}%</td></tr>
    <tr><th>Interpretation</th><td>${resultData.interpretation}</td></tr>
  ` : "";

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
        table { border-collapse: collapse; width: 100%; max-width: 480px; }
        th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 14px; }
        th { background: #f4f4f4; font-weight: 600; }
        .footer { margin-top: 30px; font-size: 11px; color: #888; }
      </style>
    </head>
    <body>
      <h1>GINGIGAUGE</h1>
      <div class="subtitle">Deep-learning Based Measurement of KGW</div>
      <img src="${imageSrc}" alt="KGW Output" />
      <table>${rows}</table>
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

function formatKgw(kgw) {
  return typeof kgw === "number" ? kgw + " mm" : kgw;
}
