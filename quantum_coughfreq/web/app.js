const state = {
  analysis: null,
  activeSpectrogramTab: "mel",
  config: {
    supported_formats: [],
    default_dataset_dir: "",
  },
};

const elements = {};

document.addEventListener("DOMContentLoaded", async () => {
  bindElements();
  bindEvents();
  await loadConfig();
  renderInitialState();
});

function bindElements() {
  elements.audioInput = document.getElementById("audioInput");
  elements.audioPlayer = document.getElementById("audioPlayer");
  elements.analyzeButton = document.getElementById("analyzeButton");
  elements.waveformCanvas = document.getElementById("waveformCanvas");
  elements.sphereCanvas = document.getElementById("sphereCanvas");
  elements.spectrogramCanvas = document.getElementById("spectrogramCanvas");
  elements.latentCanvas = document.getElementById("latentCanvas");
  elements.fileName = document.getElementById("fileName");
  elements.fileDuration = document.getElementById("fileDuration");
  elements.fileSampleRate = document.getElementById("fileSampleRate");
  elements.resultBadge = document.getElementById("resultBadge");
  elements.confidenceValue = document.getElementById("confidenceValue");
  elements.tbProbabilityValue = document.getElementById("tbProbabilityValue");
  elements.signatureValue = document.getElementById("signatureValue");
  elements.classBars = document.getElementById("classBars");
  elements.pipelineLog = document.getElementById("pipelineLog");
  elements.circuitGrid = document.getElementById("circuitGrid");
  elements.circuitMeta = document.getElementById("circuitMeta");
  elements.spectrogramLegend = document.getElementById("spectrogramLegend");
  elements.heroRiskLabel = document.getElementById("heroRiskLabel");
  elements.datasetPathInput = document.getElementById("datasetPathInput");
  elements.datasetMaxFilesInput = document.getElementById("datasetMaxFilesInput");
  elements.datasetAnalyzeButton = document.getElementById("datasetAnalyzeButton");
  elements.datasetSummaryCards = document.getElementById("datasetSummaryCards");
  elements.topRiskList = document.getElementById("topRiskList");
  elements.skippedList = document.getElementById("skippedList");
  elements.tabButtons = [...document.querySelectorAll(".tab-button")];
}

function bindEvents() {
  elements.audioInput.addEventListener("change", handleFileSelection);
  elements.analyzeButton.addEventListener("click", analyzeSelectedFile);
  elements.datasetAnalyzeButton.addEventListener("click", analyzeDataset);
  elements.tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.activeSpectrogramTab = button.dataset.tab;
      elements.tabButtons.forEach((item) => item.classList.toggle("active", item === button));
      drawSpectrogramPanel();
    });
  });
}

async function loadConfig() {
  try {
    const response = await fetch("/api/dashboard-config");
    state.config = await response.json();
    elements.datasetPathInput.value = state.config.default_dataset_dir || "";
  } catch (error) {
    console.error(error);
  }
}

function renderInitialState() {
  drawWaveform(new Array(512).fill(0));
  drawSphere({
    infection_type: "Waiting",
    confidence_score: 0,
    tb_probability: 0,
    sphere: { radius: 0.45, glow: 0.2, energy: 0.15 },
  });
  renderPipelineLog([
    "Awaiting cough upload",
    "Select a file to begin waveform and quantum screening",
    "Dataset folder path can be analyzed below",
  ]);
  renderClassBars({ Viral: 0, Bacterial: 0, TB: 0 });
  renderCircuit({
    qubits: 4,
    layers: 3,
    rows: Array.from({ length: 4 }, (_, index) => ({
      qubit: `q${index}`,
      gates: [
        { label: "H", kind: "prep" },
        { label: "Rz(x)", kind: "encode" },
        { label: "Ry(x)", kind: "encode" },
        { label: "ZZ", kind: "entangle" },
        { label: "Rx(w)", kind: "rotation" },
        { label: "Ry(w)", kind: "rotation" },
        { label: "Rz(w)", kind: "rotation" },
        { label: "M", kind: "measure" },
      ],
    })),
  });
  drawLatent({
    points: [],
    sample: { x: 0, y: 0, label: "Waiting", color: "#51f0ff" },
  });
  renderDatasetSummary(null);
  drawSpectrogramPanel();
}

function handleFileSelection(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }

  elements.fileName.textContent = file.name;
  elements.fileDuration.textContent = "Pending";
  elements.fileSampleRate.textContent = "Pending";
  const objectUrl = URL.createObjectURL(file);
  elements.audioPlayer.src = objectUrl;
  renderPipelineLog([
    `File selected: ${file.name}`,
    "Ready for quantum analysis",
    "Press Start Quantum Analysis to run the backend",
  ]);
}

async function analyzeSelectedFile() {
  const file = elements.audioInput.files?.[0];
  if (!file) {
    renderPipelineLog(["Please choose a cough audio file first."]);
    return;
  }

  const formData = new FormData();
  formData.append("audio_file", file);

  elements.analyzeButton.disabled = true;
  elements.analyzeButton.textContent = "Analyzing...";
  renderPipelineLog([
    `Uploading ${file.name}`,
    "Running MFCC, Mel spectrum, and quantum screening",
    "Waiting for classifier response",
  ]);

  try {
    const response = await fetch("/api/analyze-cough", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Analysis failed");
    }
    state.analysis = payload;
    renderAnalysis(payload);
  } catch (error) {
    console.error(error);
    renderPipelineLog([`Analysis failed: ${error.message}`]);
  } finally {
    elements.analyzeButton.disabled = false;
    elements.analyzeButton.textContent = "Start Quantum Analysis";
  }
}

async function analyzeDataset() {
  const datasetDir = elements.datasetPathInput.value.trim();
  const maxFilesValue = elements.datasetMaxFilesInput.value.trim();
  if (!datasetDir) {
    renderDatasetSummary({
      error: "Enter a dataset folder path before running the scan.",
    });
    return;
  }

  elements.datasetAnalyzeButton.disabled = true;
  elements.datasetAnalyzeButton.textContent = "Scanning...";

  try {
    const response = await fetch("/api/classify-dataset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_dir: datasetDir,
        top_k: 10,
        workers: 1,
        max_files: maxFilesValue ? Number(maxFilesValue) : null,
        detailed: false,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Dataset analysis failed");
    }
    renderDatasetSummary(payload);
  } catch (error) {
    console.error(error);
    renderDatasetSummary({ error: error.message });
  } finally {
    elements.datasetAnalyzeButton.disabled = false;
    elements.datasetAnalyzeButton.textContent = "Analyze Folder";
  }
}

function renderAnalysis(payload) {
  elements.fileName.textContent = payload.audio_file;
  elements.fileDuration.textContent = `${payload.duration_seconds.toFixed(2)}s`;
  elements.fileSampleRate.textContent = `${payload.sample_rate} Hz`;
  elements.resultBadge.textContent = `${payload.tb_risk_level} Risk`;
  elements.resultBadge.className = `result-badge result-${payload.tb_risk_level.toLowerCase()}`;
  elements.confidenceValue.textContent = `${payload.confidence_score.toFixed(2)}%`;
  elements.tbProbabilityValue.textContent = payload.tb_probability.toFixed(4);
  elements.signatureValue.textContent = `${payload.infection_type} / ${payload.acoustic_signature}`;
  elements.heroRiskLabel.textContent = `${payload.infection_type} ${payload.tb_risk_level}`;

  renderPipelineLog(payload.pipeline_log);
  renderClassBars(payload.class_probabilities);
  renderCircuit(payload.quantum_circuit);
  drawWaveform(payload.waveform_points);
  drawSphere(payload);
  drawSpectrogramPanel();
  drawLatent(payload.latent_projection);
}

function renderPipelineLog(lines) {
  elements.pipelineLog.innerHTML = "";
  lines.forEach((line) => {
    const item = document.createElement("li");
    item.textContent = line;
    elements.pipelineLog.appendChild(item);
  });
}

function renderClassBars(probabilities) {
  elements.classBars.innerHTML = "";
  Object.entries(probabilities).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "class-row";
    row.innerHTML = `
      <span>${label}</span>
      <div class="class-track"><div class="class-fill" style="width:${Math.max(4, value * 100)}%"></div></div>
      <strong>${(value * 100).toFixed(1)}%</strong>
    `;
    elements.classBars.appendChild(row);
  });
}

function renderCircuit(circuit) {
  elements.circuitMeta.textContent = `${circuit.qubits} Qubits • ${circuit.layers} Layers`;
  elements.circuitGrid.innerHTML = "";

  circuit.rows.forEach((row) => {
    const rowNode = document.createElement("div");
    rowNode.className = "circuit-row";
    const label = document.createElement("div");
    label.className = "circuit-label";
    label.textContent = row.qubit;
    rowNode.appendChild(label);

    row.gates.forEach((gate) => {
      const gateNode = document.createElement("div");
      gateNode.className = `gate gate-${gate.kind}`;
      gateNode.textContent = gate.label;
      rowNode.appendChild(gateNode);
    });

    elements.circuitGrid.appendChild(rowNode);
  });
}

function drawWaveform(points) {
  const canvas = elements.waveformCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const background = ctx.createLinearGradient(0, 0, 0, height);
  background.addColorStop(0, "#0a1020");
  background.addColorStop(1, "#13192c");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  for (let i = 1; i < 4; i += 1) {
    const y = (height / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#f279ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = (index / Math.max(points.length - 1, 1)) * width;
    const y = height / 2 - point * (height * 0.32);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function drawSphere(payload) {
  const canvas = elements.sphereCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * (payload.sphere?.radius || 0.45) * 0.52;
  const glowStrength = payload.sphere?.glow || 0.3;

  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createRadialGradient(centerX, centerY, radius * 0.12, centerX, centerY, radius * 1.4);
  gradient.addColorStop(0, `rgba(201, 255, 226, ${0.7 + glowStrength * 0.2})`);
  gradient.addColorStop(0.45, `rgba(123, 255, 172, ${0.38 + glowStrength * 0.25})`);
  gradient.addColorStop(1, "rgba(35, 63, 52, 0)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius * 1.45, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "rgba(82, 255, 163, 0.18)";
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(240, 255, 184, 0.92)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(centerX, centerY + 4, radius * 0.72, 0.2 * Math.PI, 0.88 * Math.PI);
  ctx.stroke();

  ctx.strokeStyle = "rgba(255,255,255,0.14)";
  ctx.beginPath();
  ctx.ellipse(centerX, centerY, radius * 0.92, radius * 0.22, 0, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "#faffcb";
  ctx.font = "700 28px 'Avenir Next Condensed', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(payload.infection_type || "Waiting", centerX, centerY + 6);
  ctx.font = "600 20px 'Avenir Next Condensed', sans-serif";
  ctx.fillText(`${(payload.confidence_score || 0).toFixed(2)}%`, centerX, centerY + 34);
}

function drawSpectrogramPanel() {
  const analysis = state.analysis;
  const canvas = elements.spectrogramCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#09111f";
  ctx.fillRect(0, 0, width, height);

  if (!analysis) {
    ctx.fillStyle = "#8ea3cc";
    ctx.font = "18px 'Avenir Next Condensed', sans-serif";
    ctx.fillText("Upload and analyze a cough file to view spectral data.", 24, 42);
    elements.spectrogramLegend.innerHTML = "";
    return;
  }

  if (state.activeSpectrogramTab === "psd") {
    drawBars(ctx, width, height, analysis.psd_curve);
    elements.spectrogramLegend.innerHTML = analysis.frequency_bands
      .map((band) => `<span class="legend-chip" style="--chip-color:${band.color}">${band.label}: ${(band.value * 100).toFixed(1)}%</span>`)
      .join("");
    return;
  }

  const matrix = state.activeSpectrogramTab === "mfcc" ? analysis.mfcc_heatmap : analysis.mel_spectrogram;
  drawHeatmap(ctx, width, height, matrix);
  elements.spectrogramLegend.innerHTML = analysis.frequency_bands
    .map((band) => `<span class="legend-chip" style="--chip-color:${band.color}">${band.label}</span>`)
    .join("");
}

function drawHeatmap(ctx, width, height, matrix) {
  const rows = matrix.length;
  const cols = matrix[0]?.length || 1;
  const cellWidth = width / cols;
  const cellHeight = height / rows;

  matrix.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      ctx.fillStyle = heatColor(value);
      ctx.fillRect(colIndex * cellWidth, rowIndex * cellHeight, cellWidth + 1, cellHeight + 1);
    });
  });
}

function drawBars(ctx, width, height, values) {
  const barWidth = width / values.length;
  values.forEach((value, index) => {
    const barHeight = Math.max(4, value * height * 0.92);
    const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
    gradient.addColorStop(0, "#34dcff");
    gradient.addColorStop(0.5, "#8cff77");
    gradient.addColorStop(1, "#ff6fe3");
    ctx.fillStyle = gradient;
    ctx.fillRect(index * barWidth + 1, height - barHeight, Math.max(2, barWidth - 2), barHeight);
  });
}

function heatColor(value) {
  const hue = 220 - value * 180;
  const saturation = 88;
  const lightness = 18 + value * 52;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

function drawLatent(projection) {
  const canvas = elements.latentCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#07101b";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.stroke();

  projection.points.forEach((point) => {
    const x = ((point.x + 1.4) / 2.8) * width;
    const y = height - ((point.y + 0.8) / 1.8) * height;
    ctx.fillStyle = point.color;
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
  });

  const sampleX = ((projection.sample.x + 1.4) / 2.8) * width;
  const sampleY = height - ((projection.sample.y + 0.8) / 1.8) * height;
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#ffffff";
  ctx.shadowBlur = 22;
  ctx.shadowColor = projection.sample.color;
  ctx.beginPath();
  ctx.arc(sampleX, sampleY, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillStyle = projection.sample.color;
  ctx.font = "700 14px 'Avenir Next Condensed', sans-serif";
  ctx.fillText("YOU", sampleX + 14, sampleY - 8);
}

function renderDatasetSummary(payload) {
  elements.datasetSummaryCards.innerHTML = "";
  elements.topRiskList.innerHTML = "";
  elements.skippedList.innerHTML = "";

  if (!payload) {
    elements.datasetSummaryCards.innerHTML = summaryCardMarkup("Status", "Ready");
    return;
  }

  if (payload.error) {
    elements.datasetSummaryCards.innerHTML = summaryCardMarkup("Dataset Error", payload.error);
    return;
  }

  const summary = payload.summary;
  const cards = [
    ["Audio Files", summary.total_audio_files],
    ["Processed", summary.processed_audio_files],
    ["TB Cases", summary.predicted_tb_cases],
    ["Other Cases", summary.predicted_other_cases],
    ["High Risk", summary.high_tb_risk],
    ["Moderate Risk", summary.moderate_tb_risk],
    ["Low Risk", summary.low_tb_risk],
  ];
  elements.datasetSummaryCards.innerHTML = cards.map(([label, value]) => summaryCardMarkup(label, value)).join("");

  (payload.top_tb_risk_files || []).forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${item.infection_type}</strong><br>${item.audio_file}<br>TB Prob: ${item.tb_probability} • Risk: ${item.tb_risk_level}`;
    elements.topRiskList.appendChild(li);
  });

  const skipped = payload.skipped || [];
  if (!skipped.length) {
    const li = document.createElement("li");
    li.textContent = "No skipped files.";
    elements.skippedList.appendChild(li);
  } else {
    skipped.slice(0, 10).forEach((item) => {
      const li = document.createElement("li");
      li.innerHTML = `${item.audio_file}<br>${item.error}`;
      elements.skippedList.appendChild(li);
    });
  }
}

function summaryCardMarkup(label, value) {
  return `
    <div class="summary-card">
      <span class="meta-label">${label}</span>
      <strong>${value}</strong>
    </div>
  `;
}
