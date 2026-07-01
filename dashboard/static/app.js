const state = {
  liveLoaded: false,
  analyticsLoaded: false,
  modelLoaded: false,
  currentFrameCounter: -1,
  livePollingHandle: null,
  streamUrl: "",
};

const elements = {
  modeSelect: document.getElementById("mode-select"),
  confidenceRange: document.getElementById("confidence-range"),
  confidenceValue: document.getElementById("confidence-value"),
  cameraInput: document.getElementById("camera-input"),
  startButton: document.getElementById("start-button"),
  stopButton: document.getElementById("stop-button"),
  resetButton: document.getElementById("reset-button"),
  stream: document.getElementById("live-stream"),
  videoPlaceholder: document.getElementById("video-placeholder"),
  engineStatus: document.getElementById("engine-status"),
  engineMessage: document.getElementById("engine-message"),
  lastAlert: document.getElementById("last-alert"),
  footerStatus: document.getElementById("footer-status"),
  footerMode: document.getElementById("footer-mode"),
  footerConfidence: document.getElementById("footer-confidence"),
  analyticsSearch: document.getElementById("analytics-search"),
  analyticsRefresh: document.getElementById("analytics-refresh"),
  benchmarkButton: document.getElementById("benchmark-button"),
  benchmarkLoader: document.getElementById("benchmark-loader"),
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function setPlaceholder(title, message) {
  elements.videoPlaceholder.innerHTML = "";
  const strong = document.createElement("strong");
  strong.textContent = title;
  const span = document.createElement("span");
  span.textContent = message;
  elements.videoPlaceholder.append(strong, span);
}

function formatNumber(value, fallback = "0") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return fallback;
  }
  return `${value}`;
}

function formatDuration(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "0.0s";
  }
  return `${Number(value).toFixed(1)}s`;
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) {
    node.textContent = value;
  }
}

function renderLiveState(payload) {
  const controls = payload.controls || {};
  const stats = payload.stats || {};
  const running = Boolean(payload.running);

  state.currentFrameCounter = payload.frame_counter ?? state.currentFrameCounter;
  elements.modeSelect.value = controls.mode || elements.modeSelect.value;
  elements.confidenceRange.value = controls.confidence ?? elements.confidenceRange.value;
  elements.confidenceValue.textContent = Number(elements.confidenceRange.value).toFixed(2);
  elements.cameraInput.value = controls.camera ?? elements.cameraInput.value;

  setText("stat-fps", Number(stats.fps || 0).toFixed(1));
  setText("stat-alerts", formatNumber(stats.alerts, "0"));
  setText("stat-duration", formatDuration(stats.duration));
  setText("stat-detections", formatNumber(stats.detections, "0"));
  setText("stat-usage-events", formatNumber(stats.usage_events, "0"));

  const error = stats.error;
  const lastAlert = stats.last_alert;
  const statusLabel = error ? "Error" : running ? "Running" : "Stopped";
  const message = error ? error : running ? "Inference loop active." : "Idle.";

  elements.engineStatus.textContent = statusLabel;
  elements.engineMessage.textContent = message;
  elements.footerStatus.textContent = message;
  elements.footerMode.textContent = `Mode: ${(controls.mode || "meme").replace(/^./, (char) => char.toUpperCase())}`;
  elements.footerConfidence.textContent = `Confidence: ${Number(controls.confidence || 0.5).toFixed(2)}`;

  if (lastAlert) {
    const mode = lastAlert.mode || "unknown";
    const timestamp = lastAlert.timestamp || "unknown time";
    elements.lastAlert.textContent = `${mode} at ${timestamp}`;
  } else {
    elements.lastAlert.textContent = "No alerts yet.";
  }

  if (running) {
    if (!state.streamUrl) {
      state.streamUrl = `/api/live/stream?t=${Date.now()}`;
      elements.stream.src = state.streamUrl;
      elements.stream.hidden = false;
      elements.videoPlaceholder.hidden = true;
    }
  } else {
    if (error) {
      setPlaceholder("Detection stopped.", error);
    } else {
      setPlaceholder("Stand by.", "Start detection to open the live camera feed.");
    }
    elements.videoPlaceholder.hidden = false;
    if (state.streamUrl) {
      elements.stream.removeAttribute("src");
      elements.stream.hidden = true;
      state.streamUrl = "";
    }
  }
}

async function refreshLiveState() {
  try {
    const payload = await requestJson("/api/live/state");
    renderLiveState(payload);
  } catch (error) {
    elements.engineStatus.textContent = "Error";
    elements.engineMessage.textContent = error.message;
    elements.footerStatus.textContent = error.message;
  }
}

function startLivePolling() {
  if (state.livePollingHandle !== null) {
    return;
  }
  refreshLiveState();
  state.livePollingHandle = window.setInterval(() => {
    refreshLiveState();
  }, 750);
}

function stopLivePolling() {
  if (state.livePollingHandle === null) {
    return;
  }
  window.clearInterval(state.livePollingHandle);
  state.livePollingHandle = null;
}

async function applyControls() {
  const payload = {
    mode: elements.modeSelect.value,
    confidence: Number(elements.confidenceRange.value),
    camera: Number(elements.cameraInput.value || 0),
  };
  const response = await requestJson("/api/live/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  renderLiveState(response);
}

async function stopDetection() {
  const response = await requestJson("/api/live/stop", { method: "POST" });
  renderLiveState(response);
}

async function resetDetection() {
  const response = await requestJson("/api/live/reset", { method: "POST" });
  renderLiveState(response);
}

function createRow(cells, { empty = false } = {}) {
  const tr = document.createElement("tr");
  if (empty) {
    tr.classList.add("highlighted");
  }
  cells.forEach((value) => {
    const td = document.createElement("td");
    td.textContent = value;
    tr.appendChild(td);
  });
  return tr;
}

function renderTableBody(tableId, rows, mapRow, emptyMessage) {
  const body = document.querySelector(`#${tableId} tbody`);
  body.innerHTML = "";

  if (!rows || rows.length === 0) {
    body.appendChild(createRow([emptyMessage, "", "", "", ""].slice(0, body.parentElement.tHead.rows[0].cells.length), { empty: true }));
    return;
  }

  rows.forEach((row) => {
    body.appendChild(createRow(mapRow(row)));
  });
}

async function loadAnalytics(force = false) {
  if (state.analyticsLoaded && !force) {
    return;
  }

  const query = encodeURIComponent(elements.analyticsSearch.value.trim());
  const payload = await requestJson(`/api/analytics?q=${query}`);

  setText("analytics-alerts-today", formatNumber(payload.summary.total_alerts_today, "0"));
  setText("analytics-most-active-hour", payload.summary.most_active_hour || "No data yet");
  setText("analytics-average-duration", payload.summary.avg_alert_duration || "No data yet");

  renderTableBody("alerts-by-hour-table", payload.alerts_by_hour, (row) => [row.hour, `${row.alerts}`], "No alert data yet.");
  renderTableBody("alerts-by-person-table", payload.alerts_by_person, (row) => [row.person_id, `${row.alerts}`], "No person data yet.");
  renderTableBody("usage-breakdown-table", payload.usage_breakdown, (row) => [row.status, `${row.count}`], "No usage data yet.");
  renderTableBody(
    "events-table",
    payload.events,
    (row) => [row.time, row.person_id, row.confidence || "-", row.mode || "-", row.duration || "-"],
    "No events logged yet."
  );

  state.analyticsLoaded = true;
}

async function loadModel(force = false) {
  if (state.modelLoaded && !force) {
    return;
  }

  const payload = await requestJson("/api/model");
  const architecture = payload.architecture || {};
  const training = payload.training || {};
  const dataset = payload.dataset || {};

  setText("model-name", architecture.name || "No data yet");
  setText("model-parameters", architecture.parameters || "No data yet");
  setText("model-layers", architecture.layers || "No data yet");
  setText("training-map50", training.final_map50 || "No data yet");
  setText("training-best-epoch", training.best_epoch || "No data yet");
  setText("training-time", training.training_time || "No data yet");
  setText("dataset-total-images", formatNumber(dataset.total_images, "No data yet"));

  renderTableBody("dataset-table", dataset.class_distribution || [], (row) => [row.class_name, `${row.count}`], "No class distribution available.");
  state.modelLoaded = true;
}

async function runBenchmark() {
  elements.benchmarkLoader.hidden = false;
  elements.benchmarkLoader.classList.add("animate");
  elements.benchmarkButton.disabled = true;
  try {
    const payload = await requestJson("/api/benchmark", { method: "POST" });
    renderTableBody(
      "benchmark-table",
      payload.rows || [],
      (row) => [row.device, row.average_fps, row.frames, row.elapsed_seconds, row.status],
      "No benchmark run yet."
    );
  } finally {
    elements.benchmarkButton.disabled = false;
    elements.benchmarkLoader.hidden = true;
    elements.benchmarkLoader.classList.remove("animate");
  }
}

function activateTab(tab) {
  const tablist = tab.closest('[role="tablist"]');
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  const panelIds = tabs.map((item) => item.getAttribute("aria-controls"));

  tabs.forEach((item) => {
    item.setAttribute("aria-selected", item === tab ? "true" : "false");
  });

  panelIds.forEach((id) => {
    const panel = document.getElementById(id);
    if (panel) {
      panel.hidden = id !== tab.getAttribute("aria-controls");
    }
  });

  const target = tab.getAttribute("aria-controls");
  if (target === "tab-analytics") {
    loadAnalytics();
  }
  if (target === "tab-model") {
    loadModel();
  }
}

function bindTabs() {
  document.querySelectorAll('[role="tab"]').forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab));
  });
}

async function bootstrap() {
  const payload = await requestJson("/api/bootstrap");
  const defaults = payload.defaults || {};
  elements.modeSelect.value = defaults.mode || "meme";
  elements.confidenceRange.value = defaults.confidence ?? 0.5;
  elements.confidenceValue.textContent = Number(elements.confidenceRange.value).toFixed(2);
  elements.cameraInput.value = defaults.camera ?? 0;
  renderLiveState(payload.live || payload);
}

function bindEvents() {
  elements.confidenceRange.addEventListener("input", () => {
    elements.confidenceValue.textContent = Number(elements.confidenceRange.value).toFixed(2);
    elements.footerConfidence.textContent = `Confidence: ${Number(elements.confidenceRange.value).toFixed(2)}`;
  });

  elements.modeSelect.addEventListener("change", () => {
    elements.footerMode.textContent = `Mode: ${elements.modeSelect.value.replace(/^./, (char) => char.toUpperCase())}`;
  });

  elements.startButton.addEventListener("click", async () => {
    await applyControls();
  });

  elements.stopButton.addEventListener("click", async () => {
    await stopDetection();
  });

  elements.resetButton.addEventListener("click", async () => {
    await resetDetection();
  });

  elements.analyticsRefresh.addEventListener("click", async () => {
    await loadAnalytics(true);
  });

  elements.analyticsSearch.addEventListener("keydown", async (event) => {
    if (event.key === "Enter") {
      await loadAnalytics(true);
    }
  });

  elements.benchmarkButton.addEventListener("click", async () => {
    await runBenchmark();
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopLivePolling();
    } else {
      startLivePolling();
    }
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  bindTabs();
  bindEvents();
  await bootstrap();
  startLivePolling();
});
