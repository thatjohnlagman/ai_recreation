/**
 * IDS + Recall-Aware AFP | High-Precision SOC Dashboard Client
 * Real-time WebSocket connectivity, Leaflet threat map, Chart.js telemetry,
 * and adaptive perturbation defense controls.
 */

// Global State
let threatMap = null;
let markersMap = {};
let recallChart = null;
let socket = null;
let currentAfpEnabled = true;
let currentTheme = localStorage.getItem("app_theme") || "light";
let currentTileLayer = null;
let selectedFlowItem = null;
let currentFeedList = [];
let currentAttacksList = [];
let currentDefenseName = "AFP";
let currentDefenseMode = "Recall-Aware";
let currentDefenseIntensity = 0.0003;

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initClock();
  initMap();
  initChart();
  initWebSocket();
  initEventListeners();
});

// -----------------------------------------------------------------------------
// Theme Management (Default Light Mode, Persisted)
// -----------------------------------------------------------------------------
function initTheme() {
  applyTheme(currentTheme);
}

function applyTheme(theme) {
  currentTheme = theme;
  localStorage.setItem("app_theme", theme);
  document.documentElement.setAttribute("data-theme", theme);

  const toggleBtn = document.getElementById("theme-toggle-btn");
  if (toggleBtn) {
    const nextLabel = (theme === "light") ? "Switch to Dark Mode" : "Switch to Light Mode";
    toggleBtn.setAttribute("title", nextLabel);
    toggleBtn.setAttribute("aria-label", nextLabel);
  }

  updateMapTheme(theme);
  updateChartTheme(theme);
}

// -----------------------------------------------------------------------------
// Live Clock with Timezone Offset (e.g., Sep 18 2026, 00:50:53 UTC+8)
// -----------------------------------------------------------------------------
function initClock() {
  const clockEl = document.getElementById("live-datetime");
  if (!clockEl) return;

  const updateTime = () => {
    const now = new Date();
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const month = months[now.getMonth()];
    const day = String(now.getDate()).padStart(2, "0");
    const year = now.getFullYear();
    const hours = String(now.getHours()).padStart(2, "0");
    const minutes = String(now.getMinutes()).padStart(2, "0");
    const seconds = String(now.getSeconds()).padStart(2, "0");

    // Dynamic Timezone Offset (e.g. UTC+8, UTC-4, UTC+5:30)
    const offsetMinutes = -now.getTimezoneOffset();
    const sign = offsetMinutes >= 0 ? "+" : "-";
    const absMinutes = Math.abs(offsetMinutes);
    const tzHours = Math.floor(absMinutes / 60);
    const tzMins = absMinutes % 60;
    const tzString = tzMins > 0 ? `UTC${sign}${tzHours}:${String(tzMins).padStart(2, "0")}` : `UTC${sign}${tzHours}`;

    clockEl.textContent = `${month} ${day} ${year}, ${hours}:${minutes}:${seconds} ${tzString}`;
  };
  updateTime();
  setInterval(updateTime, 1000);
}

// -----------------------------------------------------------------------------
// Leaflet Threat Location Map
// -----------------------------------------------------------------------------
function initMap() {
  const mapContainer = document.getElementById("threat-map");
  if (!mapContainer || typeof L === "undefined") return;

  // Initialize Map centered on world view
  threatMap = L.map("threat-map", {
    center: [22, 18],
    zoom: 2,
    minZoom: 1.5,
    maxZoom: 7,
    zoomControl: true,
    attributionControl: false,
    scrollWheelZoom: true
  });

  // Reposition zoom control to bottom left matching reference screenshot
  threatMap.zoomControl.setPosition('bottomleft');

  // Load appropriate tiles for active theme
  updateMapTheme(currentTheme);
}

function updateMapTheme(theme) {
  if (!threatMap || typeof L === "undefined") return;

  if (currentTileLayer) {
    threatMap.removeLayer(currentTileLayer);
    currentTileLayer = null;
  }

  if (theme === "dark") {
    // Dark Gray Canvas tiles for Dark Mode
    currentTileLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 16,
      attribution: ""
    }).addTo(threatMap);
  } else {
    // Unwatermarked crisp OpenStreetMap tiles for Light Mode
    currentTileLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: ""
    }).addTo(threatMap);
  }
}

function updateThreatMapMarkers(locations) {
  if (!threatMap || !locations) return;

  locations.forEach(loc => {
    const markerKey = loc.ip;
    const beaconHtml = `
      <div class="radar-beacon-marker">
        <div class="beacon-ring"></div>
        <div class="beacon-core"></div>
      </div>
    `;
    const beaconIcon = L.divIcon({
      className: 'custom-beacon-div',
      html: beaconHtml,
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    });

    const tooltipContent = `
      <div class="tooltip-ip">${escapeHtml(loc.ip)}</div>
      <div class="tooltip-detail">Location: ${escapeHtml(loc.country)}</div>
      <div class="tooltip-detail">Attacks: <span class="tooltip-attacks">${escapeHtml(loc.attacks)}</span></div>
    `;

    if (markersMap[markerKey]) {
      // Update existing marker tooltip
      markersMap[markerKey].setTooltipContent(tooltipContent);
    } else {
      // Create new marker
      const marker = L.marker([loc.lat, loc.lng], { icon: beaconIcon }).addTo(threatMap);
      marker.bindTooltip(tooltipContent, {
        permanent: false,
        direction: 'top',
        className: 'custom-threat-tooltip',
        offset: [0, -10]
      });

      markersMap[markerKey] = marker;
    }
  });
}

// -----------------------------------------------------------------------------
// Chart.js: Recall vs. AFP Intensity
// -----------------------------------------------------------------------------
function initChart() {
  const ctx = document.getElementById("recallChart");
  if (!ctx || typeof Chart === "undefined") return;

  recallChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Recall",
          data: [],
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.08)",
          borderWidth: 2,
          tension: 0.35,
          pointRadius: 2.5,
          pointBackgroundColor: "#10b981",
          fill: false
        },
        {
          label: "AFP intensity",
          data: [],
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.08)",
          borderWidth: 2,
          tension: 0.35,
          pointRadius: 2.5,
          pointBackgroundColor: "#3b82f6",
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(13, 22, 41, 0.95)",
          borderColor: "#1a2a47",
          borderWidth: 1,
          titleColor: "#f1f5f9",
          bodyColor: "#94a3b8",
          padding: 8,
          boxPadding: 4,
          usePointStyle: true
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            color: "#64748b",
            font: { size: 10 }
          }
        },
        y: {
          min: 0.00,
          max: 1.00,
          ticks: {
            stepSize: 0.20,
            color: "#64748b",
            font: { size: 10 },
            callback: (val) => val.toFixed(2)
          },
          grid: {
            color: "rgba(26, 42, 71, 0.6)",
            borderDash: [3, 3]
          }
        }
      }
    }
  });

  updateChartTheme(currentTheme);
}

function updateChart(history) {
  if (!recallChart || !history) return;
  recallChart.data.labels = history.labels;
  recallChart.data.datasets[0].data = history.recall;
  recallChart.data.datasets[1].data = history.afp_intensity;
  recallChart.update();
}

// -----------------------------------------------------------------------------
// Circular SVG Gauge & Research Defense Panel
// -----------------------------------------------------------------------------
function updateAfpPanel(afp, recallVal) {
  if (!afp) return;
  currentAfpEnabled = afp.enabled;
  currentDefenseName = (afp.defense_name || "afp").toUpperCase();
  currentDefenseMode = (afp.mode === "recall-aware") ? "Recall-Aware" : "Base";
  if (typeof afp.intensity === 'number') {
    currentDefenseIntensity = afp.intensity;
  }
  const defName = currentDefenseName;
  const modeName = (afp.mode === "recall-aware") ? "RA" : "Base";
  const stateName = afp.controller_state || (afp.enabled ? "Green" : "Bypassed");

  // Header Badge & Top Status Pill
  const headerBadge = document.getElementById("afp-header-badge");
  if (headerBadge) {
    headerBadge.textContent = afp.enabled ? `${defName} Active` : "Bypassed";
    headerBadge.className = `badge-status-pill ${afp.enabled ? "green" : "red"}`;
  }

  const afpStatusText = document.getElementById("afp-status-text");
  const afpStatusDot = document.getElementById("afp-status-dot");
  if (afpStatusText && afpStatusDot) {
    afpStatusText.textContent = afp.enabled ? `${defName} (${modeName})` : "Bypassed";
    afpStatusText.className = `status-val ${afp.enabled ? "cyan" : "red"}`;
    afpStatusDot.className = `status-dot ${afp.enabled ? "cyan" : "red"}`;
  }

  // Panel Title
  const panelTitle = document.getElementById("defense-panel-title");
  if (panelTitle) {
    if (!afp.enabled || afp.defense_name === "none") {
      panelTitle.textContent = "IDS (No Defense)";
    } else {
      panelTitle.textContent = `${afp.mode === "recall-aware" ? "Recall-Aware" : "Base"} ${defName}`;
    }
  }

  // Active Buttons in Pill / Segment Groups
  const defenseBtns = document.querySelectorAll("#defense-selector-group .segment-btn, #defense-selector-group .pill-btn");
  defenseBtns.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.defense === (afp.defense_name || "afp"));
  });

  const modeBtns = document.querySelectorAll("#mode-selector-group .segment-btn, #mode-selector-group .pill-btn");
  modeBtns.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === (afp.mode || "recall-aware"));
  });

  // Controller State Pill
  const statePill = document.getElementById("controller-state-pill");
  if (statePill) {
    statePill.textContent = stateName;
    statePill.className = `ra-status-badge ${stateName.toLowerCase()}`;
  }

  // Circular Gauge Stroke & Values (Truthful zero and unavailable handling)
  const gaugeCircle = document.getElementById("gauge-circle-bar");
  const gaugeText = document.getElementById("gauge-recall-val");
  const gaugePill = document.getElementById("gauge-health-pill");

  const isRecallAvailable = (recallVal !== null && recallVal !== undefined && recallVal !== "—" && recallVal !== "");
  if (!isRecallAvailable) {
    if (gaugeCircle) gaugeCircle.style.strokeDashoffset = 264;
    if (gaugeText) gaugeText.textContent = "—";
    if (gaugePill) {
      gaugePill.textContent = "Awaiting Data";
      gaugePill.className = "gauge-badge-pill";
    }
  } else {
    const gaugeVal = Number(recallVal);
    const totalCircumference = 264;
    const offset = totalCircumference * (1.0 - Math.min(1.0, Math.max(0, gaugeVal)));
    if (gaugeCircle) gaugeCircle.style.strokeDashoffset = offset;
    if (gaugeText) gaugeText.textContent = gaugeVal.toFixed(3);
    if (gaugePill) {
      const isHealthy = gaugeVal >= (afp.threshold_warning || 0.85);
      gaugePill.textContent = isHealthy ? "Healthy" : "Alert";
      gaugePill.className = `gauge-badge-pill ${isHealthy ? "green" : "alert"}`;
    }
  }

  // Intensity Bar, Bounds & Labels
  const intensityVal = document.getElementById("afp-intensity-val");
  const intensityBar = document.getElementById("afp-bar-fill");
  const minBound = (typeof afp.intensity_min === 'number') ? afp.intensity_min : 0.0;
  const maxBound = (typeof afp.intensity_max === 'number' && afp.intensity_max > 0) ? afp.intensity_max : 0.0003;

  const paramLabel = document.getElementById("afp-param-name");
  if (paramLabel) {
    if (afp.defense_name === "rs") paramLabel.textContent = "RS Intensity (σ)";
    else if (afp.defense_name === "fs") paramLabel.textContent = "FS Rounding (d)";
    else paramLabel.textContent = "AFP Intensity (ε)";
  }

  const minBoundEl = document.getElementById("afp-min-bound-val");
  const maxBoundEl = document.getElementById("afp-max-bound-val");
  const barMinEl = document.getElementById("bar-min-label");
  const barMaxEl = document.getElementById("bar-max-label");

  if (minBoundEl) minBoundEl.textContent = (afp.defense_name === "fs") ? "0" : minBound.toFixed(4);
  if (maxBoundEl) maxBoundEl.textContent = (afp.defense_name === "fs") ? "2" : maxBound.toFixed(4);
  if (barMinEl) barMinEl.textContent = (afp.defense_name === "fs") ? "0" : minBound.toFixed(4);
  if (barMaxEl) barMaxEl.textContent = (afp.defense_name === "fs") ? "2" : maxBound.toFixed(4);

  if (intensityVal) {
    if (afp.defense_name === "fs") {
      intensityVal.textContent = Math.round(afp.intensity || 2);
    } else {
      const num = Number(afp.intensity || 0);
      if (num === 0) {
        intensityVal.textContent = "0.0000";
      } else if (num < 0.0001) {
        // Display up to 6 decimals for decayed intensities (e.g., 0.000075)
        intensityVal.textContent = num.toFixed(6);
      } else {
        // Standard calibrated intensity (e.g., 0.0003)
        intensityVal.textContent = num.toFixed(4);
      }
    }
  }

  if (intensityBar) {
    const span = Math.max(1e-9, maxBound - minBound);
    const pct = Math.max(5, Math.min(100, (((afp.intensity || 0) - minBound) / span) * 100));
    intensityBar.style.width = `${pct}%`;
  }
}

// -----------------------------------------------------------------------------
// Table Renderers
// -----------------------------------------------------------------------------
function renderTopThreats(threats) {
  const tbody = document.getElementById("top-threats-tbody");
  if (!tbody) return;
  if (!threats || threats.length === 0) {
    tbody.innerHTML = '<tr><td colspan="3" class="text-center" style="color: var(--text-muted); padding: 14px;">No threat locations in session</td></tr>';
    return;
  }

  tbody.innerHTML = threats.map(item => `
    <tr>
      <td>
        <div class="ip-cell">
          <span class="status-dot red"></span>
          <span>${escapeHtml(item.ip)}</span>
        </div>
      </td>
      <td>${escapeHtml(item.location)}</td>
      <td class="text-right">${escapeHtml(item.attacks)}</td>
    </tr>
  `).join("");
}

// -----------------------------------------------------------------------------
// Flow / Attack Forensics Inspector Logic & Heuristics
// -----------------------------------------------------------------------------
function inferLocation(ip) {
  if (!ip) return "Unknown";
  // RFC 1918 Private Subnets & Loopback
  if (
    ip.startsWith("10.") ||
    ip.startsWith("192.168.") ||
    ip.startsWith("127.") ||
    (ip.startsWith("172.") && parseInt(ip.split(".")[1], 10) >= 16 && parseInt(ip.split(".")[1], 10) <= 31)
  ) {
    return "Private Network";
  }
  // Public IP with no offline GeoIP DB
  return "Unknown";
}

function formatIntensityDecimal(val) {
  const num = Number(val || 0);
  if (num === 0) return "0.0000";
  if (num < 0.0001) return num.toFixed(6);
  return num.toFixed(4);
}

function updateInspectorUI(item) {
  if (!item) return;

  const isAttack = (item.status && (item.status.toLowerCase() === "attack" || item.status.toLowerCase() === "malicious"));
  const timeStr = item.timestamp || item.time || "Recent";
  const srcIp = item.source_ip || "192.168.1.45";
  const dstIp = item.destination_ip || "192.168.1.10:8000";

  // 1. Attack Scenario (What the attacker is doing)
  const attackScenario = item.attack_scenario || "None";
  const scenarioSub = (attackScenario && attackScenario !== "None" && attackScenario !== "Background Traffic")
    ? "Simulation Session"
    : "Background Traffic";

  // 2. Traffic Family (What the data represents)
  const trafficFamily = item.traffic_family || item.type || "Unknown";
  const trafficSource = item.traffic_family_source || (trafficFamily !== "Unknown" ? "Dataset-derived" : "Synthetic / Non-dataset");

  // 3. IDS Classification (Binary authoritative model prediction)
  const idsClassification = isAttack ? "Attack" : "Benign";

  // 4. Geolocation (Legitimate IP location or Private Network / Unknown)
  const location = (item.location && item.location !== "Unknown") ? item.location : inferLocation(srcIp);

  // Confidence calculation
  let conf = 0.93;
  if (typeof item.confidence === "number") {
    conf = item.confidence;
  } else if (item.confidence) {
    conf = parseFloat(item.confidence) || 0.93;
  } else {
    conf = isAttack ? 0.93 : 0.95;
  }

  // Active Defense & Intensity
  const defense = item.defense || currentDefenseName || "AFP";
  const mode = item.mode || currentDefenseMode || "Recall-Aware";
  const intensity = (item.intensity !== undefined) ? item.intensity : currentDefenseIntensity;

  // Mitigation Action
  const action = item.action || (isAttack ? "BLOCKED (403)" : "ALLOWED (200)");

  // Top header in inspector
  const titleEl = document.getElementById("insp-flow-title");
  if (titleEl) titleEl.textContent = `${srcIp} → ${dstIp} [${trafficFamily}]`;

  const timeEl = document.getElementById("insp-time");
  if (timeEl) timeEl.textContent = timeStr;

  const badgeEl = document.getElementById("insp-verdict-badge");
  if (badgeEl) {
    badgeEl.textContent = isAttack ? "Blocked (403)" : "Allowed (200)";
    badgeEl.className = `badge-status ${isAttack ? "malicious" : "benign"}`;
  }

  // Card 1: Source Origin
  const srcIpEl = document.getElementById("insp-source-ip");
  if (srcIpEl) srcIpEl.textContent = srcIp;

  const locEl = document.getElementById("insp-location");
  if (locEl) locEl.textContent = location;

  // Card 2: Target Endpoint
  const dstIpEl = document.getElementById("insp-dest-ip");
  if (dstIpEl) dstIpEl.textContent = dstIp;

  const protoEl = document.getElementById("insp-protocol");
  if (protoEl) protoEl.textContent = "Protected Data Service (HTTP)";

  // Card 3: Attack Scenario
  const scenarioEl = document.getElementById("insp-attack-scenario");
  if (scenarioEl) scenarioEl.textContent = attackScenario;

  const scenarioSubEl = document.getElementById("insp-scenario-sub");
  if (scenarioSubEl) scenarioSubEl.textContent = scenarioSub;

  // Card 4: Traffic Family
  const familyEl = document.getElementById("insp-traffic-family");
  if (familyEl) familyEl.textContent = trafficFamily;

  const sourceEl = document.getElementById("insp-traffic-source");
  if (sourceEl) sourceEl.textContent = trafficSource;

  // Card 5: IDS Classification
  const idsClassEl = document.getElementById("insp-ids-classification");
  if (idsClassEl) {
    idsClassEl.textContent = idsClassification;
    idsClassEl.style.color = isAttack ? "#f87171" : "#34d399";
  }

  const confValEl = document.getElementById("insp-confidence-val");
  if (confValEl) confValEl.textContent = `${(conf * 100).toFixed(1)}% Confidence`;

  const confFillEl = document.getElementById("insp-confidence-fill");
  if (confFillEl) {
    confFillEl.style.width = `${Math.min(100, Math.max(5, conf * 100))}%`;
    confFillEl.style.background = isAttack ? "#ef4444" : "#10b981";
  }

  // Card 6: Defense & Mitigation
  const defNameEl = document.getElementById("insp-defense-name");
  if (defNameEl) defNameEl.textContent = `${defense} (${mode})`;

  const defIntEl = document.getElementById("insp-defense-intensity");
  if (defIntEl) defIntEl.textContent = `ε = ${formatIntensityDecimal(intensity)}`;

  const actionNameEl = document.getElementById("insp-action-name");
  if (actionNameEl) {
    actionNameEl.textContent = action;
    actionNameEl.className = `insp-card-sub ${isAttack ? "text-red" : "text-green"}`;
  }
}

window.onRowSelectFeed = function(idx) {
  if (currentFeedList && currentFeedList[idx]) {
    selectedFlowItem = currentFeedList[idx];
    updateInspectorUI(selectedFlowItem);
    renderRecentFeed(currentFeedList);
    renderRecentAttacks(currentAttacksList);
  }
};

window.onRowSelectAttack = function(idx) {
  if (currentAttacksList && currentAttacksList[idx]) {
    selectedFlowItem = currentAttacksList[idx];
    updateInspectorUI(selectedFlowItem);
    renderRecentFeed(currentFeedList);
    renderRecentAttacks(currentAttacksList);
  }
};

// Aliases for compatibility
window.handleSelectFeed = window.onRowSelectFeed;
window.handleSelectAttack = window.onRowSelectAttack;

function resetInspectorUI() {
  selectedFlowItem = null;
  const titleEl = document.getElementById("insp-flow-title");
  if (titleEl) titleEl.textContent = "Awaiting flow telemetry";
  const timeEl = document.getElementById("insp-time");
  if (timeEl) timeEl.textContent = "—";
  const badgeEl = document.getElementById("insp-verdict-badge");
  if (badgeEl) {
    badgeEl.textContent = "Idle";
    badgeEl.className = "badge-status";
  }
  const srcIpEl = document.getElementById("insp-source-ip");
  if (srcIpEl) srcIpEl.textContent = "—";
  const locEl = document.getElementById("insp-location");
  if (locEl) locEl.textContent = "—";
  const dstIpEl = document.getElementById("insp-dest-ip");
  if (dstIpEl) dstIpEl.textContent = "—";
  const scenarioEl = document.getElementById("insp-attack-scenario");
  if (scenarioEl) scenarioEl.textContent = "—";
  const scenarioSubEl = document.getElementById("insp-scenario-sub");
  if (scenarioSubEl) scenarioSubEl.textContent = "No session active";
  const familyEl = document.getElementById("insp-traffic-family");
  if (familyEl) familyEl.textContent = "—";
  const sourceEl = document.getElementById("insp-traffic-source");
  if (sourceEl) sourceEl.textContent = "—";
  const idsClassEl = document.getElementById("insp-ids-classification");
  if (idsClassEl) {
    idsClassEl.textContent = "—";
    idsClassEl.style.color = "";
  }
  const confValEl = document.getElementById("insp-confidence-val");
  if (confValEl) confValEl.textContent = "—";
  const confFillEl = document.getElementById("insp-confidence-fill");
  if (confFillEl) {
    confFillEl.style.width = "0%";
    confFillEl.style.background = "";
  }
  const defNameEl = document.getElementById("insp-defense-name");
  if (defNameEl) defNameEl.textContent = "—";
  const defIntEl = document.getElementById("insp-defense-intensity");
  if (defIntEl) defIntEl.textContent = "—";
  const actionNameEl = document.getElementById("insp-action-name");
  if (actionNameEl) {
    actionNameEl.textContent = "—";
    actionNameEl.className = "insp-card-sub";
  }
}

function renderRecentFeed(feed) {
  const tbody = document.getElementById("recent-feed-tbody");
  if (!tbody) return;
  if (!feed || feed.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-center" style="color: var(--text-muted); padding: 14px;">No traffic flows recorded in session</td></tr>';
    currentFeedList = [];
    resetInspectorUI();
    return;
  }
  currentFeedList = feed;

  // Auto-select first attack (or top item) if none selected yet
  if (!selectedFlowItem && feed.length > 0) {
    const firstAttack = feed.find(item => item.status && (item.status.toLowerCase() === "attack" || item.status.toLowerCase() === "malicious"));
    selectedFlowItem = firstAttack || feed[0];
    if (selectedFlowItem) {
      updateInspectorUI(selectedFlowItem);
    }
  }

  const selectedTime = selectedFlowItem ? (selectedFlowItem.timestamp || selectedFlowItem.time) : null;

  tbody.innerHTML = feed.map((item, idx) => {
    const isAttack = (item.status && (item.status.toLowerCase() === "attack" || item.status.toLowerCase() === "malicious"));
    const badgeClass = item.is_query ? "neutral" : (isAttack ? "malicious" : "benign");
    const statusLabel = item.is_query ? "Query" : (isAttack ? "Attack" : "Benign");
    const familyLabel = item.traffic_family || item.type || "Unknown";
    const itemTime = item.timestamp || item.time || "";
    const isSelected = selectedFlowItem && (
      selectedFlowItem.source_ip === item.source_ip &&
      selectedTime === itemTime
    );
    const selectedClass = isSelected ? "selected-row" : "";
    const confVal = typeof item.confidence === 'number' ? item.confidence.toFixed(2) : (item.confidence || "—");
    return `
      <tr class="${selectedClass}" data-idx="${idx}" onclick="onRowSelectFeed(${idx})" title="Click to view detailed flow forensics">
        <td>${escapeHtml(itemTime)}</td>
        <td>${escapeHtml(item.source_ip)}</td>
        <td>${escapeHtml(item.destination_ip || "192.168.1.10")}</td>
        <td>${escapeHtml(familyLabel)}</td>
        <td>${escapeHtml(confVal)}</td>
        <td><span class="badge-status ${badgeClass}">${escapeHtml(statusLabel)}</span></td>
      </tr>
    `;
  }).join("");
}

function renderRecentAttacks(attacks) {
  const tbody = document.getElementById("recent-attacks-tbody");
  if (!tbody) return;
  if (!attacks || attacks.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="text-center" style="color: var(--text-muted); padding: 14px;">No attack flows recorded in session</td></tr>';
    currentAttacksList = [];
    return;
  }
  currentAttacksList = attacks;

  // Auto-select if still null
  if (!selectedFlowItem && attacks.length > 0) {
    selectedFlowItem = attacks[0];
    updateInspectorUI(selectedFlowItem);
  }

  const selectedTime = selectedFlowItem ? (selectedFlowItem.timestamp || selectedFlowItem.time) : null;

  tbody.innerHTML = attacks.map((item, idx) => {
    const isAttack = (item.status && (item.status.toLowerCase() === "attack" || item.status.toLowerCase() === "malicious"));
    const badgeClass = isAttack ? "malicious" : "benign";
    const statusLabel = isAttack ? "Attack" : "Benign";
    const familyLabel = item.traffic_family || item.type || "Unknown";
    const locationLabel = (item.location && item.location !== "Unknown") ? item.location : inferLocation(item.source_ip);
    const itemTime = item.timestamp || item.time || "";
    const isSelected = selectedFlowItem && (
      selectedFlowItem.source_ip === item.source_ip &&
      selectedTime === itemTime
    );
    const selectedClass = isSelected ? "selected-row" : "";
    return `
      <tr class="${selectedClass}" data-idx="${idx}" onclick="onRowSelectAttack(${idx})" title="Click to view detailed attack forensics">
        <td>${escapeHtml(itemTime)}</td>
        <td>${escapeHtml(item.source_ip)}</td>
        <td>${escapeHtml(locationLabel)}</td>
        <td>${escapeHtml(familyLabel)}</td>
        <td><span class="badge-status ${badgeClass}">${escapeHtml(statusLabel)}</span></td>
      </tr>
    `;
  }).join("");
}

// -----------------------------------------------------------------------------
// Full Dashboard State Sync
// -----------------------------------------------------------------------------
function syncDashboard(payload) {
  if (!payload) return;

  // 0. Active Attack Scenario Simulation Badge
  if (payload.simulation) {
    const scText = document.getElementById("scenario-status-text");
    const scDot = document.getElementById("scenario-status-dot");
    if (scText && scDot) {
      const isAct = payload.simulation.is_active && payload.simulation.active_scenario !== "None";
      scText.textContent = isAct ? payload.simulation.active_scenario : "None";
      scText.className = `status-val ${isAct ? "orange" : "gray"}`;
      scDot.className = `status-dot ${isAct ? "orange" : "gray"}`;
    }
  }

  // 0b. Data Profile & Pool Badge
  if (payload.data_profile) {
    const dsText = document.getElementById("dataset-status-text");
    const dsDot = document.getElementById("dataset-status-dot");
    if (dsText && dsDot) {
      if (payload.data_profile === "expanded") {
        const qCount = (payload.stats && payload.stats.query_count) ? ` | ${payload.stats.query_count}q` : "";
        dsText.textContent = `Expanded (72k)${qCount}`;
        dsText.className = "status-val blue";
        dsDot.className = "status-dot blue";
      } else {
        dsText.textContent = "Fixture20 (20)";
        dsText.className = "status-val gray";
        dsDot.className = "status-dot gray";
      }
    }
  }

  // 1. KPI Cards
  if (payload.stats) {
    document.getElementById("kpi-total-traffic").textContent = payload.stats.total_traffic;
    document.getElementById("kpi-detected-attacks").textContent = payload.stats.detected_attacks;
    document.getElementById("kpi-recall").textContent = payload.stats.detection_recall;
    document.getElementById("kpi-fpr").textContent = payload.stats.false_positive_rate;
  }

  // 2. AFP Control Panel
  if (payload.afp) {
    updateAfpPanel(payload.afp, payload.stats ? payload.stats.detection_recall : null);
  }

  // 3. Threat Map
  if (payload.threat_locations) {
    updateThreatMapMarkers(payload.threat_locations);
  }

  // 4. Tables
  if (payload.top_threat_ips) {
    renderTopThreats(payload.top_threat_ips);
  }
  if (payload.recent_feed) {
    renderRecentFeed(payload.recent_feed);
  }
  if (payload.recent_attacks) {
    renderRecentAttacks(payload.recent_attacks);
  }

  // 5. Chart
  if (payload.history) {
    updateChart(payload.history);
  }
}

// -----------------------------------------------------------------------------
// Real-time WebSocket Feed
// -----------------------------------------------------------------------------
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  function connect() {
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log("[SOC Dashboard] WebSocket connected to backend telemetry stream.");
    };

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.payload) {
          syncDashboard(msg.payload);
        }
      } catch (err) {
        console.error("Error parsing WebSocket message:", err);
      }
    };

    socket.onclose = () => {
      console.warn("[SOC Dashboard] WebSocket disconnected. Reconnecting in 2s...");
      setTimeout(connect, 2000);
    };

    socket.onerror = (err) => {
      console.error("[SOC Dashboard] WebSocket error:", err);
    };
  }

  connect();

  // Fallback initial REST fetch
  fetch("/api/dashboard/stats")
    .then(r => r.json())
    .then(data => syncDashboard(data))
    .catch(err => console.warn("Initial REST fetch error:", err));
}

function updateChartTheme(theme) {
  if (!recallChart) return;
  const isDark = (theme === "dark");
  const gridColor = isDark ? "rgba(26, 42, 71, 0.6)" : "rgba(226, 232, 240, 0.9)";
  const textColor = isDark ? "#94a3b8" : "#64748b";
  const tooltipBg = isDark ? "rgba(13, 22, 41, 0.95)" : "rgba(255, 255, 255, 0.98)";
  const tooltipBorder = isDark ? "#1a2a47" : "#cbd5e1";
  const tooltipTitle = isDark ? "#f1f5f9" : "#0f172a";
  const tooltipBody = isDark ? "#94a3b8" : "#475569";

  recallChart.options.scales.y.grid.color = gridColor;
  recallChart.options.scales.x.ticks.color = textColor;
  recallChart.options.scales.y.ticks.color = textColor;
  recallChart.options.plugins.tooltip.backgroundColor = tooltipBg;
  recallChart.options.plugins.tooltip.borderColor = tooltipBorder;
  recallChart.options.plugins.tooltip.titleColor = tooltipTitle;
  recallChart.options.plugins.tooltip.bodyColor = tooltipBody;
  recallChart.update();
}

// -----------------------------------------------------------------------------
// Interactive Controls & Event Listeners
// -----------------------------------------------------------------------------
function initEventListeners() {
  // Theme Toggle Button in Header
  const themeToggleBtn = document.getElementById("theme-toggle-btn");
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      const nextTheme = (currentTheme === "light") ? "dark" : "light";
      applyTheme(nextTheme);
    });
  }

  // Toggle AFP Defense Button in Sidebar
  const toggleBtn = document.getElementById("toggle-afp-btn");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", async () => {
      const nextState = !currentAfpEnabled;
      try {
        const res = await fetch("/api/dashboard/toggle-afp", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: nextState })
        });
        const data = await res.json();
        console.log("AFP Defense toggled:", data);
      } catch (err) {
        console.error("Error toggling AFP:", err);
      }
    });
  }

  // Defense Selector Buttons (AFP, RS, FS, None)
  const defenseBtns = document.querySelectorAll("#defense-selector-group .pill-btn");
  defenseBtns.forEach(btn => {
    btn.addEventListener("click", async () => {
      const selectedDef = btn.dataset.defense;
      try {
        const res = await fetch("/api/dashboard/set-defense", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ defense: selectedDef })
        });
        const data = await res.json();
        console.log("Defense changed:", data);
      } catch (err) {
        console.error("Error switching defense:", err);
      }
    });
  });

  // Mode Selector Buttons (Recall-Aware, Base)
  const modeBtns = document.querySelectorAll("#mode-selector-group .pill-btn");
  modeBtns.forEach(btn => {
    btn.addEventListener("click", async () => {
      const selectedMode = btn.dataset.mode;
      try {
        const res = await fetch("/api/dashboard/set-mode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: selectedMode })
        });
        const data = await res.json();
        console.log("Mode changed:", data);
      } catch (err) {
        console.error("Error switching mode:", err);
      }
    });
  });
}
