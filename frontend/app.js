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


// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Purge any legacy operator tokens from storage
  sessionStorage.removeItem("ids_operator_token");
  localStorage.removeItem("ids_operator_token");
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
let threatGeoJsonLayer = null;

function initMap() {
  const mapContainer = document.getElementById("threat-map");
  if (!mapContainer || typeof L === "undefined") return;

  // Initialize Map centered on world view
  threatMap = L.map("threat-map", {
    center: [20, 0],
    zoom: 2,
    minZoom: 1.5,
    maxZoom: 10,
    zoomControl: true,
    attributionControl: true,
    scrollWheelZoom: false
  });

  threatMap.zoomControl.setPosition('bottomleft');

  // 1. Bundled Offline World Outline Basemap (Natural Earth 110m GeoJSON)
  fetch("/static/vendor/ne_110m_admin_0_countries.geojson")
    .then(r => r.json())
    .then(geo => {
      threatGeoJsonLayer = L.geoJSON(geo, {
        style: function() {
          const isDark = (currentTheme === "dark");
          return {
            fillColor: isDark ? '#1e293b' : '#e2e8f0',
            weight: 1,
            opacity: 0.9,
            color: isDark ? '#334155' : '#94a3b8',
            fillOpacity: isDark ? 0.75 : 0.5
          };
        }
      }).addTo(threatMap);
    })
    .catch(err => {
      console.warn("[Map] Offline vector basemap note:", err);
    });

  // 2. OpenStreetMap online tile layer with attribution
  updateMapTheme(currentTheme);

  // Handle window resize to avoid tile seams
  window.addEventListener("resize", () => {
    if (threatMap) threatMap.invalidateSize();
  });
}

function updateMapTheme(theme) {
  if (!threatMap || typeof L === "undefined") return;

  if (currentTileLayer) {
    threatMap.removeLayer(currentTileLayer);
    currentTileLayer = null;
  }

  // Use crisp OpenStreetMap tiles with attribution
  currentTileLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>'
  }).addTo(threatMap);

  if (threatGeoJsonLayer) {
    const isDark = (theme === "dark");
    threatGeoJsonLayer.setStyle({
      fillColor: isDark ? '#1e293b' : '#e2e8f0',
      color: isDark ? '#334155' : '#94a3b8'
    });
  }
}

function updateThreatMapMarkers(locations) {
  if (!threatMap) return;
  const statusEl = document.getElementById("map-status-text");

  // Handle reset or empty locations
  if (!locations || locations.length === 0) {
    Object.values(markersMap).forEach(m => threatMap.removeLayer(m));
    markersMap = {};
    if (statusEl) statusEl.textContent = "No geolocated detections yet";
    return;
  }

  if (statusEl) {
    statusEl.textContent = `${locations.length} geolocated source IP(s) detected`;
  }

  // Clean up removed markers
  const activeIps = new Set(locations.map(l => l.ip));
  for (const ip of Object.keys(markersMap)) {
    if (!activeIps.has(ip)) {
      threatMap.removeLayer(markersMap[ip]);
      delete markersMap[ip];
    }
  }

  locations.forEach(loc => {
    const lat = parseFloat(loc.lat);
    const lng = parseFloat(loc.lng);
    if (isNaN(lat) || isNaN(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
      return;
    }

    const markerKey = loc.ip;
    const count = loc.attacks || loc.count || 1;
    const locText = loc.city ? `${loc.city}, ${loc.country}` : (loc.country || "Unknown");

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
      <div class="tooltip-ip font-mono" style="font-weight:700; color:#ef4444;">${escapeHtml(loc.ip)}</div>
      <div class="tooltip-detail">Location: ${escapeHtml(locText)}</div>
      <div class="tooltip-detail">Detected requests: <strong>${escapeHtml(count)}</strong></div>
    `;

    if (markersMap[markerKey]) {
      markersMap[markerKey].setTooltipContent(tooltipContent);
    } else {
      const marker = L.marker([lat, lng], { icon: beaconIcon }).addTo(threatMap);
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
          fill: false,
          yAxisID: 'y'
        },
        {
          label: "Defense Intensity",
          data: [],
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.08)",
          borderWidth: 2,
          tension: 0.35,
          pointRadius: 2.5,
          pointBackgroundColor: "#3b82f6",
          fill: false,
          yAxisID: 'y1'
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
          type: 'linear',
          display: true,
          position: 'left',
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
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          grid: {
            drawOnChartArea: false, // only want the grid lines for one axis to show up
          },
          ticks: {
            color: "#64748b",
            font: { size: 10 },
            callback: function(val) {
                // Return high precision if it's a tiny number (like max 0.0003 for AFP)
                // ChartJS automatically bounds min/max based on data
                return Number(val).toPrecision(2);
            }
          }
        }
      }
    }
  });

  updateChartTheme(currentTheme);
}

function updateChart(history, defenseName) {
  if (!recallChart || !history) return;
  recallChart.data.labels = history.labels;
  recallChart.data.datasets[0].data = history.recall;
  recallChart.data.datasets[1].data = history.afp_intensity;
  if (defenseName) {
      recallChart.data.datasets[1].label = `${defenseName.toUpperCase()} Intensity`;
  }
  recallChart.update();
}

// -----------------------------------------------------------------------------
// Circular SVG Gauge & Research Defense Panel
// -----------------------------------------------------------------------------
function updateAfpPanel(afp) {
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
    let text = stateName;
    if (afp.mode === 'recall-aware' && afp.rolling_recall !== undefined && afp.rolling_recall !== null) {
      text += ` (RR: ${Number(afp.rolling_recall).toFixed(3)})`;
    }
    statePill.textContent = text;
    statePill.className = `ra-status-badge ${stateName.toLowerCase()}`;
  }

  // Circular Gauge Stroke & Values (Truthful zero and unavailable handling)
  const gaugeCircle = document.getElementById("gauge-circle-bar");
  const gaugeText = document.getElementById("gauge-recall-val");
  const gaugePill = document.getElementById("gauge-health-pill");

  const isRecallAvailable = (afp.mode === 'recall-aware' && afp.rolling_recall !== null && afp.rolling_recall !== undefined && afp.window_batch_count > 0);
  const gaugeContext = document.getElementById("gauge-window-context");

  if (!isRecallAvailable) {
    if (gaugeCircle) gaugeCircle.style.strokeDashoffset = 264;
    if (gaugeText) gaugeText.textContent = "—";
    if (gaugePill) {
      gaugePill.textContent = "Awaiting feedback";
      gaugePill.className = "gauge-badge-pill";
    }
    if (gaugeContext) gaugeContext.textContent = afp.mode !== 'recall-aware' ? "Base mode active" : "Awaiting attack feedback";
  } else {
    const gaugeVal = Number(afp.rolling_recall);
    const totalCircumference = 264;
    const offset = totalCircumference * (1.0 - Math.min(1.0, Math.max(0, gaugeVal)));
    
    if (gaugeCircle) gaugeCircle.style.strokeDashoffset = offset;
    if (gaugeText) gaugeText.textContent = gaugeVal.toFixed(3);
    
    if (gaugePill) {
      let healthText = "Green";
      let healthClass = "green";
      
      if (gaugeVal < 0.85) {
        healthText = "Red";
        healthClass = "alert";
      } else if (gaugeVal < 0.95) {
        healthText = "Yellow";
        healthClass = "yellow";
      }
      
      gaugePill.textContent = healthText;
      gaugePill.className = `gauge-badge-pill ${healthClass}`;
    }
    
    if (gaugeContext) {
      gaugeContext.textContent = `Last ${afp.window_batch_count} completed attack batches`;
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
  const timeStr = item.timestamp || item.time || "—";
  const srcIp = item.source_ip || "—";
  const dstIp = item.destination_ip || "—";

  // IDS Classification (Authoritative binary classification: Benign vs Attack)
  const idsClassification = isAttack ? "Attack" : "Benign";

  // Geolocation: truthful location or Private Network / Unknown
  const location = (item.location && item.location !== "Unknown") ? item.location : (srcIp !== "—" ? inferLocation(srcIp) : "—");

  // Confidence calculation without invented defaults (0.93 / 0.95 removed; real zeros preserved)
  let conf = null;
  if (typeof item.confidence === "number" && !isNaN(item.confidence)) {
    conf = item.confidence;
  } else if (item.confidence !== undefined && item.confidence !== null && item.confidence !== "" && !isNaN(parseFloat(item.confidence))) {
    conf = parseFloat(item.confidence);
  }

  // Active Defense & Intensity
  const defense = item.defense || currentDefenseName || "—";
  const mode = item.mode || currentDefenseMode || "—";
  const intensity = (item.intensity !== undefined) ? item.intensity : currentDefenseIntensity;

  // Mitigation Action
  const action = item.action || (isAttack ? "Request rejected (HTTP 403)" : "Request allowed (HTTP 200)");

  // Top header in inspector
  const titleEl = document.getElementById("insp-flow-title");
  if (titleEl) titleEl.textContent = (srcIp !== "—" && dstIp !== "—") ? `${srcIp} → ${dstIp}` : "Selected Flow";

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

  // Card 3: IDS Classification & Confidence
  const idsClassEl = document.getElementById("insp-ids-classification");
  if (idsClassEl) {
    idsClassEl.textContent = idsClassification + (item.is_query ? " [Query]" : "");
    idsClassEl.style.color = isAttack ? "#f87171" : "#34d399";
  }

  const confValEl = document.getElementById("insp-confidence-val");
  const confFillEl = document.getElementById("insp-confidence-fill");
  if (conf !== null) {
    if (confValEl) confValEl.textContent = `${(conf * 100).toFixed(1)}% Confidence`;
    if (confFillEl) {
      confFillEl.style.width = `${Math.min(100, Math.max(0, conf * 100))}%`;
      confFillEl.style.background = isAttack ? "#ef4444" : "#10b981";
    }
  } else {
    if (confValEl) confValEl.textContent = "Unavailable";
    if (confFillEl) {
      confFillEl.style.width = "0%";
      confFillEl.style.background = "";
    }
  }

  // Card 4: Defense & Action
  const defNameEl = document.getElementById("insp-defense-name");
  if (defNameEl) defNameEl.textContent = `${defense} (${mode})`;

  const defIntEl = document.getElementById("insp-defense-intensity");
  if (defIntEl) defIntEl.textContent = (intensity !== undefined && intensity !== null) ? `ε = ${formatIntensityDecimal(intensity)}` : "—";

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

function renderTopThreats(threats) {
  const tbody = document.getElementById("top-threats-tbody");
  if (!tbody) return;
  if (!threats || threats.length === 0) {
    tbody.innerHTML = '<tr><td colspan="3" class="text-center" style="color: var(--text-muted); padding: 14px;">No threat locations in session</td></tr>';
    return;
  }
  tbody.innerHTML = threats.map(item => {
    const attacks = item.attacks || item.count || 1;
    return `
      <tr>
        <td class="font-mono">${escapeHtml(item.ip || "—")}</td>
        <td>${escapeHtml(item.location || "Unknown")}</td>
        <td class="text-right font-bold" style="color:#ef4444;">${escapeHtml(attacks)}</td>
      </tr>
    `;
  }).join("");
}

function renderRecentFeed(feed) {
  const tbody = document.getElementById("recent-feed-tbody");
  if (!tbody) return;
  if (!feed || feed.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="text-center" style="color: var(--text-muted); padding: 14px;">No traffic flows recorded in session</td></tr>';
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
    const badgeClass = isAttack ? "malicious" : "benign";
    const statusLabel = isAttack ? "Attack" : "Benign";
    const queryBadge = item.is_query ? ' <span class="badge-status neutral" style="font-size: 10px; margin-left: 4px;">Query</span>' : '';
    const itemTime = item.timestamp || item.time || "";
    const isSelected = selectedFlowItem && (
      selectedFlowItem.source_ip === item.source_ip &&
      selectedTime === itemTime
    );
    const selectedClass = isSelected ? "selected-row" : "";
    let confVal = "—";
    if (typeof item.confidence === 'number' && !isNaN(item.confidence)) {
      confVal = item.confidence.toFixed(2);
    } else if (item.confidence !== undefined && item.confidence !== null && item.confidence !== "" && !isNaN(parseFloat(item.confidence))) {
      confVal = parseFloat(item.confidence).toFixed(2);
    }
    return `
      <tr class="${selectedClass}" data-idx="${idx}" onclick="onRowSelectFeed(${idx})" title="Click to view detailed flow forensics">
        <td>${escapeHtml(itemTime)}</td>
        <td>${escapeHtml(item.source_ip || "—")}</td>
        <td>${escapeHtml(item.destination_ip || "—")}</td>
        <td>${escapeHtml(confVal)}</td>
        <td><span class="badge-status ${badgeClass}">${escapeHtml(statusLabel)}</span>${queryBadge}</td>
      </tr>
    `;
  }).join("");
}

function renderRecentAttacks(attacks) {
  const tbody = document.getElementById("recent-attacks-tbody");
  if (!tbody) return;
  if (!attacks || attacks.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="text-center" style="color: var(--text-muted); padding: 14px;">No attack flows recorded in session</td></tr>';
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
    const locationLabel = (item.location && item.location !== "Unknown") ? item.location : (item.source_ip ? inferLocation(item.source_ip) : "—");
    const itemTime = item.timestamp || item.time || "";
    const isSelected = selectedFlowItem && (
      selectedFlowItem.source_ip === item.source_ip &&
      selectedTime === itemTime
    );
    const selectedClass = isSelected ? "selected-row" : "";
    return `
      <tr class="${selectedClass}" data-idx="${idx}" onclick="onRowSelectAttack(${idx})" title="Click to view detailed attack forensics">
        <td>${escapeHtml(itemTime)}</td>
        <td>${escapeHtml(item.source_ip || "—")}</td>
        <td>${escapeHtml(locationLabel)}</td>
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

  // 0. Data Profile & Pool Badge
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
    let p = 0, r = 0, f1 = 0;
    const tp = payload.stats.tp || 0;
    const fp = payload.stats.fp || 0;
    const fn = payload.stats.fn || 0;
    if (tp + fp > 0) p = tp / (tp + fp);
    if (tp + fn > 0) r = tp / (tp + fn);
    if (p + r > 0) f1 = 2 * p * r / (p + r);
    document.getElementById("kpi-prf").textContent = `${p.toFixed(2)} / ${r.toFixed(2)} / ${f1.toFixed(2)}`;
  }

  // 2. AFP Control Panel
  if (payload.afp) {
    updateAfpPanel(payload.afp);
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
    updateChart(payload.history, payload.state ? payload.state.defense : null);
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
  if (recallChart.options.scales.y1) {
      recallChart.options.scales.y1.ticks.color = textColor;
  }
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

  // Toggle AFP Defense Button in Header
  const toggleBtn = document.getElementById("toggle-afp-btn");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", async () => {
      const nextState = !currentAfpEnabled;
      try {
        const res = await authorizedFetch("/api/dashboard/toggle-afp", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: nextState })
        });
        if (res && res.ok) {
          const data = await res.json();
          console.log("AFP Defense toggled:", data);
        }
      } catch (err) {
        console.error("Error toggling AFP:", err);
      }
    });
  }
}

// -----------------------------------------------------------------------------
// Operator Management API Handlers (Automated Session Cookie Authorization)
// -----------------------------------------------------------------------------
async function authorizedFetch(url, options = {}) {
  // Rely on HttpOnly browser session cookie established via loopback launch workflow
  options.credentials = "same-origin";
  options.headers = options.headers || {};

  let res = null;
  try {
    res = await fetch(url, options);
  } catch (netErr) {
    console.error("Network error on management request:", netErr);
    return null;
  }

  const alertBanner = document.getElementById("operator-auth-alert-banner");
  if (res && res.status === 401) {
    if (alertBanner) alertBanner.style.display = "block";
    return res;
  } else if (res && res.ok) {
    if (alertBanner) alertBanner.style.display = "none";
  }
  return res;
}

async function handleSetDefense(defName) {
  const btns = document.querySelectorAll("#defense-selector-group .segment-btn");
  try {
    const res = await authorizedFetch("/api/dashboard/set-defense", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ defense: defName })
    });
    if (res && res.ok) {
      btns.forEach(b => b.classList.toggle("active", b.dataset.defense === defName));
      currentDefenseName = defName.toUpperCase();
    }
  } catch (err) {
    console.error("Error setting defense:", err);
  }
}

async function handleSetMode(modeName) {
  const btns = document.querySelectorAll("#mode-selector-group .segment-btn");
  try {
    const res = await authorizedFetch("/api/dashboard/set-mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: modeName })
    });
    if (res && res.ok) {
      btns.forEach(b => b.classList.toggle("active", b.dataset.mode === modeName));
      currentDefenseMode = (modeName === "recall-aware") ? "Recall-Aware" : "Base";
    }
  } catch (err) {
    console.error("Error setting mode:", err);
  }
}

async function handleReset() {
  try {
    const res = await authorizedFetch("/api/dashboard/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    if (res && res.ok) {
      const data = await res.json();
      syncDashboard(data);
    }
  } catch (err) {
    console.error("Error resetting session state:", err);
  }
}

// Global window exposure for inline onclick handlers and console inspection
window.handleSetDefense = handleSetDefense;
window.handleSetMode = handleSetMode;
window.handleReset = handleReset;
window.authorizedFetch = authorizedFetch;


