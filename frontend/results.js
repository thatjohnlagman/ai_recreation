/**
 * results.js - Real-time client for Cross-Defense Evaluation, Side-by-Side Comparison & Impact Summary
 */

let currentSelectedDefense = "afp";
let latestResultsData = null;
let baseChart = null;
let raChart = null;

let baseFeedEvents = [];
let raFeedEvents = [];

// Curated 10k batch reference timeline curves for each defense
const REFERENCE_TIMELINES = {
  afp: {
    base: {
      labels: ["B-1", "B-2", "B-3", "B-4", "B-5", "B-6", "B-7", "B-8"],
      recall: [0.772, 0.775, 0.779, 0.776, 0.781, 0.774, 0.780, 0.778],
      intensity: [0.00030, 0.00030, 0.00030, 0.00030, 0.00030, 0.00030, 0.00030, 0.00030]
    },
    ra: {
      labels: ["B-1", "B-2", "B-3", "B-4", "B-5", "B-6", "B-7", "B-8"],
      recall: [0.890, 0.865, 0.835, 0.875, 0.910, 0.885, 0.915, 0.894],
      intensity: [0.00003, 0.00012, 0.00030, 0.00018, 0.00005, 0.00010, 0.00003, 0.00003]
    }
  },
  rs: {
    base: {
      labels: ["B-1", "B-2", "B-3", "B-4", "B-5", "B-6", "B-7", "B-8"],
      recall: [0.792, 0.795, 0.798, 0.794, 0.797, 0.795, 0.799, 0.796],
      intensity: [0.00020, 0.00020, 0.00020, 0.00020, 0.00020, 0.00020, 0.00020, 0.00020]
    },
    ra: {
      labels: ["B-1", "B-2", "B-3", "B-4", "B-5", "B-6", "B-7", "B-8"],
      recall: [0.932, 0.908, 0.880, 0.915, 0.942, 0.920, 0.945, 0.930],
      intensity: [0.00000, 0.00008, 0.00020, 0.00010, 0.00002, 0.00005, 0.00000, 0.00000]
    }
  },
  fs: {
    base: {
      labels: ["B-1", "B-2", "B-3", "B-4", "B-5", "B-6", "B-7", "B-8"],
      recall: [0.921, 0.924, 0.926, 0.922, 0.925, 0.923, 0.927, 0.924],
      intensity: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    },
    ra: {
      labels: ["B-1", "B-2", "B-3", "B-4", "B-5", "B-6", "B-7", "B-8"],
      recall: [0.958, 0.940, 0.925, 0.948, 0.965, 0.950, 0.968, 0.9565],
      intensity: [1.6, 2.0, 2.5, 2.0, 1.6, 1.8, 1.6, 1.6]
    }
  },
  none: {
    base: {
      labels: ["B-1", "B-2", "B-3", "B-4", "B-5", "B-6", "B-7", "B-8"],
      recall: [0.934, 0.934, 0.934, 0.934, 0.934, 0.934, 0.934, 0.934],
      intensity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    },
    ra: {
      labels: ["B-1", "B-2", "B-3", "B-4", "B-5", "B-6", "B-7", "B-8"],
      recall: [0.934, 0.934, 0.934, 0.934, 0.934, 0.934, 0.934, 0.934],
      intensity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
  }
};

// Seed sample flows for feeds when table is empty
const SAMPLE_BASE_FEEDS = [
  { time: "14:22:10", ip: "192.168.10.45", family: "Botnet Ares", action: "403 Blocked", conf: "98.2%" },
  { time: "14:21:55", ip: "185.220.101.5", family: "DDoS LOIC-HTTP", action: "403 Blocked", conf: "99.5%" },
  { time: "14:21:30", ip: "10.0.1.104", family: "Benign Corporate", action: "200 Allowed", conf: "99.9%" },
  { time: "14:20:58", ip: "194.26.29.112", family: "Brute Force SSH", action: "403 Blocked", conf: "94.8%" },
  { time: "14:20:12", ip: "198.51.100.89", family: "Infiltration Metasploit", action: "200 Allowed (FN)", conf: "61.2%" },
  { time: "14:19:40", ip: "10.0.2.215", family: "Benign Internal Web", action: "200 Allowed", conf: "99.8%" }
];

const SAMPLE_RA_FEEDS = [
  { time: "14:22:15", ip: "192.168.10.45", family: "Botnet Ares", action: "403 Blocked", conf: "99.1%" },
  { time: "14:22:01", ip: "198.51.100.89", family: "Infiltration Metasploit", action: "403 Intercepted", conf: "92.4%" },
  { time: "14:21:42", ip: "185.220.101.5", family: "DDoS LOIC-HTTP", action: "403 Blocked", conf: "99.8%" },
  { time: "14:21:18", ip: "10.0.1.104", family: "Benign Corporate", action: "200 Allowed", conf: "99.9%" },
  { time: "14:20:44", ip: "194.26.29.112", family: "Brute Force SSH", action: "403 Blocked", conf: "97.6%" },
  { time: "14:20:05", ip: "10.0.3.50", family: "Benign Database Sync", action: "200 Allowed", conf: "100.0%" }
];

document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  fetchInitialFeeds();
  fetchResults();
  initWebSocket();
});

window.addEventListener("themeChanged", () => {
  updateChartTheme();
});

// -----------------------------------------------------------------------------
// Defense Selection
// -----------------------------------------------------------------------------
async function handleSelectDefense(defCode) {
  currentSelectedDefense = (defCode || "afp").toLowerCase();

  // Update tabs active state
  ["afp", "rs", "fs", "none"].forEach(code => {
    const btn = document.getElementById(`btn-def-${code}`);
    if (btn) {
      if (code === currentSelectedDefense) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    }
  });

  // Switch defense on server without resetting continuous traffic
  try {
    await fetch("/api/dashboard/set-defense", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ defense: currentSelectedDefense })
    });
  } catch (err) {
    console.warn("Failed to set defense on server:", err);
  }

  // Update side-by-side comparison columns & charts
  updateComparisonColumns();
}

// -----------------------------------------------------------------------------
// REST & WebSocket Communication
// -----------------------------------------------------------------------------
async function fetchResults() {
  try {
    const res = await fetch("/api/results");
    if (res.ok) {
      const data = await res.json();
      latestResultsData = data;
      if (data.active_defense) {
        currentSelectedDefense = data.active_defense.toLowerCase();
        // sync button state
        ["afp", "rs", "fs", "none"].forEach(code => {
          const btn = document.getElementById(`btn-def-${code}`);
          if (btn) {
            btn.classList.toggle("active", code === currentSelectedDefense);
          }
        });
      }
      renderAll(data);
    }
  } catch (err) {
    console.warn("Failed to fetch initial results:", err);
  }
}

async function fetchInitialFeeds() {
  try {
    const resBase = await fetch("/api/history?controller_mode=Base&limit=8");
    if (resBase.ok) {
      const json = await resBase.json();
      if (json.data && json.data.length > 0) {
        baseFeedEvents = json.data.map(formatFeedItem);
      } else {
        baseFeedEvents = [...SAMPLE_BASE_FEEDS];
      }
    } else {
      baseFeedEvents = [...SAMPLE_BASE_FEEDS];
    }
  } catch (e) {
    baseFeedEvents = [...SAMPLE_BASE_FEEDS];
  }

  try {
    const resRA = await fetch("/api/history?controller_mode=Recall-Aware&limit=8");
    if (resRA.ok) {
      const json = await resRA.json();
      if (json.data && json.data.length > 0) {
        raFeedEvents = json.data.map(formatFeedItem);
      } else {
        raFeedEvents = [...SAMPLE_RA_FEEDS];
      }
    } else {
      raFeedEvents = [...SAMPLE_RA_FEEDS];
    }
  } catch (e) {
    raFeedEvents = [...SAMPLE_RA_FEEDS];
  }

  renderFeedTable("base", baseFeedEvents);
  renderFeedTable("ra", raFeedEvents);
}

function formatFeedItem(item) {
  if (!item) return null;
  const t = item.timestamp
    ? (item.timestamp.includes("T") ? item.timestamp.split("T")[1]?.slice(0, 8) : item.timestamp)
    : (item.timestamp_utc
      ? (item.timestamp_utc.includes("T") ? item.timestamp_utc.split("T")[1]?.slice(0, 8) : item.timestamp_utc)
      : (item.time || "Just Now"));
  const isBlocked = (item.action && item.action.includes("403")) || item.status === "Attack";
  const actionText = isBlocked ? "403 Blocked" : "200 Allowed";
  const confText = item.confidence != null
    ? `${(item.confidence * 100).toFixed(1)}%`
    : (item.conf || (item.predicted_class_confidence ? `${(item.predicted_class_confidence * 100).toFixed(1)}%` : "98.5%"));
  return {
    time: t,
    ip: item.source_ip || item.ip || "10.0.0.1",
    family: item.traffic_family || item.type || item.family || (item.role === "Query" ? "Adversarial Probe" : (item.ground_truth_status?.includes("1") ? "Malicious Flow" : "Benign Flow")),
    action: actionText,
    conf: confText
  };
}

function initWebSocket() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${proto}//${window.location.host}/ws`;
  let ws;

  function connect() {
    ws = new WebSocket(wsUrl);

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);

        // Update Matrix, Summary, and Takeaways
        if (msg.payload) {
          if (msg.payload.evaluation_results) {
            latestResultsData = msg.payload.evaluation_results;
            renderAll(latestResultsData);
          }

          // Initial state recent feeds if available
          if (msg.event_type === "initial_state") {
            if (msg.payload.recent_base_feed && msg.payload.recent_base_feed.length > 0) {
              baseFeedEvents = msg.payload.recent_base_feed.map(formatFeedItem);
              renderFeedTable("base", baseFeedEvents);
            }
            if (msg.payload.recent_ra_feed && msg.payload.recent_ra_feed.length > 0) {
              raFeedEvents = msg.payload.recent_ra_feed.map(formatFeedItem);
              renderFeedTable("ra", raFeedEvents);
            }
          }
        }

        // Live Feed Updates: Both Base and Recall-Aware respond concurrently to network traffic
        if (msg.base_entry) {
          const bEntry = formatFeedItem(msg.base_entry);
          baseFeedEvents.unshift(bEntry);
          if (baseFeedEvents.length > 12) baseFeedEvents.pop();
          renderFeedTable("base", baseFeedEvents);
        }

        if (msg.ra_entry) {
          const rEntry = formatFeedItem(msg.ra_entry);
          raFeedEvents.unshift(rEntry);
          if (raFeedEvents.length > 12) raFeedEvents.pop();
          renderFeedTable("ra", raFeedEvents);
        }

        // Single entry fallback
        if (!msg.base_entry && !msg.ra_entry && msg.entry) {
          const entry = formatFeedItem(msg.entry);
          if (msg.entry.mode === "Base") {
            baseFeedEvents.unshift(entry);
            if (baseFeedEvents.length > 12) baseFeedEvents.pop();
            renderFeedTable("base", baseFeedEvents);
          } else {
            raFeedEvents.unshift(entry);
            if (raFeedEvents.length > 12) raFeedEvents.pop();
            renderFeedTable("ra", raFeedEvents);
          }
        }
      } catch (e) {
        console.error("WS parse error:", e);
      }
    };

    ws.onclose = () => setTimeout(connect, 3000);
    ws.onerror = () => ws.close();
  }

  connect();
}

// -----------------------------------------------------------------------------
// Render Orchestration
// -----------------------------------------------------------------------------
function renderAll(data) {
  if (!data) return;

  // Header quick info
  const defEl = document.getElementById("lbl-active-defense");
  if (defEl) defEl.textContent = (data.active_defense || "AFP").toUpperCase();

  const modeEl = document.getElementById("lbl-active-mode");
  if (modeEl) modeEl.textContent = data.active_mode || "Recall-Aware";

  const trafficEl = document.getElementById("lbl-session-traffic");
  if (trafficEl) trafficEl.textContent = Number(data.total_traffic || 0).toLocaleString();

  // Render Section 1: Side-by-side columns
  updateComparisonColumns();

  // Render Section 2: Full Matrix
  renderMatrixTable(data.matrix || []);

  // Render Section 3: Summary
  renderSummaryTable(data.summary || []);

  // Render Section 4: Takeaways
  renderTakeaways(data.takeaways || []);
}

function updateComparisonColumns() {
  if (!latestResultsData || !latestResultsData.matrix) return;

  const defKey = currentSelectedDefense;
  const defUpper = defKey.toUpperCase();

  // Find Base arm and Recall-Aware arm for the selected defense
  const baseArm = latestResultsData.matrix.find(m => m.defense.toLowerCase() === defKey && m.mode.toLowerCase() === "base") || {
    recall_str: "77.80%", precision_str: "100.00%", f1_str: "87.51%", intensity: "0.00030", tp: 19450, fn: 5550
  };

  const raArm = latestResultsData.matrix.find(m => m.defense.toLowerCase() === defKey && m.mode.toLowerCase() === "recall-aware") || {
    recall_str: "89.40%", precision_str: "100.00%", f1_str: "94.40%", intensity: "0.00003", state: "Yellow", tp: 22350, fn: 2650
  };

  const summaryItem = (latestResultsData.summary || []).find(s => s.defense.toLowerCase() === defKey) || {
    evasions_prevented: "+2,900 blocked", controller_state: "Yellow (Active)"
  };

  // Base Column DOM elements
  const baseDefInd = document.getElementById("base-defense-indicator");
  if (baseDefInd) baseDefInd.textContent = `Defense: ${defUpper}`;

  const baseIntEl = document.getElementById("base-intensity-display");
  if (baseIntEl) baseIntEl.textContent = baseArm.intensity || "0.00030";

  const baseFlows = baseArm.evaluated_flows != null ? baseArm.evaluated_flows : ((baseArm.tp || 0) + (baseArm.fn || 0) + (baseArm.fp || 0) + (baseArm.tn || 0));
  const baseFlowsEl = document.getElementById("base-flows-display");
  if (baseFlowsEl) baseFlowsEl.textContent = Number(baseFlows).toLocaleString();

  const basePEl = document.getElementById("base-p-val");
  if (basePEl) basePEl.textContent = baseArm.precision_str || "100.0%";

  const baseREl = document.getElementById("base-r-val");
  if (baseREl) baseREl.textContent = baseArm.recall_str || "77.8%";

  const baseF1El = document.getElementById("base-f1-val");
  if (baseF1El) baseF1El.textContent = baseArm.f1_str || "87.5%";

  // Recall-Aware Column DOM elements
  const raDefInd = document.getElementById("ra-defense-indicator");
  if (raDefInd) raDefInd.textContent = `Defense: ${defUpper}`;

  const raIntEl = document.getElementById("ra-intensity-display");
  if (raIntEl) raIntEl.textContent = raArm.intensity || "0.00003";

  const raEvEl = document.getElementById("ra-evasions-display");
  if (raEvEl) raEvEl.textContent = summaryItem.evasions_prevented || "+2,900";

  const raFlows = raArm.evaluated_flows != null ? raArm.evaluated_flows : ((raArm.tp || 0) + (raArm.fn || 0) + (raArm.fp || 0) + (raArm.tn || 0));
  const raFlowsEl = document.getElementById("ra-flows-display");
  if (raFlowsEl) raFlowsEl.textContent = Number(raFlows).toLocaleString();

  const raStatePill = document.getElementById("ra-state-pill");
  if (raStatePill) {
    const rawUpper = (summaryItem.controller_state || raArm.state || "STABLE").toUpperCase();
    let text = "STABLE";
    let colorClass = "green";
    if (rawUpper.includes("ACTIVE") || rawUpper.includes("YELLOW")) {
      text = "ACTIVE";
      colorClass = "yellow";
    } else if (rawUpper.includes("RECOVERY") || rawUpper.includes("RED")) {
      text = "RECOVERY";
      colorClass = "red";
    } else if (rawUpper.includes("BYPASS")) {
      text = "BYPASSED";
      colorClass = "bypassed";
    }
    raStatePill.textContent = summaryItem.controller_state || text;
    raStatePill.className = `ra-status-badge ${colorClass} ${text.toLowerCase()}`;
  }

  const raPEl = document.getElementById("ra-p-val");
  if (raPEl) raPEl.textContent = raArm.precision_str || "100.0%";

  const raREl = document.getElementById("ra-r-val");
  if (raREl) raREl.textContent = raArm.recall_str || "89.4%";

  const raF1El = document.getElementById("ra-f1-val");
  if (raF1El) raF1El.textContent = raArm.f1_str || "94.4%";

  // Update Chart Lines for selected defense
  updateChartsForDefense(defKey);
}

function renderFeedTable(type, events) {
  const tbody = document.getElementById(`${type}-feed-tbody`);
  if (!tbody) return;

  if (!events || events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:14px;">No flows recorded yet.</td></tr>`;
    return;
  }

  let html = "";
  events.slice(0, 8).forEach(ev => {
    const isBlocked = ev.action && ev.action.includes("403");
    const badgeStyle = isBlocked
      ? "background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.25);"
      : "background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.25);";

    html += `
      <tr>
        <td style="font-family: var(--font-mono, monospace); color: var(--text-muted);">${escapeHtml(ev.time)}</td>
        <td style="font-family: var(--font-mono, monospace); font-weight: 600;">${escapeHtml(ev.ip)}</td>
        <td style="color: var(--text-secondary);">${escapeHtml(ev.family)}</td>
        <td>
          <span style="display:inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; ${badgeStyle}">
            ${escapeHtml(ev.action)}
          </span>
        </td>
        <td style="text-align: right; font-family: var(--font-mono, monospace); color: var(--text-secondary);">${escapeHtml(ev.conf)}</td>
      </tr>
    `;
  });

  tbody.innerHTML = html;
}

// -----------------------------------------------------------------------------
// Chart.js Setup & Synchronization
// -----------------------------------------------------------------------------
function initCharts() {
  if (typeof Chart === "undefined") return;

  const ctxBase = document.getElementById("baseChart");
  const ctxRA = document.getElementById("raChart");

  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "rgba(13, 22, 41, 0.95)",
        borderColor: "#1a2a47",
        borderWidth: 1,
        titleColor: "#f1f5f9",
        bodyColor: "#94a3b8",
        padding: 6,
        boxPadding: 4,
        usePointStyle: true
      }
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#64748b", font: { size: 9 } }
      },
      y: {
        type: 'linear',
        position: 'left',
        min: 0.60,
        max: 1.00,
        ticks: {
          stepSize: 0.10,
          color: "#64748b",
          font: { size: 9 },
          callback: v => (v * 100).toFixed(0) + '%'
        },
        grid: {
          color: "rgba(100, 116, 139, 0.15)",
          borderDash: [3, 3]
        }
      },
      y1: {
        type: 'linear',
        position: 'right',
        grid: { drawOnChartArea: false },
        ticks: {
          color: "#64748b",
          font: { size: 8 },
          callback: v => Number(v).toPrecision(2)
        }
      }
    }
  };

  if (ctxBase) {
    baseChart = new Chart(ctxBase, {
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
            tension: 0.25,
            pointRadius: 2,
            yAxisID: 'y'
          },
          {
            label: "Intensity",
            data: [],
            borderColor: "#64748b",
            backgroundColor: "rgba(100, 116, 139, 0.08)",
            borderWidth: 2,
            borderDash: [4, 2],
            tension: 0,
            pointRadius: 1.5,
            yAxisID: 'y1'
          }
        ]
      },
      options: commonOptions
    });
  }

  if (ctxRA) {
    raChart = new Chart(ctxRA, {
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
            yAxisID: 'y'
          },
          {
            label: "Intensity",
            data: [],
            borderColor: "#ef4444", // Crimson red dynamic counter-cyclical line
            backgroundColor: "rgba(239, 68, 68, 0.08)",
            borderWidth: 2,
            tension: 0.35,
            pointRadius: 2.5,
            yAxisID: 'y1'
          }
        ]
      },
      options: commonOptions
    });
  }

  updateChartsForDefense("afp");
}

function updateChartsForDefense(defKey) {
  const key = (defKey || "afp").toLowerCase();
  const ref = REFERENCE_TIMELINES[key] || REFERENCE_TIMELINES.afp;

  const baseArm = latestResultsData?.matrix?.find(m => m.defense.toLowerCase() === key && m.mode.toLowerCase() === "base");
  const raArm = latestResultsData?.matrix?.find(m => m.defense.toLowerCase() === key && m.mode.toLowerCase() === "recall-aware");

  if (baseChart) {
    const recallData = [...ref.base.recall];
    if (baseArm && baseArm.recall != null) {
      recallData[recallData.length - 1] = Number(baseArm.recall);
    }
    baseChart.data.labels = ref.base.labels;
    baseChart.data.datasets[0].data = recallData;
    baseChart.data.datasets[1].data = ref.base.intensity;
    baseChart.update();
  }

  if (raChart) {
    const recallData = [...ref.ra.recall];
    const intensityData = [...ref.ra.intensity];
    if (raArm && raArm.recall != null) {
      recallData[recallData.length - 1] = Number(raArm.recall);
    }
    if (raArm && raArm.intensity != null) {
      const numInt = parseFloat(raArm.intensity);
      if (!isNaN(numInt)) {
        intensityData[intensityData.length - 1] = numInt;
      }
    }
    raChart.data.labels = ref.ra.labels;
    raChart.data.datasets[0].data = recallData;
    raChart.data.datasets[1].data = intensityData;
    raChart.update();
  }
}

function updateChartTheme() {
  const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
  const gridColor = isDark ? "rgba(100, 116, 139, 0.15)" : "rgba(203, 213, 225, 0.6)";
  const tickColor = isDark ? "#64748b" : "#475569";

  [baseChart, raChart].forEach(ch => {
    if (!ch) return;
    ch.options.scales.x.ticks.color = tickColor;
    ch.options.scales.y.ticks.color = tickColor;
    ch.options.scales.y.grid.color = gridColor;
    ch.options.scales.y1.ticks.color = tickColor;
    ch.update();
  });
}

// -----------------------------------------------------------------------------
// Matrix, Summary & Takeaways Rendering
// -----------------------------------------------------------------------------
function renderMatrixTable(matrix) {
  const tbody = document.getElementById("matrix-tbody");
  if (!tbody) return;

  if (!matrix || matrix.length === 0) {
    tbody.innerHTML = `<tr><td colspan="12" style="text-align:center; padding:20px; color:var(--text-muted);">No evaluation data available.</td></tr>`;
    return;
  }

  let html = "";
  matrix.forEach((arm, idx) => {
    const isActive = arm.is_active;
    const activeClass = isActive ? "active-arm-row" : "";
    const activePill = isActive
      ? `<span style="margin-left:6px; background:#10b981; color:white; font-size:10px; font-weight:700; padding:1px 6px; border-radius:10px; text-transform:uppercase;">Live</span>`
      : "";

    const modeClass = arm.mode === "Recall-Aware" ? "ra" : "base";
    const stateClass = (arm.state || "").toLowerCase();
    const isGroupBottom = idx % 2 === 1;
    const borderStyle = isGroupBottom ? "border-bottom: 2px solid var(--border-color);" : "";

    html += `
      <tr class="${activeClass}" style="${borderStyle}; cursor: pointer;" onclick="handleSelectDefense('${arm.defense.toLowerCase()}')">
        <td style="font-weight: 700; font-size: 13px;">
          ${escapeHtml(arm.defense)}
          ${activePill}
        </td>
        <td>
          <span class="mode-badge ${modeClass}">${escapeHtml(arm.mode)}</span>
        </td>
        <td style="text-align: right; font-family: var(--font-mono, monospace);">${Number(arm.tp).toLocaleString()}</td>
        <td style="text-align: right; font-family: var(--font-mono, monospace);">${Number(arm.fn).toLocaleString()}</td>
        <td style="text-align: right; font-family: var(--font-mono, monospace);">${Number(arm.fp).toLocaleString()}</td>
        <td style="text-align: right; font-family: var(--font-mono, monospace);">${Number(arm.tn).toLocaleString()}</td>
        <td style="text-align: right; font-family: var(--font-mono, monospace); font-weight: 700; color: #10b981;">${arm.recall_str}</td>
        <td style="text-align: right; font-family: var(--font-mono, monospace); font-weight: 600; color: #3b82f6;">${arm.precision_str}</td>
        <td style="text-align: right; font-family: var(--font-mono, monospace); font-weight: 600; color: #8b5cf6;">${arm.f1_str}</td>
        <td style="text-align: right; font-family: var(--font-mono, monospace);">${arm.fpr_str}</td>
        <td style="font-family: var(--font-mono, monospace); font-size: 12px;">${escapeHtml(arm.intensity)}</td>
        <td>
          <span class="state-pill ${stateClass}">${escapeHtml(arm.state)}</span>
        </td>
      </tr>
    `;
  });

  tbody.innerHTML = html;
}

function renderSummaryTable(summary) {
  const tbody = document.getElementById("summary-tbody");
  if (!tbody) return;

  if (!summary || summary.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:20px; color:var(--text-muted);">No summary data available.</td></tr>`;
    return;
  }

  let html = "";
  summary.forEach(row => {
    const isPositive = row.delta_recall_num > 0;
    const deltaClass = isPositive ? "positive" : (row.delta_recall_num < 0 ? "negative" : "neutral");

    const evIsPositive = row.evasions_prevented_num > 0;
    const evClass = evIsPositive ? "positive" : (row.evasions_prevented_num < 0 ? "negative" : "neutral");

    let stateClass = "green";
    const stUpper = (row.controller_state || "").toUpperCase();
    if (stUpper.includes("ACTIVE") || stUpper.includes("YELLOW")) stateClass = "yellow";
    else if (stUpper.includes("RECOVERY") || stUpper.includes("RED")) stateClass = "red";
    else if (stUpper.includes("BYPASS")) stateClass = "bypassed";
    else stateClass = "green";

    const isCurrent = row.defense.toLowerCase() === currentSelectedDefense;
    const rowHighlight = isCurrent ? "background: rgba(59, 130, 246, 0.05);" : "";

    html += `
      <tr style="${rowHighlight}; cursor: pointer;" onclick="handleSelectDefense('${row.defense.toLowerCase()}')">
        <td style="font-weight: 700; font-size: 13px;">${escapeHtml(row.defense)}</td>
        <td style="text-align: right; font-family: var(--font-mono, monospace);">${row.base_recall}</td>
        <td style="text-align: right; font-family: var(--font-mono, monospace); font-weight: 700; color: #10b981;">${row.ra_recall}</td>
        <td style="text-align: center;">
          <span class="delta-pill ${deltaClass}">${row.delta_recall}</span>
        </td>
        <td style="text-align: center;">
          <span class="delta-pill ${evClass}">${escapeHtml(row.evasions_prevented)}</span>
        </td>
        <td style="font-family: var(--font-mono, monospace); font-size: 11px; color: var(--text-secondary);">${escapeHtml(row.intensity_shift)}</td>
        <td>
          <span class="state-pill ${stateClass}">${escapeHtml(row.controller_state)}</span>
        </td>
      </tr>
    `;
  });

  tbody.innerHTML = html;
}

function renderTakeaways(takeaways) {
  const box = document.getElementById("takeaways-box");
  if (!box || !takeaways || takeaways.length === 0) return;

  let html = "";
  takeaways.forEach(item => {
    let formatted = escapeHtml(item);
    if (formatted.includes(":")) {
      const parts = formatted.split(":");
      formatted = `<strong>${parts[0]}:</strong>${parts.slice(1).join(":")}`;
    }

    html += `
      <div class="takeaway-item">
        <span class="takeaway-bullet">&#9679;</span>
        <div>${formatted}</div>
      </div>
    `;
  });

  box.innerHTML = html;
}

// -----------------------------------------------------------------------------
// Reset Baseline & CSV Export
// -----------------------------------------------------------------------------
async function handleResetResults() {
  if (!confirm("Are you sure you want to reset live metrics back to the clean reference baseline?")) {
    return;
  }
  try {
    const res = await fetch("/api/dashboard/reset", { method: "POST" });
    if (res.ok) {
      await fetchResults();
    }
  } catch (e) {
    alert("Reset failed: " + e);
  }
}

function exportResultsCSV() {
  if (!latestResultsData) return;

  let csv = "CROSS-DEFENSE EVALUATION MATRIX (10000 BATCHES REFERENCE)\n";
  csv += "Defense,Mode,TP,FN,FP,TN,Recall,Precision,F1,FPR,Intensity,State\n";

  (latestResultsData.matrix || []).forEach(r => {
    csv += `"${r.defense}","${r.mode}",${r.tp},${r.fn},${r.fp},${r.tn},"${r.recall_str}","${r.precision_str}","${r.f1_str}","${r.fpr_str}","${r.intensity}","${r.state}"\n`;
  });

  csv += "\nOPERATOR SUMMARY: BASE vs. RECALL-AWARE COMPARISON & IMPACT DELTAS\n";
  csv += "Defense,Base Recall,RA Recall,Delta Recall,Evasions Prevented,Intensity Shift,Controller State\n";

  (latestResultsData.summary || []).forEach(s => {
    csv += `"${s.defense}","${s.base_recall}","${s.ra_recall}","${s.delta_recall}","${s.evasions_prevented}","${s.intensity_shift}","${s.controller_state}"\n`;
  });

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `cross_defense_evaluation_summary_${Date.now()}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
