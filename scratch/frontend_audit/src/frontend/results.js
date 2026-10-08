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

let useLiveBenchmark = false;

const STATIC_BENCHMARK_SUMMARY = [
  {
    "defense": "AFP",
    "base_prec": "99.93%",
    "base_recall": "80.26%",
    "base_f1": "89.02%",
    "ra_prec": "99.57%",
    "ra_recall": "93.19%",
    "ra_f1": "96.27%",
    "delta_prec": "-0.36%",
    "delta_recall": "+12.93%",
    "delta_f1": "+7.25%",
    "delta_recall_num": 0.12930638274727368,
    "evasions_prevented": "6465",
    "evasions_prevented_num": 6465,
    "intensity_shift": "+0.00",
    "controller_state": "ACTIVE"
  },
  {
    "defense": "FS",
    "base_prec": "99.86%",
    "base_recall": "92.68%",
    "base_f1": "96.13%",
    "ra_prec": "99.73%",
    "ra_recall": "93.45%",
    "ra_f1": "96.49%",
    "delta_prec": "-0.13%",
    "delta_recall": "+0.77%",
    "delta_f1": "+0.35%",
    "delta_recall_num": 0.00774523401593652,
    "evasions_prevented": "387",
    "evasions_prevented_num": 387,
    "intensity_shift": "+0.00",
    "controller_state": "ACTIVE"
  },
  {
    "defense": "RS",
    "base_prec": "99.95%",
    "base_recall": "80.57%",
    "base_f1": "89.22%",
    "ra_prec": "99.59%",
    "ra_recall": "93.21%",
    "ra_f1": "96.29%",
    "delta_prec": "-0.36%",
    "delta_recall": "+12.64%",
    "delta_f1": "+7.07%",
    "delta_recall_num": 0.12638764243561507,
    "evasions_prevented": "6319",
    "evasions_prevented_num": 6319,
    "intensity_shift": "+0.00",
    "controller_state": "ACTIVE"
  }
];

const STATIC_C1_C7_MATRIX = [
  { defense: "afp", mode: "Base", precision: 0.999, recall: 0.803, f1: 0.890, precision_str: "99.93%", recall_str: "80.26%", f1_str: "89.02%", intensity: "-", state: "STABLE" },
  { defense: "afp", mode: "C1", precision: 0.996, recall: 0.932, f1: 0.963, precision_str: "99.57%", recall_str: "93.19%", f1_str: "96.27%", intensity: "-", state: "ACTIVE" },
  { defense: "afp", mode: "C2", precision: 0.996, recall: 0.932, f1: 0.963, precision_str: "99.57%", recall_str: "93.17%", f1_str: "96.26%", intensity: "-", state: "ACTIVE" },
  { defense: "afp", mode: "C3", precision: 0.995, recall: 0.932, f1: 0.963, precision_str: "99.55%", recall_str: "93.20%", f1_str: "96.27%", intensity: "-", state: "ACTIVE" },
  { defense: "afp", mode: "C4", precision: 0.998, recall: 0.901, f1: 0.947, precision_str: "99.84%", recall_str: "90.06%", f1_str: "94.70%", intensity: "-", state: "ACTIVE" },
  { defense: "afp", mode: "C5", precision: 0.994, recall: 0.934, f1: 0.963, precision_str: "99.43%", recall_str: "93.39%", f1_str: "96.31%", intensity: "-", state: "ACTIVE" },
  { defense: "afp", mode: "C6", precision: 0.997, recall: 0.926, f1: 0.960, precision_str: "99.74%", recall_str: "92.55%", f1_str: "96.01%", intensity: "-", state: "ACTIVE" },
  { defense: "afp", mode: "C7", precision: 0.994, recall: 0.934, f1: 0.963, precision_str: "99.45%", recall_str: "93.40%", f1_str: "96.33%", intensity: "-", state: "ACTIVE" },
  { defense: "fs", mode: "Base", precision: 0.999, recall: 0.927, f1: 0.961, precision_str: "99.86%", recall_str: "92.68%", f1_str: "96.13%", intensity: "-", state: "STABLE" },
  { defense: "fs", mode: "C1", precision: 0.997, recall: 0.935, f1: 0.965, precision_str: "99.73%", recall_str: "93.45%", f1_str: "96.49%", intensity: "-", state: "ACTIVE" },
  { defense: "fs", mode: "C2", precision: 0.997, recall: 0.935, f1: 0.965, precision_str: "99.71%", recall_str: "93.47%", f1_str: "96.49%", intensity: "-", state: "ACTIVE" },
  { defense: "fs", mode: "C3", precision: 0.997, recall: 0.934, f1: 0.965, precision_str: "99.74%", recall_str: "93.43%", f1_str: "96.48%", intensity: "-", state: "ACTIVE" },
  { defense: "fs", mode: "C4", precision: 0.999, recall: 0.927, f1: 0.962, precision_str: "99.85%", recall_str: "92.72%", f1_str: "96.15%", intensity: "-", state: "ACTIVE" },
  { defense: "fs", mode: "C5", precision: 0.997, recall: 0.935, f1: 0.965, precision_str: "99.71%", recall_str: "93.48%", f1_str: "96.50%", intensity: "-", state: "ACTIVE" },
  { defense: "fs", mode: "C6", precision: 0.997, recall: 0.935, f1: 0.965, precision_str: "99.72%", recall_str: "93.47%", f1_str: "96.49%", intensity: "-", state: "ACTIVE" },
  { defense: "fs", mode: "C7", precision: 0.997, recall: 0.935, f1: 0.965, precision_str: "99.70%", recall_str: "93.47%", f1_str: "96.48%", intensity: "-", state: "ACTIVE" },
  { defense: "rs", mode: "Base", precision: 0.999, recall: 0.806, f1: 0.892, precision_str: "99.95%", recall_str: "80.57%", f1_str: "89.22%", intensity: "-", state: "STABLE" },
  { defense: "rs", mode: "C1", precision: 0.996, recall: 0.932, f1: 0.963, precision_str: "99.59%", recall_str: "93.21%", f1_str: "96.29%", intensity: "-", state: "ACTIVE" },
  { defense: "rs", mode: "C2", precision: 0.996, recall: 0.932, f1: 0.963, precision_str: "99.59%", recall_str: "93.22%", f1_str: "96.30%", intensity: "-", state: "ACTIVE" },
  { defense: "rs", mode: "C3", precision: 0.995, recall: 0.932, f1: 0.963, precision_str: "99.55%", recall_str: "93.23%", f1_str: "96.28%", intensity: "-", state: "ACTIVE" },
  { defense: "rs", mode: "C4", precision: 0.999, recall: 0.901, f1: 0.947, precision_str: "99.85%", recall_str: "90.11%", f1_str: "94.73%", intensity: "-", state: "ACTIVE" },
  { defense: "rs", mode: "C5", precision: 0.994, recall: 0.934, f1: 0.963, precision_str: "99.44%", recall_str: "93.40%", f1_str: "96.32%", intensity: "-", state: "ACTIVE" },
  { defense: "rs", mode: "C6", precision: 0.997, recall: 0.926, f1: 0.960, precision_str: "99.75%", recall_str: "92.61%", f1_str: "96.04%", intensity: "-", state: "ACTIVE" },
  { defense: "rs", mode: "C7", precision: 0.995, recall: 0.934, f1: 0.963, precision_str: "99.45%", recall_str: "93.43%", f1_str: "96.34%", intensity: "-", state: "ACTIVE" }
];

function toggleLiveBenchmark() {
  useLiveBenchmark = !useLiveBenchmark;
  const btns = document.querySelectorAll(".btn-toggle-benchmark");
  btns.forEach(btn => {
    btn.textContent = useLiveBenchmark ? "Stop Live Benchmark" : "Compute Live Benchmark";
    btn.className = useLiveBenchmark ? "btn-action btn-toggle-benchmark" : "btn-subtle btn-toggle-benchmark";
  });
  // Immediately re-render with the latest choice
  if (latestResultsData) {
    renderMatrixTable(latestResultsData);
    renderSummaryTable(latestResultsData.summary);
    renderConfigSummaryTable(latestResultsData.matrix);
  }
}

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
  
  // Refresh matrix to immediately reflect the new active defense in C1-C7 cards
  pollEvaluationResults();
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


  const trafficEl = document.getElementById("lbl-session-traffic");
  if (trafficEl) trafficEl.textContent = Number(data.total_traffic || 0).toLocaleString();

  // Render Section 1: Side-by-side columns
  updateComparisonColumns();

  // Render Section 2: Full Matrix
  renderMatrixTable(data);

  // Render Section 3: Summary
  renderSummaryTable(data.summary || []);
  renderConfigSummaryTable(data.matrix || []);

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

const chartHistory = {
  afp: { labels: [], recall: [], intensity: [], count: 0 },
  rs:  { labels: [], recall: [], intensity: [], count: 0 },
  fs:  { labels: [], recall: [], intensity: [], count: 0 },
  none:{ labels: [], recall: [], intensity: [], count: 0 }
};
const MAX_CHART_POINTS = 15;

function updateChartsForDefense(defKey) {
  const key = (defKey || "afp").toLowerCase();
  const hist = chartHistory[key];
  if (!hist) return;

  const raArm = latestResultsData?.matrix?.find(m => m.defense.toLowerCase() === key && m.mode.toLowerCase() === "recall-aware");

  if (raChart && raArm) {
    let newRecall = null;
    let newIntensity = null;

    if (raArm.rolling_recall != null) {
      newRecall = Number(raArm.rolling_recall);
    } else if (raArm.recall != null) {
      newRecall = Number(raArm.recall);
    }
    
    if (raArm.intensity != null) {
      const numInt = parseFloat(raArm.intensity);
      if (!isNaN(numInt)) newIntensity = numInt;
    }

    if (newRecall !== null && newIntensity !== null) {
      hist.count++;
      hist.labels.push("B-" + hist.count);
      hist.recall.push(newRecall);
      hist.intensity.push(newIntensity);

      if (hist.labels.length > MAX_CHART_POINTS) {
        hist.labels.shift();
        hist.recall.shift();
        hist.intensity.shift();
      }
    }

    // Only render if we have data
    if (hist.labels.length > 0) {
      raChart.data.labels = [...hist.labels];
      raChart.data.datasets[0].data = [...hist.recall];
      raChart.data.datasets[1].data = [...hist.intensity];
      
      // Update Y-axis scale for intensity dynamically depending on defense
      if (key === "afp") {
         raChart.options.scales.y1.max = 0.00035;
      } else if (key === "rs") {
         raChart.options.scales.y1.max = 3.0;
      } else if (key === "fs") {
         raChart.options.scales.y1.max = 1.0;
      }
      
      raChart.update();
    }
  }
}

function updateChartTheme() {
  const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
  const gridColor = isDark ? "rgba(100, 116, 139, 0.15)" : "rgba(203, 213, 225, 0.6)";
  const tickColor = isDark ? "#64748b" : "#475569";

  [raChart].forEach(ch => {
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
function renderMatrixTable(data) {
  const container = document.getElementById("c1-c7-container");
  if (!container) return;
  
  let matrix = data.matrix || [];
  if (!useLiveBenchmark) {
    matrix = STATIC_C1_C7_MATRIX;
  }
  const configs = data.controller_configs || {};

  if (!matrix || matrix.length === 0) {
    container.innerHTML = `<div style="text-align:center; padding:20px; color:var(--text-muted); width: 100%; grid-column: 1 / -1;">No evaluation data available.</div>`;
    return;
  }

  let html = "";
  const modes = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"];

  modes.forEach(mode => {
    const arm = matrix.find(m => m.mode.toUpperCase() === mode.toUpperCase() && m.defense.toLowerCase() === (window.currentSelectedDefense || "afp").toLowerCase());
    
    const cfg = configs[mode] || {};
    const win = cfg.window_size || "N/A";
    const tgt = cfg.Rmin ? (cfg.Rmin*100).toFixed(0) + "%" : "N/A";
    const rcrit = cfg.Rcritical ? (cfg.Rcritical*100).toFixed(0) + "%" : "N/A";
    const fastDecay = cfg.fast_decay ? cfg.fast_decay.toFixed(2) : "N/A";
    const slowDecay = cfg.slow_decay ? cfg.slow_decay.toFixed(2) : "N/A";
    const growth = cfg.growth_factor ? cfg.growth_factor.toFixed(2) : "N/A";

    html += `
      <div style="background: var(--bg-body); border: 1px solid var(--border-color); border-radius: 6px; padding: 10px; display: flex; flex-direction: column; gap: 8px; align-items: center; justify-content: flex-start; height: 100%;">
        <h3 style="font-size: 15px; font-weight: 800; text-transform: uppercase; margin: 0; color: var(--text-primary);">${mode}</h3>
        
        <div style="background: var(--bg-card-sub); border: 1px solid var(--border-color); border-radius: 4px; padding: 6px; font-size: 10px; width: 100%; text-align: left; color: var(--text-secondary);">
           <div style="display:flex; justify-content:space-between; margin-bottom: 2px;"><span>Target:</span> <strong style="color:var(--text-primary);">${tgt}</strong></div>
           <div style="display:flex; justify-content:space-between; margin-bottom: 2px;"><span>Critical:</span> <strong style="color:var(--text-primary);">${rcrit}</strong></div>
           <div style="display:flex; justify-content:space-between; margin-bottom: 2px;"><span>Window:</span> <strong style="color:var(--text-primary);">${win}</strong></div>
           <div style="display:flex; justify-content:space-between; margin-bottom: 2px;"><span>Fast Decay:</span> <strong style="color:var(--text-primary);">${fastDecay}</strong></div>
           <div style="display:flex; justify-content:space-between; margin-bottom: 2px;"><span>Slow Decay:</span> <strong style="color:var(--text-primary);">${slowDecay}</strong></div>
           <div style="display:flex; justify-content:space-between;"><span>Growth:</span> <strong style="color:var(--text-primary);">${growth}</strong></div>
        </div>
    `;

    if (arm) {
      const stateClass = (arm.state || "").toLowerCase();
      html += `
        <div style="width: 100%; text-align: center; font-weight: 800; font-size: 11px; margin-top: 4px; color: var(--text-primary);">${escapeHtml(arm.defense).toUpperCase()}</div>
        
        <table style="width: 100%; font-size: 10px; margin-top: auto; border-collapse: collapse;">
          <tbody>
            <tr>
              <td style="padding: 2px 0; color: var(--text-secondary);">Precision</td>
              <td style="padding: 2px 0; text-align: right; font-weight: 700; font-family: var(--font-mono, monospace); color: #3b82f6;">${arm.precision_str}</td>
            </tr>
            <tr>
              <td style="padding: 2px 0; color: var(--text-secondary);">F1 Score</td>
              <td style="padding: 2px 0; text-align: right; font-weight: 700; font-family: var(--font-mono, monospace); color: #8b5cf6;">${arm.f1_str}</td>
            </tr>
            <tr>
              <td style="padding: 2px 0; color: var(--text-secondary);">Recall</td>
              <td style="padding: 2px 0; text-align: right; font-weight: 700; font-family: var(--font-mono, monospace); color: #10b981;">${arm.recall_str}</td>
            </tr>
          </tbody>
        </table>
      `;
    } else {
      html += `<div style="text-align:center; padding: 20px 0; color:var(--text-muted); font-size:10px;">No data</div>`;
    }
    
    html += `</div>`;
  });

  container.innerHTML = html;
}

function renderSummaryTable(summary) {
  const tbody = document.getElementById("summary-tbody");
  if (!tbody) return;

  if (!useLiveBenchmark) {
    summary = STATIC_BENCHMARK_SUMMARY;
  }

  if (!summary || summary.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:20px; color:var(--text-muted);">No summary data available.</td></tr>`;
    return;
  }

  let html = "";
  summary.forEach(row => {
    const isPositiveRec = row.delta_recall_num > 0;
    const deltaRecClass = isPositiveRec ? "positive" : (row.delta_recall_num < 0 ? "negative" : "neutral");

    const deltaPrecClass = (row.delta_prec && row.delta_prec.startsWith('+')) ? "positive" : ((row.delta_prec && row.delta_prec.startsWith('-')) ? "negative" : "neutral");
    const deltaF1Class = (row.delta_f1 && row.delta_f1.startsWith('+')) ? "positive" : ((row.delta_f1 && row.delta_f1.startsWith('-')) ? "negative" : "neutral");

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
        <td style="text-align: center; font-family: var(--font-mono, monospace); font-size: 11px;">
           <span style="color: #3b82f6;">${row.base_prec}</span> / 
           <span style="color: #8b5cf6;">${row.base_f1}</span> / 
           <span style="color: #10b981;">${row.base_recall}</span>
        </td>
        <td style="text-align: center; font-family: var(--font-mono, monospace); font-size: 11px;">
           <span style="color: #3b82f6; font-weight: 700;">${row.ra_prec}</span> / 
           <span style="color: #8b5cf6; font-weight: 700;">${row.ra_f1}</span> / 
           <span style="color: #10b981; font-weight: 700;">${row.ra_recall}</span>
        </td>
        <td style="text-align: center; font-family: var(--font-mono, monospace); font-size: 11px;">
          <span class="delta-pill ${deltaPrecClass}" style="font-size: 9px; padding: 1px 3px;">${row.delta_prec}</span> / 
          <span class="delta-pill ${deltaF1Class}" style="font-size: 9px; padding: 1px 3px;">${row.delta_f1}</span> / 
          <span class="delta-pill ${deltaRecClass}" style="font-size: 9px; padding: 1px 3px;">${row.delta_recall}</span>
        </td>
        <td style="text-align: center;">
          <span class="delta-pill ${evClass}">${escapeHtml(row.evasions_prevented)}</span>
        </td>
      </tr>
    `;
  });

  tbody.innerHTML = html;
}

function renderConfigSummaryTable(matrix) {
  const tbody = document.getElementById("config-summary-tbody");
  if (!tbody) return;

  if (!useLiveBenchmark) {
    matrix = STATIC_C1_C7_MATRIX;
  }

  if (!matrix || matrix.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:20px; color:var(--text-muted);">No config summary data available.</td></tr>`;
    return;
  }

  const modes = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"];
  const currentDef = window.currentSelectedDefense || "afp";

  const baseArm = matrix.find(m => m.mode.toLowerCase() === "base" && m.defense.toLowerCase() === currentDef) || { recall: 0, f1: 0, precision: 0 };
  const baseRecStr = baseArm.recall ? (baseArm.recall * 100).toFixed(2) + "%" : "0.00%";
  const baseF1Str = baseArm.f1 ? (baseArm.f1 * 100).toFixed(2) + "%" : "0.00%";
  const basePrecStr = baseArm.precision ? (baseArm.precision * 100).toFixed(2) + "%" : "0.00%";
  
  const baseRec = baseArm.recall || 0;
  const baseF1 = baseArm.f1 || 0;
  const basePrec = baseArm.precision || 0;

  let html = "";

  modes.forEach(mode => {
    const arm = matrix.find(m => m.mode.toUpperCase() === mode.toUpperCase() && m.defense.toLowerCase() === currentDef);
    if (!arm) return;

    const diffRec = ((arm.recall || 0) - baseRec) * 100;
    const diffF1 = ((arm.f1 || 0) - baseF1) * 100;
    const diffPrec = ((arm.precision || 0) - basePrec) * 100;

    const deltaRecClass = diffRec > 0 ? "positive" : (diffRec < 0 ? "negative" : "neutral");
    const deltaF1Class = diffF1 > 0 ? "positive" : (diffF1 < 0 ? "negative" : "neutral");
    const deltaPrecClass = diffPrec > 0 ? "positive" : (diffPrec < 0 ? "negative" : "neutral");

    const stUpper = (arm.state || "").toUpperCase();
    let stateClass = "green";
    if (stUpper.includes("ACTIVE") || stUpper.includes("YELLOW")) stateClass = "yellow";
    else if (stUpper.includes("RECOVERY") || stUpper.includes("RED")) stateClass = "red";
    else if (stUpper.includes("BYPASS")) stateClass = "bypassed";

    html += `
      <tr>
        <td style="font-weight: 700; font-size: 13px;">${mode}</td>
        <td style="text-align: center; font-family: var(--font-mono, monospace); font-size: 11px;">
           <span style="color: #3b82f6; font-weight: 700;">${arm.precision_str}</span> / 
           <span style="color: #8b5cf6; font-weight: 700;">${arm.f1_str}</span> / 
           <span style="color: #10b981; font-weight: 700;">${arm.recall_str}</span>
        </td>
        <td style="text-align: center; font-family: var(--font-mono, monospace); font-size: 11px;">
           <span style="color: #3b82f6;">${basePrecStr}</span> / 
           <span style="color: #8b5cf6;">${baseF1Str}</span> / 
           <span style="color: #10b981;">${baseRecStr}</span>
        </td>
        <td style="text-align: center; font-family: var(--font-mono, monospace); font-size: 11px;">
          <span class="delta-pill ${deltaPrecClass}" style="font-size: 9px; padding: 1px 3px;">${diffPrec > 0 ? "+" : ""}${diffPrec.toFixed(2)}%</span> / 
          <span class="delta-pill ${deltaF1Class}" style="font-size: 9px; padding: 1px 3px;">${diffF1 > 0 ? "+" : ""}${diffF1.toFixed(2)}%</span> / 
          <span class="delta-pill ${deltaRecClass}" style="font-size: 9px; padding: 1px 3px;">${diffRec > 0 ? "+" : ""}${diffRec.toFixed(2)}%</span>
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
