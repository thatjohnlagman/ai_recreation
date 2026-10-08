/**
 * utils.js - Unified SOC Platform Utilities & Global Theme Controller
 * Synchronizes theme (Dark/Light), UTC clock, and formatting across all tabs.
 */

// -----------------------------------------------------------------------------
// 1. Immediate Unified Theme Initialization (Prevents flash of wrong theme)
// -----------------------------------------------------------------------------
(function initImmediateTheme() {
  try {
    const saved = localStorage.getItem("app_theme") || localStorage.getItem("soc_theme") || "dark";
    document.documentElement.setAttribute("data-theme", saved);
  } catch (e) {
    document.documentElement.setAttribute("data-theme", "dark");
  }
})();

// -----------------------------------------------------------------------------
// 2. Global Theme Management API
// -----------------------------------------------------------------------------
function getTheme() {
  return document.documentElement.getAttribute("data-theme") || 
         localStorage.getItem("app_theme") || 
         localStorage.getItem("soc_theme") || 
         "dark";
}

function applyTheme(theme) {
  const targetTheme = (theme === "light") ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", targetTheme);
  
  try {
    localStorage.setItem("app_theme", targetTheme);
    localStorage.setItem("soc_theme", targetTheme);
  } catch (e) {}

  // Update toggle button tooltip if present
  const btn = document.getElementById("theme-toggle-btn");
  if (btn) {
    const nextLabel = (targetTheme === "light") ? "Switch to Dark Mode" : "Switch to Light Mode";
    btn.setAttribute("title", nextLabel);
    btn.setAttribute("aria-label", nextLabel);
  }

  // Dispatch custom event for Chart.js / Leaflet / Component re-rendering
  window.dispatchEvent(new CustomEvent("themeChanged", {
    detail: { theme: targetTheme, isDark: targetTheme === "dark" }
  }));
}

function toggleTheme() {
  const current = getTheme();
  const next = (current === "light") ? "dark" : "light";
  applyTheme(next);
}

// -----------------------------------------------------------------------------
// 3. Global Cross-Tab Storage Listener
// -----------------------------------------------------------------------------
window.addEventListener("storage", (e) => {
  if ((e.key === "app_theme" || e.key === "soc_theme") && e.newValue) {
    const current = document.documentElement.getAttribute("data-theme");
    if (current !== e.newValue) {
      document.documentElement.setAttribute("data-theme", e.newValue);
      window.dispatchEvent(new CustomEvent("themeChanged", {
        detail: { theme: e.newValue, isDark: e.newValue === "dark" }
      }));
    }
  }
});

// -----------------------------------------------------------------------------
// 4. Global DOM Ready Auto-Init (Theme Toggle Button & UTC Clock)
// -----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Bind Theme Toggle Button if present and not already bound
  const themeBtn = document.getElementById("theme-toggle-btn");
  if (themeBtn && !themeBtn.dataset.themeBound) {
    themeBtn.dataset.themeBound = "true";
    themeBtn.addEventListener("click", toggleTheme);
    const curr = getTheme();
    themeBtn.setAttribute("title", curr === "light" ? "Switch to Dark Mode" : "Switch to Light Mode");
  }

  // Auto-init UTC Clock if live-datetime element exists
  initGlobalLiveClock();
});

function initGlobalLiveClock() {
  const clockEl = document.getElementById("live-datetime");
  if (!clockEl) return;
  const update = () => {
    const now = new Date();
    clockEl.textContent = now.toUTCString().replace("GMT", "UTC");
  };
  update();
  setInterval(update, 1000);
}

// -----------------------------------------------------------------------------
// 5. Formatting & Utility Functions
// -----------------------------------------------------------------------------
function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  const text = document.createTextNode(str);
  const p = document.createElement('p');
  p.appendChild(text);
  return p.innerHTML;
}

function formatLocalTime(isoStr) {
  if (!isoStr) return 'N/A';
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return isoStr;
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true });
  } catch (e) {
    return isoStr;
  }
}

function formatLocalDateTime(isoStr) {
  if (!isoStr) return 'N/A';
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return isoStr;
    return d.toLocaleString([], { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true });
  } catch (e) {
    return isoStr;
  }
}
