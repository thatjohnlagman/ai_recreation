// Export Modal Logic
const EXPORT_MODAL_HTML = `
<div id="export-modal" class="modal-overlay">
  <div class="modal-content" style="max-width: 800px; width: 90%;">
    <span class="modal-close" onclick="closeExportModal()">&times;</span>
    <h3 class="modal-title" style="margin-bottom: 20px;">Export Data</h3>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
      <div>
        <label style="display:block; font-size:12px; font-weight:bold; color:var(--text-muted); margin-bottom:8px; text-transform:uppercase;">Export Scope</label>
        <select id="export-scope" onchange="updateExportPreview()" style="width:100%; padding:10px; border-radius:6px; border:1px solid var(--border-color); background:var(--bg-primary); color:var(--text-primary);">
          <option value="filtered">Current Filtered Results</option>
          <option value="session">Current Session</option>
          <option value="all">All Recorded History (Explicit)</option>
        </select>
      </div>
      <div>
        <label style="display:block; font-size:12px; font-weight:bold; color:var(--text-muted); margin-bottom:8px; text-transform:uppercase;">Export Type</label>
        <select id="export-type" onchange="updateExportPreview()" style="width:100%; padding:10px; border-radius:6px; border:1px solid var(--border-color); background:var(--bg-primary); color:var(--text-primary);">
          <option value="metadata">Event Metadata</option>
          <option value="features">Feature Details</option>
          <option value="sessions">Session Summary</option>
        </select>
      </div>
    </div>

    <div style="margin-bottom: 20px;">
      <p id="export-summary-text" style="font-size:14px; color:var(--text-primary); margin-bottom: 12px; font-weight:bold;"></p>
      
      <div style="max-height: 250px; overflow: auto; border: 1px solid var(--border-color); border-radius: var(--radius-md);">
        <table id="export-preview-table" class="data-table" style="width: 100%; border-collapse: collapse; font-size: 11px;">
          <thead id="export-preview-head"></thead>
          <tbody id="export-preview-body"></tbody>
        </table>
      </div>
    </div>

    <div style="display: flex; justify-content: flex-end; align-items: center; gap: 15px; margin-top: 20px; border-top: 1px solid var(--border-color); padding-top: 15px;">
      <span id="export-filename" style="font-size:12px; color:var(--text-muted); font-family: monospace;"></span>
      <button onclick="downloadExport()" style="background: #10b981; color: white; border: none; padding: 10px 20px; border-radius: var(--radius-md); font-weight: bold; cursor: pointer;">Download CSV</button>
    </div>
  </div>
</div>
`;

// Close on Escape or Outside Click
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('export-modal') && document.getElementById('export-modal').style.display === 'flex') {
        closeExportModal();
    }
});

document.addEventListener('click', (e) => {
    const modal = document.getElementById('export-modal');
    if (modal && e.target === modal) {
        closeExportModal();
    }
});

document.addEventListener("DOMContentLoaded", () => {
    const div = document.createElement('div');
    div.innerHTML = EXPORT_MODAL_HTML;
    document.body.appendChild(div.firstElementChild);
});

let currentExportParams = "";
let currentExportEndpoint = "";
let currentFilename = "";

function openExportModal(defaultType = 'metadata', defaultScope = 'filtered', session_id = null) {
    document.getElementById('export-modal').style.display = 'flex';
    document.getElementById('export-type').value = defaultType;
    
    const scopeSelect = document.getElementById('export-scope');
    
    // Set up session option
    let hasSession = false;
    let urlSession = null;
    
    if (session_id) {
        urlSession = session_id;
        hasSession = true;
    } else {
        const urlParams = new URLSearchParams(window.location.search);
        urlSession = urlParams.get('session_id');
        hasSession = !!urlSession;
    }
    
    // If we have a specific session passed or in URL, lock it to session or update options
    Array.from(scopeSelect.options).forEach(opt => {
        if (opt.value === 'session') {
            if (hasSession) {
                opt.style.display = 'block';
                opt.setAttribute('data-session', urlSession);
            } else {
                opt.style.display = 'none';
            }
        }
    });

    if (hasSession && defaultScope === 'session') {
        scopeSelect.value = 'session';
    } else {
        scopeSelect.value = defaultScope === 'session' ? 'filtered' : defaultScope;
    }
    
    // Prevent Session Summary export from Traffic Explorer
    const typeSelect = document.getElementById('export-type');
    const isExplorer = window.location.pathname.includes('/explorer');
    Array.from(typeSelect.options).forEach(opt => {
        if (opt.value === 'sessions') {
            opt.style.display = isExplorer ? 'none' : 'block';
        }
    });
    
    if (isExplorer && typeSelect.value === 'sessions') {
        typeSelect.value = 'metadata';
    }

    updateExportPreview();
}

function closeExportModal() {
    document.getElementById('export-modal').style.display = 'none';
}

function getActiveFilters() {
    let q = [];
    const filterRole = document.getElementById('filter-role');
    const filterAction = document.getElementById('filter-action');
    const filterAttack = document.getElementById('filter-attack');
    const filterDefense = document.getElementById('filter-defense');
    
    if (filterRole && filterRole.value) {
        q.push(`role=${encodeURIComponent(filterRole.value)}`);
        if (filterRole.value === "") { // "All Roles" logic: include queries
            q.push(`include_queries=true`);
        } else if (filterRole.value === "Query") {
            q.push(`query_only=true`);
        }
    } else {
        q.push(`include_queries=true`);
    }
    if (filterAction && filterAction.value) q.push(`action=${encodeURIComponent(filterAction.value)}`);
    if (filterAttack && filterAttack.value) q.push(`is_attack=${encodeURIComponent(filterAttack.value)}`);
    if (filterDefense && filterDefense.value) q.push(`defense=${encodeURIComponent(filterDefense.value)}`);
    // Note: session_id can be in filters
    return q.join("&");
}

async function updateExportPreview() {
    const scope = document.getElementById('export-scope').value;
    const type = document.getElementById('export-type').value;
    
    let queryParams = "";
    let summaryText = "";
    
    if (scope === 'filtered') {
        queryParams = getActiveFilters();
        
        // Include session ID from URL if it exists
        const urlParams = new URLSearchParams(window.location.search);
        const sid = urlParams.get('session_id');
        if (sid && !queryParams.includes('session_id=')) {
            queryParams += (queryParams ? "&" : "") + `session_id=${encodeURIComponent(sid)}`;
        }
        
        summaryText = `Scope: Filtered Results\nFilters applied: ${queryParams || 'None'}`;
    } else if (scope === 'session') {
        const opt = document.querySelector('#export-scope option[value="session"]');
        const sid = opt.getAttribute('data-session');
        queryParams = `session_id=${encodeURIComponent(sid)}`;
        summaryText = `Scope: Session\nSelected session ID: ${sid}`;
    } else if (scope === 'all') {
        summaryText = "Scope: ALL recorded history\nWarning: Export can be very large";
    }

    let endpoint = "";
    let filename = "";
    
    if (type === 'metadata') {
        endpoint = "/api/history/export";
        filename = "traffic_metadata.csv";
    } else if (type === 'features') {
        endpoint = "/api/history/export/features";
        filename = "feature_details.csv";
    } else if (type === 'sessions') {
        endpoint = "/api/history/export/sessions";
        filename = "session_summary.csv";
    }
    
    currentFilename = filename;
    currentExportEndpoint = endpoint;
    currentExportParams = queryParams;
    
    document.getElementById('export-filename').innerText = filename;
    document.getElementById('export-summary-text').innerText = `${summaryText}\nType: ${type}\n(Previewing first 5 matching events)`;
    
    const previewUrl = endpoint + "?preview=true" + (queryParams ? "&" + queryParams : "");
    
    const headEl = document.getElementById('export-preview-head');
    const bodyEl = document.getElementById('export-preview-body');
    
    headEl.innerHTML = "";
    let th = document.createElement('th');
    th.textContent = "Loading preview...";
    let tr = document.createElement('tr');
    tr.appendChild(th);
    headEl.appendChild(tr);
    
    bodyEl.innerHTML = "";

    try {
        const res = await fetch(previewUrl);
        if (!res.ok) throw new Error("Failed to fetch preview");
        const text = await res.text();
        
        const lines = text.trim().split('\n');
        if (lines.length === 0 || text === "No data") {
            headEl.innerHTML = "";
            let th = document.createElement('th');
            th.textContent = "No data found";
            let tr = document.createElement('tr');
            tr.appendChild(th);
            headEl.appendChild(tr);
            return;
        }
        
        // Basic CSV split
        const headers = lines[0].split(',');
        headEl.innerHTML = "";
        let headRow = document.createElement('tr');
        headers.forEach(h => {
            let th = document.createElement('th');
            th.style.padding = '8px';
            th.style.borderBottom = '1px solid var(--border-color)';
            th.textContent = h.replace(/"/g,'');
            headRow.appendChild(th);
        });
        headEl.appendChild(headRow);
        
        bodyEl.innerHTML = "";
        for (let i = 1; i < lines.length; i++) {
            if (!lines[i]) continue;
            // Handle quotes properly in a naive way for preview
            const cols = lines[i].split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);
            let bodyRow = document.createElement('tr');
            cols.forEach(c => {
                let val = c.replace(/^"|"$/g, '').trim();
                if (val.length > 50) val = val.substring(0, 50) + "...";
                let td = document.createElement('td');
                td.style.padding = '8px';
                td.style.borderBottom = '1px solid var(--border-color)';
                td.textContent = val;
                bodyRow.appendChild(td);
            });
            bodyEl.appendChild(bodyRow);
        }
        
    } catch (err) {
        headEl.innerHTML = "";
        let th = document.createElement('th');
        th.style.color = 'red';
        th.textContent = `Error: ${err.message}`;
        let tr = document.createElement('tr');
        tr.appendChild(th);
        headEl.appendChild(tr);
    }
}

function downloadExport() {
    const url = currentExportEndpoint + "?" + currentExportParams;
    
    // Create an invisible anchor to trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = currentFilename; // Optional, usually header takes precedence
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    closeExportModal();
}
