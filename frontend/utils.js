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

window.addEventListener('storage', (e) => {
    if (e.key === 'app_theme' && e.newValue) {
        if (typeof applyTheme === 'function') {
            applyTheme(e.newValue);
        } else {
            document.documentElement.setAttribute('data-theme', e.newValue);
        }
    }
});
