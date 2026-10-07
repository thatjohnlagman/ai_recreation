function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    const text = document.createTextNode(str);
    const p = document.createElement('p');
    p.appendChild(text);
    return p.innerHTML;
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
