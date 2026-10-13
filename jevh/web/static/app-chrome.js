// The desktop app's own title bar (jevh/app.py, jevh/winchrome.py): inside the app's frameless window only, the page's
// headers drag the window (app-region: drag, which WebView2 hands to the window) and three small buttons sit in the top
// right corner: minimize, maximize/restore, close (to the tray). Its resize border takes the page's background colour.
// In a browser, or on the phone, it does nothing.
(function () {
  const api = () => (window.pywebview && window.pywebview.api) || null;
  const ICONS = {
    min: '<path d="M2 6.5h8"/>',
    max: '<rect x="2" y="2" width="8" height="8" rx="1"/>',
    restore: '<rect x="2" y="4" width="6" height="6" rx="1"/><path d="M4.5 4V3a1 1 0 0 1 1-1H9a1 1 0 0 1 1 1v3.5a1 1 0 0 1-1 1H8"/>',
    close: '<path d="M2.5 2.5l7 7M9.5 2.5l-7 7"/>',
  };
  const svg = (k) => `<svg viewBox="0 0 12 12" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.1" stroke-linecap="round">${ICONS[k]}</svg>`;
  const CSS = `
    html[data-app] .app-drag, html[data-app] .topbar, html[data-app] .sidehead, html[data-app] .rail .logo { app-region: drag; -webkit-app-region: drag; }
    html[data-app] .topbar :is(button, input, select, a, label, [role=button]),
    html[data-app] .sidehead :is(button, input, select, a, label, [role=button]),
    html[data-app] .app-drag :is(button, input, select, a) { app-region: no-drag; -webkit-app-region: no-drag; }
    html[data-app] .app[data-panel="closed"] .topbar, html[data-app] .app:not([data-panel]) .topbar { padding-right: 150px; }
    html[data-app] .app[data-panel="open"] .sidehead { padding-right: 150px; }
    #winbtns { position: fixed; top: 0; right: 0; z-index: 2147483000; display: flex; app-region: no-drag; -webkit-app-region: no-drag; }
    #winbtns button { width: 46px; height: 34px; border: 0; margin: 0; padding: 0; background: transparent; color: inherit;
      opacity: .75; display: grid; place-items: center; cursor: default; border-radius: 0; box-shadow: none; }
    #winbtns button:hover { background: rgba(128, 128, 128, .18); opacity: 1; }
    #winbtns button[data-w="close"]:hover { background: #c42b1c; color: #fff; }
    #winbtns button:focus-visible { outline: 1px solid currentColor; outline-offset: -3px; }`;

  function hex(c) {   // "rgb(18, 21, 27)" -> "#12151b" (the window's border wants a plain colour)
    const m = String(c).match(/\d+(\.\d+)?/g);
    if (!m || m.length < 3) return null;
    return "#" + m.slice(0, 3).map((v) => Math.round(+v).toString(16).padStart(2, "0")).join("");
  }
  let lastBorder = "";
  function border() {
    const b = getComputedStyle(document.body).backgroundColor, h = hex(b);
    const a = (String(b).match(/[\d.]+/g) || [])[3];
    if (!h || (a !== undefined && +a < 0.5)) return;   // (a see-through body: its colour is not what shows)
    if (h !== lastBorder) { lastBorder = h; try { api().win_border(h); } catch {} }
  }
  async function state(btns) {
    let st = {};
    try { st = await api().win_state(); } catch {}
    const max = btns.querySelector('[data-w="max"]');
    max.innerHTML = svg(st.maximized ? "restore" : "max");
    max.title = st.maximized ? "Restore" : "Maximize";
    return st;
  }
  async function init() {
    if (!api() || !api().win_state || document.documentElement.dataset.app) return;
    let st;
    try { st = await api().win_state(); } catch { return; }
    if (!st || !st.frameless) return;
    document.documentElement.dataset.app = "1";
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
    const btns = document.createElement("div");
    btns.id = "winbtns";
    btns.innerHTML = `<button type="button" data-w="min" title="Minimize" aria-label="Minimize">${svg("min")}</button>` +
      `<button type="button" data-w="max" aria-label="Maximize or restore">${svg("max")}</button>` +
      `<button type="button" data-w="close" title="Close to the tray" aria-label="Close">${svg("close")}</button>`;
    document.body.appendChild(btns);
    btns.addEventListener("click", async (ev) => {
      const b = ev.target.closest("[data-w]");
      if (!b) return;
      const w = b.dataset.w;
      try {
        if (w === "min") api().win_minimize();
        else if (w === "max") { await api().win_maximize(); setTimeout(() => state(btns), 250); }
        else if (w === "close") api().win_close();
      } catch {}
    });
    window.addEventListener("resize", () => { clearTimeout(init.t); init.t = setTimeout(() => state(btns), 200); });
    state(btns);
    border();
    setInterval(border, 2000);   // (a theme change in Appearance)
  }
  window.addEventListener("pywebviewready", init);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
