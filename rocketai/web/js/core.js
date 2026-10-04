// Shared helpers: formatting, API, toasts, modals, DOM patching.

export const h = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export const fmt = {
  int: (n) => (n == null ? "–" : Math.round(n).toLocaleString("de-DE")),
  num: (n, d = 2) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString("de-DE", { maximumFractionDigits: d, minimumFractionDigits: d })),
  pct: (n) => (n == null ? "–" : `${Math.round(n * 100)} %`),
  steps: (n) => {
    if (n == null) return "–";
    if (n >= 1e9) return `${(n / 1e9).toLocaleString("de-DE", { maximumFractionDigits: 2 })} Mrd.`;
    if (n >= 1e6) return `${(n / 1e6).toLocaleString("de-DE", { maximumFractionDigits: 1 })} Mio.`;
    if (n >= 1e3) return `${Math.round(n / 1e3).toLocaleString("de-DE")} Tsd.`;
    return String(n);
  },
  ago: (t) => {
    if (!t) return "–";
    const s = Math.max(0, Date.now() / 1000 - t);
    if (s < 60) return "gerade eben";
    if (s < 3600) return `vor ${Math.round(s / 60)} min`;
    if (s < 86400) return `vor ${Math.round(s / 3600)} h`;
    return `vor ${Math.round(s / 86400)} Tagen`;
  },
  duration: (s) => {
    if (s == null || !Number.isFinite(s)) return "–";
    if (s < 60) return `${Math.round(s)} s`;
    if (s < 3600) return `${Math.round(s / 60)} min`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
    const d = Math.floor(s / 86400);
    return `${d} Tag${d > 1 ? "e" : ""} ${Math.round((s % 86400) / 3600)} h`;
  },
  clock: (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`,
};

export const STATE_LABEL = {
  new: "Neu", starting: "Startet", running: "Trainiert", evaluating: "Bewertet", stopping: "Stoppt",
  stopped: "Gestoppt", finished: "Fertig", failed: "Fehler", interrupted: "Unterbrochen",
  queued: "Wartet", done: "Fertig", idle: "Bereit", live: "Live",
};
export const pill = (s, label) => `<span class="pill ${h(s)}">${h(label || STATE_LABEL[s] || s)}</span>`;
export const ACTIVE = new Set(["starting", "running", "evaluating", "stopping"]);
export const STAGES = { 1: "Ball treffen", 2: "Tore schießen", 3: "Komplettes Spiel" };
export const OPPONENT_LABEL = { idle: "Stillstand", random: "Zufall", chaser: "Balljäger", defender: "Verteidiger" };
export const opponentLabel = (s) => OPPONENT_LABEL[s] || s;

export const icon = (name) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ""}</svg>`;
const ICONS = {
  play: '<path d="M8 5v14l11-7z" fill="currentColor"/>',
  pause: '<path d="M7 5h4v14H7zm6 0h4v14h-4z" fill="currentColor"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  trash: '<path d="M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13"/>',
  eye: '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  bolt: '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/>',
  chart: '<path d="M4 19V5m0 14h16M8 15l3-4 3 2 5-6"/>',
  check: '<path d="m5 12 5 5 9-10"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  cam: '<path d="M3 8h4l2-3h6l2 3h4v11H3z"/><circle cx="12" cy="13" r="3.5"/>',
  cube: '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12 4 7.5"/>',
  grid: '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M12 4v16M4 12h16"/>',
  brain: '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3 3 3 0 0 0 3-3V7a3 3 0 0 0-3-3zm6 0a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3 3 3 0 0 1-3-3"/>',
  expand: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
};

// ------------------------------------------------------------------ api

export async function api(path, options = {}) {
  const init = { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } };
  if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
  const response = await fetch(path, init);
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!response.ok) {
    const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : data.detail.map?.((d) => d.msg).join(", ") || JSON.stringify(data.detail)) : response.statusText;
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return data;
}

export function toast(title, message = "", kind = "info") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<div><b>${h(title)}</b>${message ? `<span>${h(message)}</span>` : ""}</div><button class="toast-x" aria-label="Schließen">${icon("x")}</button>`;
  el.querySelector("button").addEventListener("click", () => el.remove());
  document.getElementById("toasts").appendChild(el);
  setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 250); }, kind === "error" ? 8000 : 3800);
}

export async function attempt(fn, errorTitle = "Das hat nicht geklappt") {
  try { return await fn(); } catch (error) { toast(errorTitle, error.message, "error"); return null; }
}

export async function waitForJob(id, onDone) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 1000));
    const job = await api(`/api/jobs/${id}`).catch(() => null);
    if (!job) return null;
    if (job.state === "done") return onDone(job);
    if (job.state === "failed") { toast("Auftrag fehlgeschlagen", job.error, "error"); return null; }
  }
}

// ------------------------------------------------------------------ modal

/** Promise-based dialog. fields: [{id, label, type, value, hint, min, step}] → resolves to values or null. */
export function modal({ title, body = "", fields = [], confirm = "OK", cancel = "Abbrechen", danger = false }) {
  return new Promise((resolve) => {
    const root = document.getElementById("modal-root");
    const wrap = document.createElement("div");
    wrap.className = "modal-backdrop";
    wrap.innerHTML = `<div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <h2 id="modal-title">${h(title)}</h2>
      ${body ? `<div class="modal-body">${body}</div>` : ""}
      ${fields.map((f) => `<label class="field"><span>${h(f.label)}</span><input data-field="${h(f.id)}" type="${f.type || "text"}" value="${h(f.value ?? "")}" ${f.min != null ? `min="${f.min}"` : ""} ${f.step ? `step="${f.step}"` : ""}>${f.hint ? `<small>${h(f.hint)}</small>` : ""}</label>`).join("")}
      <div class="row end"><button class="btn ghost" data-r="cancel">${h(cancel)}</button><button class="btn ${danger ? "danger-solid" : "primary"}" data-r="ok">${h(confirm)}</button></div>
    </div>`;
    const close = (result) => { wrap.classList.add("out"); document.removeEventListener("keydown", onKey); setTimeout(() => wrap.remove(), 160); resolve(result); };
    const collect = () => Object.fromEntries([...wrap.querySelectorAll("[data-field]")].map((i) => [i.dataset.field, i.type === "number" ? Number(i.value) : i.value]));
    const onKey = (e) => { if (e.key === "Escape") close(null); if (e.key === "Enter" && e.target.tagName === "INPUT") close(collect()); };
    wrap.addEventListener("click", (e) => {
      if (e.target === wrap) close(null);
      const r = e.target.closest("[data-r]");
      if (r) close(r.dataset.r === "ok" ? collect() : null);
    });
    document.addEventListener("keydown", onKey);
    root.appendChild(wrap);
    (wrap.querySelector("input") || wrap.querySelector("[data-r=ok]")).focus();
  });
}

// ------------------------------------------------------------------ DOM patching

/**
 * Update ``target``'s children to match ``html`` without rebuilding unchanged
 * nodes: scroll positions, focus, typed text, open <details> and canvases
 * survive. Elements with ``data-static`` keep their children untouched.
 */
export function patch(target, html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  morphChildren(target, template.content);
}

function keyOf(node) {
  return node.nodeType === 1 ? node.getAttribute("data-key") : null;
}

function morphChildren(from, to) {
  const next = [...to.childNodes];
  let current = from.firstChild;
  for (const want of next) {
    const key = keyOf(want);
    if (key != null) {
      // find a keyed match further along
      let scan = current;
      while (scan && keyOf(scan) !== key) scan = scan.nextSibling;
      if (scan && scan !== current) from.insertBefore(scan, current);
      if (scan) current = scan;
    }
    if (!current) { from.appendChild(want); continue; }
    if (sameKind(current, want)) {
      morph(current, want);
      current = current.nextSibling;
    } else {
      const replaced = current;
      current = current.nextSibling;
      from.replaceChild(want, replaced);
    }
  }
  while (current) { const n = current.nextSibling; from.removeChild(current); current = n; }
}

function sameKind(a, b) {
  if (a.nodeType !== b.nodeType || a.nodeName !== b.nodeName) return false;
  if (a.nodeType === 1 && keyOf(a) !== keyOf(b)) return false;
  return true;
}

function morph(from, to) {
  if (from.nodeType === 3 || from.nodeType === 8) {
    if (from.nodeValue !== to.nodeValue) from.nodeValue = to.nodeValue;
    return;
  }
  if (from.nodeType !== 1) return;
  const focused = document.activeElement === from;
  for (const { name, value } of [...to.attributes]) {
    if (name === "value" && (focused || from.dataset.dirty)) continue;
    if (name === "open" && from.tagName === "DETAILS") continue;
    if (from.getAttribute(name) !== value) from.setAttribute(name, value);
  }
  for (const { name } of [...from.attributes]) {
    if (name === "open" && from.tagName === "DETAILS") continue;
    if (name === "data-dirty") continue;
    if (!to.hasAttribute(name)) from.removeAttribute(name);
  }
  if (from.tagName === "INPUT" || from.tagName === "TEXTAREA") {
    if (!focused && !from.dataset.dirty && to.hasAttribute("value") && from.value !== to.getAttribute("value")) from.value = to.getAttribute("value");
    if (from.type === "checkbox") from.checked = to.hasAttribute("checked");
    return;
  }
  if (from.tagName === "SELECT") {
    const chosen = from.value;
    morphChildren(from, to);
    if (focused || from.dataset.dirty) from.value = chosen;
    else { const sel = to.querySelector("option[selected]"); if (sel) from.value = sel.value; }
    return;
  }
  if (from.hasAttribute("data-static")) return;
  morphChildren(from, to);
}

// ------------------------------------------------------------------ misc

export function groupBy(list, key) {
  return list.reduce((acc, item) => { (acc[key(item)] ||= []).push(item); return acc; }, {});
}

export function smooth(values, window = 5) {
  return values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - window + 1), i + 1).filter((v) => v != null);
    return slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : null;
  });
}

let chartId = 0;
/** Small SVG line chart; ``marks`` = [[x, label]] vertical guides (e.g. checkpoints). */
export function lineChart(points, { height = 170, format = (v) => fmt.num(v, 1), marks = [] } = {}) {
  const clean = points.filter((p) => p[1] != null && Number.isFinite(p[1]));
  if (clean.length < 2) return `<div class="chart-empty">Noch zu wenige Daten – die Kurve erscheint nach ein paar Iterationen.</div>`;
  const W = 600, H = height, L = 46, R = 10, T = 10, B = 22;
  const xs = clean.map((p) => p[0]), ys = clean.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
  const sx = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const path = clean.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
  const area = `${path}L${sx(x1).toFixed(1)},${H - B}L${sx(x0).toFixed(1)},${H - B}Z`;
  const id = `fade${++chartId}`;
  let grid = "";
  for (let i = 0; i <= 3; i++) {
    const v = y0 + ((y1 - y0) * i) / 3, y = sy(v);
    grid += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text class="axis" x="${L - 8}" y="${y + 3}" text-anchor="end">${h(format(v))}</text>`;
  }
  for (const [mx, label, cls = ""] of marks) {
    if (mx < x0 || mx > x1) continue;
    grid += `<line class="mark-line ${cls}" x1="${sx(mx)}" x2="${sx(mx)}" y1="${T}" y2="${H - B}">${label ? `<title>${label}</title>` : ""}</line>`;
    if (cls === "stage") grid += `<text class="mark-text" x="${sx(mx) + 4}" y="${T + 10}">${label}</text>`;
  }
  grid += `<text class="axis" x="${L}" y="${H - 4}">${fmt.steps(x0)}</text><text class="axis" x="${W - R}" y="${H - 4}" text-anchor="end">${fmt.steps(x1)}</text>`;
  const lastPoint = clean[clean.length - 1];
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c6f432" stop-opacity=".2"/><stop offset="1" stop-color="#c6f432" stop-opacity="0"/></linearGradient></defs>
    ${grid}<path d="${area}" fill="url(#${id})"/><path class="line" d="${path}" vector-effect="non-scaling-stroke"/>
    <circle class="dot" cx="${sx(lastPoint[0])}" cy="${sy(lastPoint[1])}" r="3.5"/></svg>`;
}

export function sparkline(values, { width = 220, height = 44, min = null, max = null } = {}) {
  const clean = values.filter((v) => Number.isFinite(v));
  if (clean.length < 2) return `<svg class="spark" viewBox="0 0 ${width} ${height}"></svg>`;
  const lo = min ?? Math.min(...clean), hi = max ?? Math.max(...clean);
  const span = hi - lo || 1;
  const pts = clean.map((v, i) => `${((i / (clean.length - 1)) * width).toFixed(1)},${(height - 3 - ((v - lo) / span) * (height - 6)).toFixed(1)}`);
  return `<svg class="spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><polyline points="${pts.join(" ")}" vector-effect="non-scaling-stroke"/></svg>`;
}
