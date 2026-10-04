// RocketAI web app — no framework, no build step, works offline.

const view = document.getElementById("view");
const state = { snapshot: null, cleanup: [], route: "", onSnapshot: null };

// ------------------------------------------------------------------ utils

const h = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const fmt = {
  int: (n) => (n == null ? "–" : Math.round(n).toLocaleString("de-DE")),
  num: (n, d = 2) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString("de-DE", { maximumFractionDigits: d, minimumFractionDigits: d })),
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
    const hours = Math.floor(s / 3600);
    return `${hours} h ${Math.round((s % 3600) / 60)} min`;
  },
  clock: (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`,
};

const STATE_LABEL = {
  new: "Neu", starting: "Startet", running: "Trainiert", evaluating: "Bewertet", stopping: "Stoppt",
  stopped: "Gestoppt", finished: "Fertig", failed: "Fehler", interrupted: "Unterbrochen",
  queued: "Wartet", done: "Fertig", idle: "Bereit",
};
const pill = (s) => `<span class="pill ${h(s)}">${h(STATE_LABEL[s] || s)}</span>`;
const ACTIVE = new Set(["starting", "running", "evaluating", "stopping"]);
const STAGES = { 1: "Ball treffen", 2: "Tore schießen", 3: "Komplettes Spiel" };

async function api(path, options = {}) {
  const init = { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } };
  if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
  const response = await fetch(path, init);
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!response.ok) {
    const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : response.statusText;
    throw new Error(detail);
  }
  return data;
}

function toast(title, message = "", kind = "info") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<b>${h(title)}</b>${message ? `<span>${h(message)}</span>` : ""}`;
  document.getElementById("toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 3500);
}

async function attempt(fn, errorTitle = "Das hat nicht geklappt") {
  try { return await fn(); } catch (error) { toast(errorTitle, error.message, "error"); return null; }
}

async function waitForJob(id, onDone) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 1000));
    const job = await api(`/api/jobs/${id}`).catch(() => null);
    if (!job) return;
    if (job.state === "done") return onDone(job);
    if (job.state === "failed") return toast("Auftrag fehlgeschlagen", job.error, "error");
  }
}

// ------------------------------------------------------------------ charts

function lineChart(points, { height = 170, format = (v) => fmt.num(v, 1) } = {}) {
  const clean = points.filter((p) => p[1] != null && Number.isFinite(p[1]));
  if (clean.length < 2) return `<div class="empty" style="padding:48px 12px">Noch zu wenige Daten</div>`;
  const W = 600, H = height, L = 44, R = 8, T = 10, B = 22;
  const xs = clean.map((p) => p[0]), ys = clean.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
  const sx = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const path = clean.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
  const area = `${path}L${sx(x1).toFixed(1)},${H - B}L${sx(x0).toFixed(1)},${H - B}Z`;
  let grid = "";
  for (let i = 0; i <= 3; i++) {
    const v = y0 + ((y1 - y0) * i) / 3, y = sy(v);
    grid += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text class="axis" x="${L - 8}" y="${y + 3}" text-anchor="end">${h(format(v))}</text>`;
  }
  grid += `<text class="axis" x="${L}" y="${H - 4}">${fmt.steps(x0)}</text><text class="axis" x="${W - R}" y="${H - 4}" text-anchor="end">${fmt.steps(x1)}</text>`;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <defs><linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c6f432" stop-opacity=".18"/><stop offset="1" stop-color="#c6f432" stop-opacity="0"/></linearGradient></defs>
    ${grid}<path class="area" d="${area}"/><path class="line" d="${path}" vector-effect="non-scaling-stroke"/></svg>`;
}

function smooth(values, window = 5) {
  return values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - window + 1), i + 1).filter((v) => v != null);
    return slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : null;
  });
}

// ------------------------------------------------------------------ live

function connectEvents() {
  const live = document.getElementById("live"), text = document.getElementById("live-text");
  const source = new EventSource("/api/events");
  source.onopen = () => { live.classList.add("on"); text.textContent = "Verbunden"; };
  source.onerror = () => { live.classList.remove("on"); text.textContent = "Getrennt – verbinde neu …"; };
  source.onmessage = (event) => {
    state.snapshot = JSON.parse(event.data);
    const active = state.snapshot.runs.filter((r) => ACTIVE.has(r.state));
    text.textContent = active.length ? `${active.length} Training${active.length > 1 ? "s" : ""} aktiv` : "Verbunden";
    if (state.onSnapshot) state.onSnapshot(state.snapshot);
  };
}

// ------------------------------------------------------------------ router

const routes = [
  [/^\/?$/, "overview", renderOverview],
  [/^\/training\/?$/, "training", renderTraining],
  [/^\/training\/new\/?$/, "training", renderNewRun],
  [/^\/training\/([^/]+)\/?$/, "training", renderRun],
  [/^\/arena\/?$/, "arena", renderArena],
  [/^\/arena\/(.+)$/, "arena", renderArena],
  [/^\/play\/?$/, "play", renderPlay],
  [/^\/setup\/?$/, "setup", renderSetup],
];

async function router() {
  state.cleanup.forEach((fn) => fn());
  state.cleanup = [];
  state.onSnapshot = null;
  const path = decodeURIComponent(location.hash.replace(/^#/, "").split("?")[0]);
  const query = new URLSearchParams(location.hash.split("?")[1] || "");
  for (const [pattern, nav, render] of routes) {
    const match = path.match(pattern);
    if (!match) continue;
    document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === nav));
    view.innerHTML = "";
    view.classList.remove("fade-in"); void view.offsetWidth; view.classList.add("fade-in");
    try { await render(match[1], query); } catch (error) {
      view.innerHTML = `<div class="empty"><h3>Seite konnte nicht geladen werden</h3><p>${h(error.message)}</p></div>`;
    }
    window.scrollTo(0, 0);
    return;
  }
  location.hash = "#/";
}

// ------------------------------------------------------------------ overview

function runCard(run) {
  const st = run.status || {}, cfg = run.config || {};
  const total = st.total_steps || cfg.total_steps || 1;
  const pct = Math.min(100, ((st.steps || 0) / total) * 100);
  const evalRes = run.evaluation && run.evaluation.results ? run.evaluation.results : [];
  const score = evalRes.length ? evalRes.map((r) => `${Math.round(r.score * 100)}%`).join(" · ") : "–";
  return `<a class="card link run-card" href="#/training/${encodeURIComponent(run.name)}">
    <div class="top"><div><div class="name">${h(run.name)}</div>
      <div class="meta">${cfg.team_size}v${cfg.team_size} · Stufe ${cfg.reward_stage}: ${h(STAGES[cfg.reward_stage] || "")}</div></div>${pill(st.state)}</div>
    <div><div class="progress"><i style="width:${pct}%"></i></div>
      <div class="progress-meta"><span>${fmt.steps(st.steps || 0)} / ${fmt.steps(total)} Schritte</span><span>${pct.toFixed(1)}%</span></div></div>
    <div class="kv">
      <div><span>Ballkontakte/min</span><b>${fmt.num(run.last.touches_per_minute, 1)}</b></div>
      <div><span>Tempo</span><b>${run.last.steps_per_second ? fmt.int(run.last.steps_per_second) + "/s" : "–"}</b></div>
      <div><span>Bewertung</span><b>${score}</b></div>
    </div></a>`;
}

async function renderOverview() {
  const data = await api("/api/overview");
  const runs = data.runs;
  const active = runs.filter((r) => ACTIVE.has(r.status.state));
  const totalSteps = runs.reduce((sum, r) => sum + (r.status.steps || 0), 0);
  const checkpoints = runs.reduce((sum, r) => sum + r.checkpoints, 0);
  const speed = active.reduce((sum, r) => sum + (r.status.steps_per_second || 0), 0);
  view.innerHTML = `
    <div class="page-head"><div><div class="eyebrow">Übersicht</div><h1>Deine Rocket-League-KI</h1>
      <p>Trainiere in der Simulation, sieh dir Spiele in der Arena an und lass sie im echten Rocket League gegen Bots oder dich antreten.</p></div>
      <a class="btn primary" href="#/training/new">Neues Training</a></div>
    <div class="grid cols-4">
      <div class="card stat"><div class="label">Aktive Trainings</div><div class="value">${active.length}</div><div class="foot">${speed ? fmt.int(speed) + " Schritte/s" : "nichts läuft"}</div></div>
      <div class="card stat"><div class="label">Trainierte Schritte</div><div class="value">${fmt.steps(totalSteps)}</div><div class="foot">über ${runs.length} Run${runs.length === 1 ? "" : "s"}</div></div>
      <div class="card stat"><div class="label">Checkpoints</div><div class="value">${checkpoints}</div><div class="foot">spielbare KI-Versionen</div></div>
      <div class="card stat"><div class="label">Echtes Spiel</div><div class="value" style="font-size:18px;margin-top:12px">${pill(data.play.state)}</div><div class="foot">${h(data.play.message || "über RLBot, nur offline")}</div></div>
    </div>
    <div class="section"><div class="section-head"><h2>Trainings</h2><a class="btn ghost sm" href="#/training">Alle ansehen</a></div>
      ${runs.length ? `<div class="grid cols-3">${runs.slice(0, 6).map(runCard).join("")}</div>` : `
      <div class="empty"><h3>Noch kein Training</h3><p>Starte mit dem Schnelltest (2 Minuten), um zu prüfen, ob alles läuft.</p>
      <a class="btn primary" href="#/training/new">Erstes Training anlegen</a></div>`}
    </div>
    <div class="section grid cols-2">
      <div class="card"><div class="card-head"><h2>Letzte Replays</h2><a class="btn ghost sm" href="#/arena">Arena</a></div>
        ${data.replays.length ? `<div class="replay-list">${data.replays.map(replayItem).join("")}</div>` : `<p class="faint">Replays entstehen bei jeder Bewertung während des Trainings oder über die Arena.</p>`}</div>
      <div class="card"><div class="card-head"><h2>So geht's weiter</h2></div>
        <ol class="steps-list">
          <li><b>Trainieren</b>: Stufe 1 lehrt Ballkontakt, Stufe 2 Tore, Stufe 3 das ganze Spiel.</li>
          <li><b>Bewerten</b>: Die KI spielt automatisch gegen Balljäger und Verteidiger.</li>
          <li><b>Anschauen</b>: In der Arena jedes Spiel als 2D-Replay abspielen.</li>
          <li><b>Spielen</b>: Checkpoint auswählen und in Rocket League offline antreten lassen.</li>
        </ol></div>
    </div>`;
  state.onSnapshot = debounceReload(renderOverview);
}

function debounceReload(fn) {
  let last = JSON.stringify(state.snapshot);
  return (snapshot) => {
    const next = JSON.stringify(snapshot);
    if (next !== last) { last = next; fn(); }
  };
}

// ------------------------------------------------------------------ training list

async function renderTraining() {
  const runs = await api("/api/runs");
  view.innerHTML = `
    <div class="page-head"><div><div class="eyebrow">Training</div><h1>Trainings</h1>
      <p>Jedes Training ist ein eigener Ordner unter <code>runs/</code> mit Einstellungen, Checkpoints, Messwerten und Replays.</p></div>
      <a class="btn primary" href="#/training/new">Neues Training</a></div>
    ${runs.length ? `<div class="grid cols-3">${runs.map(runCard).join("")}</div>` : `
      <div class="empty"><h3>Noch kein Training</h3><p>Lege eins an – die Vorlagen sind für den Anfang passend eingestellt.</p>
      <a class="btn primary" href="#/training/new">Training anlegen</a></div>`}`;
  state.onSnapshot = debounceReload(renderTraining);
}

// ------------------------------------------------------------------ new run

async function renderNewRun() {
  const { presets, defaults } = await api("/api/presets");
  const form = { preset: "beginner", name: "", overrides: {} };
  const values = () => ({ ...defaults, ...presets[form.preset].values, ...form.overrides });

  const draw = () => {
    const v = values();
    view.innerHTML = `
      <div class="page-head"><div><div class="eyebrow"><a href="#/training">Training</a> / Neu</div><h1>Neues Training</h1>
        <p>Wähle eine Vorlage. Du kannst ein Training jederzeit stoppen und später fortsetzen.</p></div></div>
      <div class="card form" style="max-width:860px">
        <div><div class="section-head"><h2>Vorlage</h2></div><div class="choice-grid">
          ${Object.entries(presets).map(([key, p]) => `<button class="choice ${key === form.preset ? "on" : ""}" data-preset="${key}"><b>${h(p.label)}</b><span>${h(p.description)}</span></button>`).join("")}
        </div></div>
        <div class="grid cols-2">
          <label class="field"><span>Name</span><input id="name" placeholder="z. B. mein-bot" value="${h(form.name)}" maxlength="64"><small>Buchstaben, Ziffern, - _ .</small></label>
          <label class="field"><span>Gesamtschritte</span><input id="total_steps" type="number" min="1000" step="1000" value="${v.total_steps}"><small>${fmt.steps(v.total_steps)} Schritte</small></label>
        </div>
        <div class="grid cols-2">
          <div class="field"><span class="faint" style="font-size:12.5px;font-weight:500">Spielmodus</span>
            <div class="segmented" data-key="team_size">${[1, 2, 3].map((n) => `<button class="${v.team_size === n ? "on" : ""}" data-value="${n}">${n}v${n}</button>`).join("")}</div></div>
          <div class="field"><span class="faint" style="font-size:12.5px;font-weight:500">Belohnungsstufe</span>
            <div class="segmented" data-key="reward_stage">${[1, 2, 3].map((n) => `<button class="${v.reward_stage === n ? "on" : ""}" data-value="${n}">${n} · ${STAGES[n]}</button>`).join("")}</div></div>
        </div>
        <details class="advanced"><summary>Erweiterte Einstellungen</summary>
          <div class="grid cols-3">
            ${numberField("n_workers", "Simulations-Prozesse", v.n_workers, "0 = automatisch (Kerne − 1)")}
            ${numberField("envs_per_worker", "Spiele pro Prozess", v.envs_per_worker)}
            ${numberField("steps_per_iteration", "Schritte pro Update", v.steps_per_iteration)}
            ${numberField("learning_rate", "Lernrate", v.learning_rate, "", "0.00001")}
            ${numberField("entropy_coef", "Entropie (Neugier)", v.entropy_coef, "", "0.001")}
            ${numberField("gamma", "Gamma (Weitsicht)", v.gamma, "", "0.001")}
            ${numberField("checkpoint_every_steps", "Checkpoint alle", v.checkpoint_every_steps, "Schritte")}
            ${numberField("eval_every_steps", "Bewertung alle", v.eval_every_steps, "0 = nie")}
            ${numberField("eval_games", "Spiele pro Bewertung", v.eval_games)}
          </div>
          <label class="field" style="margin-top:16px"><span>Netzgröße (Schichten)</span><input id="hidden_sizes" value="${v.hidden_sizes.join(", ")}"><small>Größer lernt mehr, ist aber langsamer. Später nicht mehr änderbar.</small></label>
        </details>
        <div class="row end"><a class="btn ghost" href="#/training">Abbrechen</a><button class="btn primary" id="start">Training starten</button></div>
      </div>`;

    view.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => { form.preset = b.dataset.preset; form.overrides = {}; keepName(); draw(); }));
    view.querySelectorAll(".segmented").forEach((seg) => seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      form.overrides[seg.dataset.key] = Number(b.dataset.value); keepName(); draw();
    })));
    view.querySelectorAll("input[data-num]").forEach((input) => input.addEventListener("change", () => { form.overrides[input.id] = Number(input.value); }));
    view.querySelector("#total_steps").addEventListener("change", (e) => { form.overrides.total_steps = Number(e.target.value); keepName(); draw(); });
    view.querySelector("#hidden_sizes").addEventListener("change", (e) => {
      form.overrides.hidden_sizes = e.target.value.split(/[ ,;]+/).filter(Boolean).map(Number);
    });
    view.querySelector("#start").addEventListener("click", start);
  };
  const keepName = () => { const el = view.querySelector("#name"); if (el) form.name = el.value.trim(); };
  const start = async () => {
    keepName();
    if (!form.name) return toast("Name fehlt", "Gib dem Training einen Namen.", "error");
    const button = view.querySelector("#start"); button.disabled = true;
    const run = await attempt(() => api("/api/runs", { method: "POST", body: { name: form.name, preset: form.preset, overrides: form.overrides } }), "Training konnte nicht starten");
    if (run) { toast("Training gestartet", run.name); location.hash = `#/training/${encodeURIComponent(run.name)}`; }
    else button.disabled = false;
  };
  draw();
}

function numberField(id, label, value, hint = "", step = "1") {
  return `<label class="field"><span>${h(label)}</span><input id="${id}" data-num type="number" step="${step}" value="${value}">${hint ? `<small>${h(hint)}</small>` : ""}</label>`;
}

// ------------------------------------------------------------------ run detail

async function renderRun(name) {
  name = decodeURIComponent(name);
  let selected = null;
  const load = async () => {
    const [run, metrics, log] = await Promise.all([
      api(`/api/runs/${encodeURIComponent(name)}`),
      api(`/api/runs/${encodeURIComponent(name)}/metrics`),
      api(`/api/runs/${encodeURIComponent(name)}/log?lines=150`),
    ]);
    draw(run, metrics, log);
  };

  const draw = (run, metrics, log) => {
    const st = run.status, cfg = run.config;
    const active = ACTIVE.has(st.state);
    const total = st.total_steps || cfg.total_steps;
    const pct = Math.min(100, ((st.steps || 0) / total) * 100);
    const sps = st.steps_per_second || run.last.steps_per_second;
    const eta = active && sps ? (total - st.steps) / sps : null;
    const last = metrics[metrics.length - 1] || {};
    const series = (key, window = 5) => {
      const values = smooth(metrics.map((m) => m[key]), window);
      return metrics.map((m, i) => [m.steps, values[i]]);
    };
    const checkpoints = [...run.checkpoints].reverse();
    if (!selected && checkpoints.length) selected = checkpoints[0].file;
    const evals = [...run.evaluations].reverse();
    const parts = last.reward_parts || {};

    view.innerHTML = `
      <div class="page-head"><div><div class="eyebrow"><a href="#/training">Training</a> / ${h(name)}</div>
        <div class="row"><h1>${h(name)}</h1>${pill(st.state)}</div>
        <p>${cfg.team_size}v${cfg.team_size} · Stufe ${cfg.reward_stage}: ${h(STAGES[cfg.reward_stage])} · Netz ${cfg.hidden_sizes.join("×")} · ${cfg.n_workers || "auto"} Prozesse</p></div>
        <div class="row">
          ${active ? `<button class="btn" id="stop" ${st.state === "stopping" ? "disabled" : ""}>Stoppen</button>`
                   : `<button class="btn primary" id="resume">${st.steps >= total ? "Weiter trainieren" : "Fortsetzen"}</button>
                      <button class="btn ghost danger" id="delete">Löschen</button>`}
        </div></div>

      <div class="card"><div class="progress"><i style="width:${pct}%"></i></div>
        <div class="progress-meta"><span>${fmt.int(st.steps || 0)} / ${fmt.int(total)} Schritte</span>
        <span>${active ? (eta ? `noch ca. ${fmt.duration(eta)}` : "läuft …") : `${pct.toFixed(1)}%`}</span></div></div>

      <div class="grid cols-4 section" style="margin-top:16px">
        <div class="card stat"><div class="label">Tempo</div><div class="value">${sps ? fmt.int(sps) : "–"}<small>Schritte/s</small></div></div>
        <div class="card stat"><div class="label">Ballkontakte</div><div class="value">${fmt.num(last.touches_per_minute, 1)}<small>pro Minute</small></div></div>
        <div class="card stat"><div class="label">Tore</div><div class="value">${fmt.num(last.goals_per_minute, 2)}<small>pro Minute</small></div></div>
        <div class="card stat"><div class="label">Belohnung</div><div class="value">${fmt.num(last.episode_reward, 1)}<small>pro Episode</small></div></div>
      </div>

      <div class="grid cols-2 section" style="margin-top:16px">
        <div class="card"><div class="card-head"><h3>Ballkontakte pro Minute</h3><span class="sub">geglättet</span></div>${lineChart(series("touches_per_minute"))}</div>
        <div class="card"><div class="card-head"><h3>Belohnung pro Episode</h3><span class="sub">geglättet</span></div>${lineChart(series("episode_reward"))}</div>
        <div class="card"><div class="card-head"><h3>Tore pro Minute</h3><span class="sub">Selbstspiel</span></div>${lineChart(series("goals_per_minute"), { format: (v) => fmt.num(v, 2) })}</div>
        <div class="card"><div class="card-head"><h3>Lernsignal</h3><span class="sub">erklärte Varianz des Kritikers</span></div>${lineChart(series("explained_variance", 3), { format: (v) => fmt.num(v, 2) })}</div>
      </div>

      <div class="section grid cols-2">
        <div class="card"><div class="card-head"><h2>Checkpoints</h2><span class="sub">${checkpoints.length} gespeichert</span></div>
          ${checkpoints.length ? `<table class="table"><thead><tr><th>Stand</th><th>Gespeichert</th><th class="num">Größe</th></tr></thead><tbody>
            ${checkpoints.map((c) => `<tr class="clickable ${c.file === selected ? "selected" : ""}" data-ckpt="${h(c.file)}"><td>${fmt.steps(c.steps)} Schritte</td><td class="faint">${fmt.ago(c.created)}</td><td class="num faint">${(c.size / 1e6).toFixed(1)} MB</td></tr>`).join("")}
          </tbody></table>
          <div class="row" style="margin-top:14px"><span class="faint">Auswahl: <b class="accent">${selected ? fmt.steps(Number(selected.replace(".pt", ""))) : "–"}</b></span><span class="spacer"></span>
            <button class="btn sm" id="eval">Bewerten</button>
            <button class="btn sm" id="watch">In der Arena ansehen</button>
            <a class="btn sm primary" href="#/play?checkpoint=${encodeURIComponent(`${name}/${selected}`)}">Im echten Spiel</a></div>`
          : `<p class="faint">Der erste Checkpoint erscheint nach ${fmt.steps(cfg.checkpoint_every_steps)} Schritten.</p>`}
        </div>
        <div class="card"><div class="card-head"><h2>Bewertungen</h2><span class="sub">Sieg/Unentschieden/Niederlage · Tore</span></div>
          ${evals.length ? `<table class="table"><thead><tr><th>Stand</th><th>Gegner</th><th class="num">S/U/N</th><th class="num">Tore</th><th class="num">Punkte</th></tr></thead><tbody>
            ${evals.slice(0, 8).flatMap((e) => e.results.map((r, i) => `<tr><td>${i ? "" : fmt.steps(e.steps) + (e.manual ? " <span class='faint'>(manuell)</span>" : "")}</td><td>${h(opponentLabel(r.opponent))}</td><td class="num">${r.wins}/${r.draws}/${r.losses}</td><td class="num">${r.goals_for}:${r.goals_against}</td><td class="num"><b>${Math.round(r.score * 100)}%</b></td></tr>`)).join("")}
          </tbody></table>` : `<p class="faint">${cfg.eval_every_steps ? `Automatisch alle ${fmt.steps(cfg.eval_every_steps)} Schritte – oder einen Checkpoint auswählen und „Bewerten“.` : "Automatische Bewertung ist aus. Wähle einen Checkpoint und klicke „Bewerten“."}</p>`}
        </div>
      </div>

      <div class="section grid cols-2">
        <div class="card"><div class="card-head"><h2>Belohnungsanteile</h2><span class="sub">pro Schritt, letzte Iteration</span></div>
          ${Object.keys(parts).length ? `<table class="table"><tbody>${Object.entries(parts).map(([k, v]) => `<tr><td>${h(REWARD_LABEL[k] || k)}</td><td class="num mono">${v.toFixed(4)}</td></tr>`).join("")}</tbody></table>` : `<p class="faint">Noch keine Daten.</p>`}
        </div>
        <div class="card"><div class="card-head"><h2>Protokoll</h2><span class="sub">train.log</span></div>
          <div class="log" id="log">${h([...log.train, ...(st.state === "failed" ? ["", "— Prozessausgabe —", ...log.process] : [])].join("\n")) || "Noch keine Ausgabe."}</div></div>
      </div>`;

    const logEl = view.querySelector("#log"); if (logEl) logEl.scrollTop = logEl.scrollHeight;
    view.querySelectorAll("[data-ckpt]").forEach((row) => row.addEventListener("click", () => { selected = row.dataset.ckpt; draw(run, metrics, log); }));
    const on = (id, fn) => { const el = view.querySelector(id); if (el) el.addEventListener("click", fn); };
    on("#stop", async () => { if (await attempt(() => api(`/api/runs/${encodeURIComponent(name)}/stop`, { method: "POST" }))) { toast("Stopp angefordert", "Die aktuelle Runde wird noch beendet und gespeichert."); load(); } });
    on("#resume", async () => {
      let body = {};
      if (st.steps >= total) {
        const more = prompt("Neues Ziel (Gesamtschritte):", String(total * 2));
        if (!more) return;
        body = { total_steps: Number(more) };
      }
      if (await attempt(() => api(`/api/runs/${encodeURIComponent(name)}/resume`, { method: "POST", body }))) { toast("Training läuft wieder"); load(); }
    });
    on("#delete", async () => {
      if (!confirm(`Training „${name}“ mit allen Checkpoints löschen?`)) return;
      if (await attempt(() => api(`/api/runs/${encodeURIComponent(name)}`, { method: "DELETE" }))) { toast("Gelöscht", name); location.hash = "#/training"; }
    });
    on("#eval", async () => {
      const job = await attempt(() => api("/api/evaluate", { method: "POST", body: { checkpoint: `${name}/${selected}`, games: 6 } }));
      if (job) { toast("Bewertung läuft", "6 Spiele je Gegner, dauert etwa eine Minute."); waitForJob(job.id, () => { toast("Bewertung fertig"); load(); }); }
    });
    on("#watch", async () => {
      const job = await attempt(() => api("/api/matches", { method: "POST", body: { blue: `${name}/${selected}`, orange: "chaser", team_size: cfg.team_size, seconds: 120 } }));
      if (job) { toast("Match wird simuliert", "Gegner: Balljäger"); waitForJob(job.id, (done) => { location.hash = `#/arena/${done.result.replay}`; }); }
    });
  };

  await load();
  state.onSnapshot = debounceReload(() => load().catch(() => {}));
}

const REWARD_LABEL = {
  speed_to_ball: "Tempo zum Ball", face_ball: "Zum Ball schauen", touch: "Ballkontakt", in_air: "In der Luft",
  ball_to_goal: "Ball Richtung Tor", goal: "Tor", boost_keep: "Boost sparen", air_touch: "Luftkontakt",
};
const OPPONENT_LABEL = { idle: "Stillstand", random: "Zufall", chaser: "Balljäger", defender: "Verteidiger" };
const opponentLabel = (s) => OPPONENT_LABEL[s] || s;

// ------------------------------------------------------------------ arena

function replayItem(r) {
  const score = r.goals_blue == null ? "" : `${r.goals_blue} : ${r.goals_orange}`;
  return `<a class="replay-item" href="#/arena/${h(r.id)}" data-id="${h(r.id)}">
    <span class="who">${h(opponentLabel(r.blue) || "?")} <span class="faint">vs</span> ${h(opponentLabel(r.orange) || "?")}</span>
    <span class="res">${score}</span><span class="when">${r.kind === "evaluation" ? "Bewertung · " : ""}${h(r.run === "_matches" ? "Arena" : r.run)} · ${fmt.ago(r.created)}</span></a>`;
}

async function renderArena(replayId) {
  const [replays, opponents] = await Promise.all([api("/api/replays"), api("/api/opponents")]);
  const current = replayId || (replays[0] && replays[0].id);
  const options = (selectedId) => [
    `<optgroup label="Eingebaute Gegner">${opponents.scripted.map((o) => `<option value="${o.id}" ${o.id === selectedId ? "selected" : ""}>${h(o.label)}</option>`).join("")}</optgroup>`,
    ...Object.entries(groupBy(opponents.checkpoints, (c) => c.run)).map(([run, list]) =>
      `<optgroup label="${h(run)}">${list.slice().reverse().map((c) => `<option value="${h(c.id)}" ${c.id === selectedId ? "selected" : ""}>${h(run)} · ${fmt.steps(c.steps)}</option>`).join("")}</optgroup>`),
  ].join("");
  const latest = opponents.checkpoints[opponents.checkpoints.length - 1];

  view.innerHTML = `
    <div class="page-head"><div><div class="eyebrow">Arena</div><h1>Arena</h1>
      <p>Simulierte Spiele als Draufsicht. Jede Bewertung speichert automatisch ein Replay – oder lass hier zwei Gegner antreten.</p></div></div>
    <div class="split">
      <div class="grid">
        <div class="card form"><h2>Neues Match</h2>
          <label class="field"><span><span style="color:var(--blue-team)">■</span> Blau</span><select id="blue">${options(latest ? latest.id : "chaser")}</select></label>
          <label class="field"><span><span style="color:var(--orange-team)">■</span> Orange</span><select id="orange">${options("chaser")}</select></label>
          <div class="grid cols-2">
            <label class="field"><span>Modus</span><select id="size"><option value="1">1v1</option><option value="2">2v2</option><option value="3">3v3</option></select></label>
            <label class="field"><span>Dauer</span><select id="secs"><option value="60">1 min</option><option value="120" selected>2 min</option><option value="300">5 min</option></select></label>
          </div>
          <button class="btn primary" id="go">Simulieren</button></div>
        <div class="card"><div class="card-head"><h2>Replays</h2><span class="sub">${replays.length}</span></div>
          ${replays.length ? `<div class="replay-list">${replays.map(replayItem).join("")}</div>` : `<p class="faint">Noch keine Replays.</p>`}</div>
      </div>
      <div class="arena-stage" id="stage">${current ? `<div class="empty">Lade Replay …</div>` : `<div class="empty"><h3>Kein Replay ausgewählt</h3><p>Starte links ein Match.</p></div>`}</div>
    </div>`;
  view.querySelectorAll(".replay-item").forEach((el) => el.classList.toggle("on", el.dataset.id === current));
  view.querySelector("#go").addEventListener("click", async (event) => {
    event.target.disabled = true;
    const body = { blue: view.querySelector("#blue").value, orange: view.querySelector("#orange").value, team_size: Number(view.querySelector("#size").value), seconds: Number(view.querySelector("#secs").value) };
    const job = await attempt(() => api("/api/matches", { method: "POST", body }));
    if (!job) { event.target.disabled = false; return; }
    toast("Match wird simuliert", "Das dauert nur ein paar Sekunden.");
    waitForJob(job.id, (done) => { location.hash = `#/arena/${done.result.replay}`; });
  });
  if (current) {
    const replay = await attempt(() => api(`/api/replays/${current}`), "Replay konnte nicht geladen werden");
    if (replay) mountPlayer(view.querySelector("#stage"), replay);
  }
}

function groupBy(list, key) {
  return list.reduce((acc, item) => { (acc[key(item)] ||= []).push(item); return acc; }, {});
}

// Field: x in [-4096, 4096], y in [-5120, 5120] (+ goals to ±6000). Drawn sideways: world y → screen x.
const FIELD = { halfX: 4096, halfY: 5120, goalHalfWidth: 893, goalDepth: 880, corner: 1152 };

function mountPlayer(stage, replay) {
  const frames = replay.frames, meta = replay.meta, fps = replay.fps || 15;
  const duration = frames.length ? frames[frames.length - 1][0] : 0;
  const goals = meta.goal_times || [];
  stage.innerHTML = `
    <div class="scoreboard">
      <div class="team blue"><span>${h(opponentLabel(meta.blue))}</span><span class="swatch" style="background:var(--blue-team)"></span></div>
      <div><div class="score" id="score">0 : 0</div><div class="clock" id="clock">0:00</div></div>
      <div class="team orange"><span class="swatch" style="background:var(--orange-team)"></span><span>${h(opponentLabel(meta.orange))}</span></div>
    </div>
    <canvas class="field" id="field"></canvas>
    <div class="player-controls">
      <button class="btn icon-btn" id="toggle" title="Abspielen/Pause (Leertaste)"></button>
      <div class="timeline" id="timeline"><div class="track"></div><div class="fill" id="fill"></div>
        ${goals.map(([t, team]) => `<span class="goal-mark" style="left:${(t / duration) * 100}%;background:${team === 0 ? "var(--blue-team)" : "var(--orange-team)"}" title="Tor bei ${fmt.clock(t)}"></span>`).join("")}
        <div class="knob" id="knob"></div></div>
      <div class="segmented" id="speed">${[0.5, 1, 2, 4].map((s) => `<button data-speed="${s}" class="${s === 1 ? "on" : ""}">${s}×</button>`).join("")}</div>
    </div>`;
  const canvas = stage.querySelector("#field"), ctx = canvas.getContext("2d");
  let t = 0, playing = true, speed = 1, lastTs = null, raf = 0;
  const playIcon = `<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>`;
  const pauseIcon = `<svg viewBox="0 0 24 24"><path d="M7 5h4v14H7zm6 0h4v14h-4z"/></svg>`;
  const toggle = stage.querySelector("#toggle");
  const setPlaying = (p) => { playing = p; toggle.innerHTML = p ? pauseIcon : playIcon; if (p && t >= duration) t = 0; };
  setPlaying(true);
  toggle.addEventListener("click", () => setPlaying(!playing));
  stage.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => {
    speed = Number(b.dataset.speed);
    stage.querySelectorAll("[data-speed]").forEach((x) => x.classList.toggle("on", x === b));
  }));
  const timeline = stage.querySelector("#timeline");
  const seek = (event) => { const r = timeline.getBoundingClientRect(); t = Math.max(0, Math.min(1, (event.clientX - r.left) / r.width)) * duration; };
  let dragging = false;
  timeline.addEventListener("pointerdown", (e) => { dragging = true; timeline.setPointerCapture(e.pointerId); seek(e); });
  timeline.addEventListener("pointermove", (e) => dragging && seek(e));
  timeline.addEventListener("pointerup", () => { dragging = false; });
  const onKey = (e) => { if (e.code === "Space" && e.target === document.body) { e.preventDefault(); setPlaying(!playing); } };
  document.addEventListener("keydown", onKey);

  const resize = () => {
    const ratio = window.devicePixelRatio || 1, w = canvas.clientWidth;
    canvas.width = Math.round(w * ratio); canvas.height = Math.round((w * 8600) / 13000 * ratio);
  };
  resize();
  window.addEventListener("resize", resize);

  const frameAt = (time) => {
    const f = Math.min(frames.length - 1, Math.max(0, time * fps));
    const i = Math.floor(f), j = Math.min(frames.length - 1, i + 1), a = f - i;
    const A = frames[i], B = frames[j];
    // After a goal the next frame jumps to kickoff — never interpolate across that.
    const jump = Math.hypot(B[1][0] - A[1][0], B[1][1] - A[1][1]) > 1500;
    const k = jump ? 0 : a;
    const lerp = (x, y) => x + (y - x) * k;
    const lerpAngle = (x, y) => { let d = y - x; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return x + d * k; };
    return {
      ball: A[1].map((v, n) => lerp(v, B[1][n])),
      cars: A[2].map((car, n) => {
        const other = B[2][n] || car;
        return { team: car[0], x: lerp(car[1], other[1]), y: lerp(car[2], other[2]), z: lerp(car[3], other[3]), yaw: lerpAngle(car[4], other[4]), boost: car[5], demo: car[6] };
      }),
    };
  };

  const draw = () => {
    const W = canvas.width, H = canvas.height, s = W / 13000;
    const X = (worldY) => W / 2 + worldY * s; // world y → screen x
    const Y = (worldX) => H / 2 + worldX * s; // world x → screen y
    ctx.clearRect(0, 0, W, H);
    // pitch outline with cut corners
    const { halfX, halfY, goalHalfWidth, goalDepth, corner } = FIELD;
    ctx.beginPath();
    const pts = [[-halfY + corner, -halfX], [halfY - corner, -halfX], [halfY, -halfX + corner], [halfY, -goalHalfWidth], [halfY + goalDepth, -goalHalfWidth], [halfY + goalDepth, goalHalfWidth], [halfY, goalHalfWidth], [halfY, halfX - corner], [halfY - corner, halfX], [-halfY + corner, halfX], [-halfY, halfX - corner], [-halfY, goalHalfWidth], [-halfY - goalDepth, goalHalfWidth], [-halfY - goalDepth, -goalHalfWidth], [-halfY, -goalHalfWidth], [-halfY, -halfX + corner]];
    pts.forEach(([wy, wx], i) => (i ? ctx.lineTo(X(wy), Y(wx)) : ctx.moveTo(X(wy), Y(wx))));
    ctx.closePath();
    ctx.fillStyle = "#11151d"; ctx.fill();
    ctx.lineWidth = Math.max(1, 2 * s * 10); ctx.strokeStyle = "#2a3140"; ctx.stroke();
    // goals tinted by team
    const goal = (sign, color) => { ctx.fillStyle = color; ctx.fillRect(Math.min(X(sign * halfY), X(sign * (halfY + goalDepth))), Y(-goalHalfWidth), goalDepth * s, 2 * goalHalfWidth * s); };
    goal(-1, "rgba(91,140,255,.16)"); goal(1, "rgba(255,154,77,.16)");
    // midfield
    ctx.strokeStyle = "#1f2430"; ctx.lineWidth = Math.max(1, 12 * s);
    ctx.beginPath(); ctx.moveTo(X(0), Y(-halfX)); ctx.lineTo(X(0), Y(halfX)); ctx.stroke();
    ctx.beginPath(); ctx.arc(X(0), Y(0), 1000 * s, 0, Math.PI * 2); ctx.stroke();

    const frame = frameAt(t);
    // cars
    for (const car of frame.cars) {
      if (car.demo) continue;
      const color = car.team === 0 ? "#5b8cff" : "#ff9a4d";
      const angle = Math.atan2(Math.cos(car.yaw), Math.sin(car.yaw)); // forward (cos, sin) in world → screen
      ctx.save(); ctx.translate(X(car.y), Y(car.x)); ctx.rotate(angle);
      const air = Math.min(1, car.z / 1200);
      ctx.shadowColor = color; ctx.shadowBlur = 18 * air * s * 40;
      ctx.fillStyle = color;
      roundRect(ctx, -118 * s, -42 * s, 236 * s, 84 * s, 22 * s); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(11,13,18,.75)"; roundRect(ctx, 30 * s, -30 * s, 50 * s, 60 * s, 10 * s); ctx.fill();
      ctx.restore();
      // boost bar
      ctx.fillStyle = "#1f2430"; ctx.fillRect(X(car.y) - 120 * s, Y(car.x) + 80 * s, 240 * s, 22 * s);
      ctx.fillStyle = "#c6f432"; ctx.fillRect(X(car.y) - 120 * s, Y(car.x) + 80 * s, 240 * s * (car.boost / 100), 22 * s);
    }
    // ball with shadow; it grows a little with height
    const [bx, by, bz] = frame.ball;
    ctx.fillStyle = "rgba(0,0,0,.45)";
    ctx.beginPath(); ctx.ellipse(X(by), Y(bx), 92 * s, 92 * s, 0, 0, Math.PI * 2); ctx.fill();
    const r = (92 + Math.min(bz, 2000) * 0.06) * s;
    const lift = Math.min(bz, 2000) * 0.08 * s;
    ctx.fillStyle = "#f2f4f8";
    ctx.beginPath(); ctx.arc(X(by), Y(bx) - lift, r, 0, Math.PI * 2); ctx.fill();

    // HUD
    const scored = goals.filter(([gt]) => gt <= t + 1e-6);
    stage.querySelector("#score").textContent = `${scored.filter((g) => g[1] === 0).length} : ${scored.filter((g) => g[1] === 1).length}`;
    stage.querySelector("#clock").textContent = `${fmt.clock(t)} / ${fmt.clock(duration)}`;
    const pct = duration ? (t / duration) * 100 : 0;
    stage.querySelector("#fill").style.width = `${pct}%`;
    stage.querySelector("#knob").style.left = `${pct}%`;
  };

  const loop = (ts) => {
    if (lastTs != null && playing && !dragging) {
      t += ((ts - lastTs) / 1000) * speed;
      if (t >= duration) { t = duration; setPlaying(false); }
    }
    lastTs = ts;
    draw();
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
  state.cleanup.push(() => { cancelAnimationFrame(raf); document.removeEventListener("keydown", onKey); window.removeEventListener("resize", resize); });
}

function roundRect(ctx, x, y, w, h_, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h_, r); ctx.arcTo(x + w, y + h_, x, y + h_, r);
  ctx.arcTo(x, y + h_, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// ------------------------------------------------------------------ play

async function renderPlay(_unused, query) {
  const [info, opponents] = await Promise.all([api("/api/play"), api("/api/opponents")]);
  const preselect = query && query.get("checkpoint");
  const form = { checkpoint: preselect || (opponents.checkpoints.at(-1) || {}).id || "", mode: "psyonix", team_size: 1, skill: "rookie", launcher: "epic", opponent_bot: "" };
  const ready = info.windows && info.server_installed;

  const draw = (current) => {
    const busy = current.state === "starting" || current.state === "running";
    view.innerHTML = `
      <div class="page-head"><div><div class="eyebrow">Spielen</div><h1>Im echten Rocket League</h1>
        <p>RLBot startet Rocket League und ein <b>Offline-Match</b>, in dem deine KI ein Auto steuert – gegen Psyonix-Bots, Community-Bots oder dich selbst.</p></div>
        <div>${pill(current.state)}</div></div>
      <div class="note" style="margin-bottom:20px"><div><b>Nur offline.</b> Seit April 2026 nutzt Rocket League Easy Anti-Cheat. Bots funktionieren ausschließlich in lokalen Matches, die RLBot startet – nie online, in Ranked oder privaten Online-Matches.</div></div>
      <div class="split">
        <div class="card"><div class="card-head"><h2>Voraussetzungen</h2></div><div class="checks">
          ${check(info.windows, "Windows-PC", info.windows ? "erkannt" : "Rocket League + RLBot laufen nur unter Windows. Trainieren geht trotzdem hier.")}
          ${check(true, "Rocket League installiert", "über Steam oder Epic Games – wird nicht automatisch geprüft")}
          ${check(info.server_installed, "RLBotServer", info.server_installed ? "installiert" : "fehlt – python install.py ausführen")}
          ${check(opponents.checkpoints.length > 0, "Trainierte KI", opponents.checkpoints.length ? `${opponents.checkpoints.length} Checkpoints` : "erst ein Training laufen lassen")}
        </div></div>
        <div class="card form"><h2>Match einrichten</h2>
          <label class="field"><span>KI (Checkpoint)</span><select id="checkpoint">
            ${opponents.checkpoints.length ? Object.entries(groupBy(opponents.checkpoints, (c) => c.run)).map(([run, list]) => `<optgroup label="${h(run)}">${list.slice().reverse().map((c) => `<option value="${h(c.id)}" ${c.id === form.checkpoint ? "selected" : ""}>${h(run)} · ${fmt.steps(c.steps)} Schritte</option>`).join("")}</optgroup>`).join("") : `<option value="">– noch keine –</option>`}
          </select></label>
          <div class="field"><span class="faint" style="font-size:12.5px;font-weight:500">Gegner</span>
            <div class="choice-grid">${Object.entries(info.modes).map(([key, label]) => `<button class="choice ${form.mode === key ? "on" : ""}" data-mode="${key}"><b>${h(label)}</b><span>${MODE_HINT[key]}</span></button>`).join("")}</div></div>
          <div class="grid cols-2">
            <div class="field"><span class="faint" style="font-size:12.5px;font-weight:500">Modus</span>
              <div class="segmented" data-key="team_size">${[1, 2, 3].map((n) => `<button class="${form.team_size === n ? "on" : ""}" data-value="${n}">${n}v${n}</button>`).join("")}</div></div>
            <div class="field"><span class="faint" style="font-size:12.5px;font-weight:500">Launcher</span>
              <div class="segmented" data-key="launcher">${Object.entries(info.launchers).map(([k, l]) => `<button class="${form.launcher === k ? "on" : ""}" data-value="${k}">${h(l)}</button>`).join("")}</div></div>
          </div>
          ${form.mode === "psyonix" || form.mode === "human" ? `<div class="field"><span class="faint" style="font-size:12.5px;font-weight:500">${form.mode === "human" ? "Stärke deiner Bot-Mitspieler" : "Stärke der Psyonix-Bots"}</span>
            <div class="segmented" data-key="skill">${Object.entries(info.skills).map(([k, l]) => `<button class="${form.skill === k ? "on" : ""}" data-value="${k}">${h(l)}</button>`).join("")}</div></div>` : ""}
          ${form.mode === "bot" ? `<label class="field"><span>bot.toml des Community-Bots</span><input id="opponent_bot" placeholder="C:\\Bots\\Nexto\\bot.toml" value="${h(form.opponent_bot)}"><small>Bots für RLBot v5, z. B. aus dem RLBot-Botpack.</small></label>` : ""}
          ${current.message ? `<p class="${current.state === "failed" ? "" : "faint"}" style="${current.state === "failed" ? "color:var(--danger)" : ""}">${h(current.message)}</p>` : ""}
          <div class="row end">
            ${busy ? `<button class="btn" id="stop">Match beenden</button>` : ""}
            <button class="btn primary" id="start" ${!ready || busy || !form.checkpoint ? "disabled" : ""}>Match starten</button>
          </div>
          ${!ready ? `<p class="faint" style="font-size:12.5px">Starten ist erst möglich, wenn alle Voraussetzungen erfüllt sind.</p>` : ""}
        </div>
      </div>`;
    const sel = view.querySelector("#checkpoint"); if (sel) sel.addEventListener("change", () => { form.checkpoint = sel.value; });
    const bot = view.querySelector("#opponent_bot"); if (bot) bot.addEventListener("change", () => { form.opponent_bot = bot.value.trim(); });
    view.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => { form.mode = b.dataset.mode; draw(current); }));
    view.querySelectorAll(".segmented").forEach((seg) => seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      const value = b.dataset.value; form[seg.dataset.key] = seg.dataset.key === "team_size" ? Number(value) : value; draw(current);
    })));
    const start = view.querySelector("#start");
    if (start) start.addEventListener("click", async () => {
      start.disabled = true;
      const result = await attempt(() => api("/api/play/start", { method: "POST", body: form }), "Match konnte nicht starten");
      if (result) { toast("Match startet", "Rocket League öffnet sich gleich."); draw(result); } else start.disabled = false;
    });
    const stop = view.querySelector("#stop");
    if (stop) stop.addEventListener("click", async () => { const result = await attempt(() => api("/api/play/stop", { method: "POST" })); if (result) draw(result); });
  };
  draw(info);
  state.onSnapshot = debounceReload(async () => draw(await api("/api/play")));
}

const MODE_HINT = {
  psyonix: "Die eingebauten Bots von Beginner bis All-Star",
  human: "Du spielst selbst gegen deine KI",
  bot: "Jeder RLBot-v5-Bot, z. B. aus dem Botpack",
  self: "Zwei Kopien deiner KI gegeneinander",
};

function check(ok, label, detail) {
  return `<div class="check ${ok ? "ok" : "bad"}"><span class="mark">${ok ? "✓" : "!"}</span><div><b>${h(label)}</b><div class="detail">${h(detail)}</div></div></div>`;
}

// ------------------------------------------------------------------ setup

async function renderSetup() {
  const system = await api("/api/system");
  const commands = [
    ["Installation (einmalig)", "python install.py"],
    ["App starten", "python start.py"],
    ["Training ohne Oberfläche", "python -m rocketai train --preset beginner --name mein-bot"],
    ["Checkpoint bewerten", "python -m rocketai eval runs/mein-bot/checkpoints/latest.pt"],
    ["Installation prüfen", "python -m rocketai doctor"],
  ];
  view.innerHTML = `
    <div class="page-head"><div><div class="eyebrow">Einrichtung</div><h1>Einrichtung</h1>
      <p>Alles, was RocketAI braucht – und wie du es ohne Oberfläche bedienst.</p></div><span class="faint">Version ${h(system.version)}</span></div>
    <div class="grid cols-2">
      <div class="card"><div class="card-head"><h2>Systemprüfung</h2></div><div class="checks">
        ${system.checks.map((c) => `<div class="check ${c.ok ? "ok" : c.required ? "bad" : ""}"><span class="mark">${c.ok ? "✓" : c.required ? "!" : "–"}</span><div><b>${h(c.label)}</b>${c.required ? "" : ' <span class="faint">(optional)</span>'}<div class="detail">${h(c.detail)}</div></div></div>`).join("")}
      </div></div>
      <div class="grid">
        <div class="card"><div class="card-head"><h2>Befehle</h2></div><div class="grid" style="gap:8px">
          ${commands.map(([label, cmd]) => `<div><div class="faint" style="font-size:12px;margin-bottom:4px">${h(label)}</div><div class="cmd"><code>${h(cmd)}</code><button class="btn ghost sm" data-copy="${h(cmd)}">Kopieren</button></div></div>`).join("")}
        </div></div>
        <div class="card"><div class="card-head"><h2>Ordner</h2></div><p class="muted">Trainings liegen in</p><div class="cmd" style="margin-top:8px"><code>${h(system.runs_folder)}</code></div></div>
      </div>
    </div>
    <div class="section card"><div class="card-head"><h2>Wie lange dauert das Training?</h2><span class="sub">Erfahrungswerte aus der RLGym-Community, dein Tempo steht beim Training</span></div>
      <table class="table"><thead><tr><th>Ziel</th><th class="num">Schritte (grob)</th><th class="num">8 Kerne (~15 000/s)</th></tr></thead><tbody>
        <tr><td>Fährt zum Ball und trifft ihn</td><td class="num">20–50 Mio.</td><td class="num">0,5–1 h</td></tr>
        <tr><td>Schießt gezielt Tore</td><td class="num">100–300 Mio.</td><td class="num">2–6 h</td></tr>
        <tr><td>Schlägt Psyonix Rookie/Pro</td><td class="num">0,3–1 Mrd.</td><td class="num">6–20 h</td></tr>
        <tr><td>Gold/Platin-Niveau</td><td class="num">mehrere Mrd.</td><td class="num">Tage bis Wochen</td></tr>
      </tbody></table></div>`;
  view.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", async () => {
    await navigator.clipboard.writeText(b.dataset.copy).then(() => toast("Kopiert", b.dataset.copy), () => toast("Kopieren nicht möglich", "", "error"));
  }));
}

// ------------------------------------------------------------------ boot

window.addEventListener("hashchange", router);
connectEvents();
router();
