// RocketAI web app — no framework, no build step, works offline.
// Pages render HTML strings; `patch` applies them without destroying focus,
// scroll positions or open panels, so live updates never get in your way.

import {
  ACTIVE, STAGES, api, attempt, fmt, groupBy, h, icon, lineChart, modal, opponentLabel,
  patch, pill, smooth, sparkline, toast, waitForJob,
} from "./js/core.js";
import { blend } from "./js/field.js";
import { createStage, fieldPads, stageToolbar } from "./js/stage.js";

const view = document.getElementById("view");
const app = { snapshot: null, snapshotText: "", rl: null };
let page = { actions: {}, onSnapshot: null, cleanup: [] };

// ------------------------------------------------------------------ events

view.addEventListener("click", (e) => {
  const el = e.target.closest("[data-action]");
  if (!el || !view.contains(el)) return;
  const fn = page.actions[el.dataset.action];
  if (!fn) return;
  e.preventDefault();
  if (el.disabled) return;
  fn(el, e);
});
view.addEventListener("change", (e) => {
  const el = e.target.closest("[data-change]");
  if (el) page.actions[el.dataset.change]?.(el, e);
});
view.addEventListener("input", (e) => {
  if (e.target.matches("input, select, textarea")) e.target.dataset.dirty = "1";
});

/** Disable a button while an async action runs. */
async function busy(el, fn) {
  el.disabled = true;
  try { return await fn(); } finally { if (el.isConnected) el.disabled = false; }
}

const seg = (key, options, current, action = "seg") =>
  `<div class="segmented">${options.map(([v, l]) => `<button type="button" class="${String(v) === String(current) ? "on" : ""}" data-action="${action}" data-key="${key}" data-value="${h(v)}">${h(l)}</button>`).join("")}</div>`;

// ------------------------------------------------------------------ polling

async function pollSnapshot() {
  const live = document.getElementById("conn");
  try {
    const snap = await api("/api/snapshot");
    const text = JSON.stringify(snap);
    live.classList.add("on");
    const active = snap.runs.filter((r) => ACTIVE.has(r.state));
    document.getElementById("conn-text").textContent = active.length ? `${active.length} Training${active.length > 1 ? "s" : ""} aktiv` : "Verbunden";
    document.querySelector('[data-route="live"]').classList.toggle("is-live", !!snap.live);
    if (text !== app.snapshotText) {
      app.snapshot = snap;
      app.snapshotText = text;
      page.onSnapshot?.(snap);
    }
  } catch {
    live.classList.remove("on");
    document.getElementById("conn-text").textContent = "Keine Verbindung …";
  }
  setTimeout(pollSnapshot, document.hidden ? 6000 : 2000);
}

async function pollRocketLeague(refresh = false) {
  try {
    app.rl = await api(`/api/rocketleague${refresh ? "?refresh=true" : ""}`);
    renderRlChip();
  } catch { /* offline */ }
}

function renderRlChip() {
  const rl = app.rl, el = document.getElementById("rl-chip");
  if (!rl || !el) return;
  const kind = !rl.supported ? "muted" : rl.game === "normal" ? "warn" : rl.ready ? "ok" : "bad";
  el.className = `rl-chip ${kind}`;
  el.innerHTML = `<span class="dot"></span><span><b>Rocket League</b><small>${h(rl.verdict)}</small></span>`;
}

// ------------------------------------------------------------------ router

const routes = [
  [/^\/?$/, "overview", pageOverview],
  [/^\/training\/?$/, "training", pageTraining],
  [/^\/training\/new\/?$/, "training", pageNewRun],
  [/^\/training\/([^/]+)\/?$/, "training", pageRun],
  [/^\/live\/?$/, "live", pageLive],
  [/^\/arena\/?$/, "arena", pageArena],
  [/^\/arena\/(.+)$/, "arena", pageArena],
  [/^\/play\/?$/, "play", pagePlay],
  [/^\/setup\/?$/, "setup", pageSetup],
];

async function router() {
  page.cleanup.forEach((fn) => { try { fn(); } catch { /* ignore */ } });
  page = { actions: {}, onSnapshot: null, cleanup: [] };
  const path = decodeURIComponent(location.hash.replace(/^#/, "").split("?")[0]);
  const query = new URLSearchParams(location.hash.split("?")[1] || "");
  for (const [pattern, nav, render] of routes) {
    const match = path.match(pattern);
    if (!match) continue;
    document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === nav));
    view.innerHTML = `<div class="loading"><span></span><span></span><span></span></div>`;
    view.classList.remove("fade-in"); void view.offsetWidth; view.classList.add("fade-in");
    window.scrollTo(0, 0);
    const current = page;
    try { await render(match[1], query); } catch (error) {
      if (page === current) view.innerHTML = `<div class="empty"><h3>Seite konnte nicht geladen werden</h3><p>${h(error.message)}</p><button class="btn" onclick="location.reload()">Neu laden</button></div>`;
    }
    return;
  }
  location.hash = "#/";
}

/** Run ``fn`` on every snapshot change, at most once at a time. */
function refreshOnSnapshot(fn) {
  let running = false, again = false;
  page.onSnapshot = async () => {
    if (running) { again = true; return; }
    running = true;
    try { await fn(); } catch { /* keep the old view */ }
    running = false;
    if (again) { again = false; page.onSnapshot?.(); }
  };
}

// ------------------------------------------------------------------ shared bits

function runCard(run) {
  const st = run.status || {}, cfg = run.config || {};
  const total = st.total_steps || cfg.total_steps || 1;
  const pct = Math.min(100, ((st.steps || 0) / total) * 100);
  const evalRes = run.evaluation && run.evaluation.results ? run.evaluation.results : [];
  const score = evalRes.length ? evalRes.map((r) => `${Math.round(r.score * 100)}%`).join(" · ") : "–";
  return `<a class="card link run-card" href="#/training/${encodeURIComponent(run.name)}" data-key="run-${h(run.name)}">
    <div class="top"><div><div class="name">${h(run.name)}</div>
      <div class="meta">${cfg.auto_curriculum ? `<span class="badge accent">Autopilot</span> ` : ""}${cfg.team_size}v${cfg.team_size} · Stufe ${cfg.reward_stage}: ${h(STAGES[cfg.reward_stage] || "")}</div></div>${pill(st.state)}</div>
    <div><div class="progress ${ACTIVE.has(st.state) ? "active" : ""}"><i style="width:${pct}%"></i></div>
      <div class="progress-meta"><span>${fmt.steps(st.steps || 0)} / ${fmt.steps(total)} Schritte</span><span>${pct.toFixed(1)} %</span></div></div>
    <div class="kv">
      <div><span>Ballkontakte/min</span><b>${fmt.num(run.last.touches_per_minute, 1)}</b></div>
      <div><span>Tempo</span><b>${run.last.steps_per_second && ACTIVE.has(st.state) ? fmt.int(run.last.steps_per_second) + "/s" : "–"}</b></div>
      <div><span>Bewertung</span><b>${score}</b></div>
    </div></a>`;
}

function replayItem(r, current) {
  const score = r.goals_blue == null ? "" : `${r.goals_blue} : ${r.goals_orange}`;
  return `<a class="replay-item ${r.id === current ? "on" : ""}" href="#/arena/${h(r.id)}" data-key="rp-${h(r.id)}">
    <span class="who"><i class="team-dot blue"></i>${h(opponentLabel(r.blue) || "?")} <span class="faint">vs</span> <i class="team-dot orange"></i>${h(opponentLabel(r.orange) || "?")}</span>
    <span class="res">${score}</span><span class="when">${r.kind === "evaluation" ? "Bewertung · " : ""}${h(r.run === "_matches" ? "Arena" : r.run)} · ${fmt.ago(r.created)}</span></a>`;
}

function checkRow(state, label, detail, extra = "") {
  const mark = { ok: icon("check"), bad: "!", warn: "!", off: "–" }[state] || "–";
  return `<div class="check ${state}"><span class="mark">${mark}</span><div><b>${h(label)}</b>${extra}<div class="detail">${h(detail)}</div></div></div>`;
}

function rlCard(rl, { compact = false } = {}) {
  if (!rl) return `<div class="card"><div class="card-head"><h2>Rocket League</h2></div><p class="faint">Prüfe …</p></div>`;
  return `<div class="card"><div class="card-head"><h2>Rocket League</h2>
      <button class="btn ghost sm" data-action="rl-refresh">${icon("refresh")}Erneut prüfen</button></div>
    <div class="verdict ${!rl.supported ? "muted" : rl.game === "normal" ? "warn" : rl.ready ? "ok" : "bad"}">${h(rl.verdict)}</div>
    <div class="checks">${rl.steps.map((s) => checkRow(s.state, s.label, s.detail)).join("")}</div>
    ${compact ? "" : `<p class="faint small" style="margin-top:12px">Geprüft wird wie bei RLBot selbst: Steam-Bibliotheken bzw. Epic-Manifeste, laufende Prozesse samt Startparametern und der RLBotServer.</p>`}
  </div>`;
}

const rlActions = {
  "rl-refresh": (el) => busy(el, async () => { await pollRocketLeague(true); page.onSnapshot?.(); toast("Rocket League geprüft", app.rl?.verdict || ""); }),
};

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try { await navigator.clipboard.writeText(text); toast("Kopiert", text); return; } catch { /* fall back */ }
  }
  const area = document.createElement("textarea");
  area.value = text; area.style.position = "fixed"; area.style.opacity = "0";
  document.body.appendChild(area); area.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch { ok = false; }
  area.remove();
  if (ok) toast("Kopiert", text);
  else modal({ title: "Zum Kopieren markieren", body: `<div class="cmd"><code class="select-all">${h(text)}</code></div><p class="faint small">Mit Strg+C kopieren.</p>`, confirm: "Fertig", cancel: "Schließen" });
}

async function startLive(blue, orange = "chaser", extra = {}) {
  const result = await attempt(() => api("/api/live/start", { method: "POST", body: { blue, orange, ...extra } }), "Live-Spiel konnte nicht starten");
  if (result) location.hash = "#/live";
  return result;
}

// ------------------------------------------------------------------ overview

async function pageOverview() {
  const draw = async () => {
    const [data, live] = await Promise.all([api("/api/overview"), api("/api/live").catch(() => ({ active: false }))]);
    const runs = data.runs;
    const active = runs.filter((r) => ACTIVE.has(r.status.state));
    const totalSteps = runs.reduce((sum, r) => sum + (r.status.steps || 0), 0);
    const checkpoints = runs.reduce((sum, r) => sum + r.checkpoints, 0);
    const speed = active.reduce((sum, r) => sum + (r.status.steps_per_second || 0), 0);
    const watchable = runs.find((r) => r.checkpoints > 0);
    patch(view, `
      <div class="hero card">
        <div class="hero-text"><div class="eyebrow">RocketAI</div><h1>Deine Rocket-League-KI</h1>
          <p>Trainiere in der Simulation, schau ihr live beim Spielen und Denken zu und lass sie im echten Rocket League offline gegen Bots oder dich antreten.</p>
          <div class="row" style="margin-top:20px">
            <a class="btn primary" href="#/training/new">${icon("plus")}Neues Training</a>
            ${live.active ? `<a class="btn" href="#/live"><span class="live-dot"></span>Live-Spiel läuft</a>`
              : watchable ? `<button class="btn" data-action="watch" data-run="${h(watchable.name)}">${icon("eye")}${h(watchable.name)} live ansehen</button>` : ""}
          </div></div>
        <div class="hero-steps">
          ${heroStep(1, "Trainieren", runs.length > 0, runs.length ? `${runs.length} Training${runs.length > 1 ? "s" : ""}` : "Vorlage wählen")}
          ${heroStep(2, "Zuschauen", checkpoints > 0, checkpoints ? "Live-Tab mit 3D & Gehirn" : "nach dem 1. Checkpoint")}
          ${heroStep(3, "Echtes Spiel", app.rl?.ready, app.rl ? app.rl.verdict : "prüfe …")}
        </div>
      </div>
      <div class="grid cols-4 section">
        <div class="card stat"><div class="label">Aktive Trainings</div><div class="value">${active.length}</div><div class="foot">${speed ? fmt.int(speed) + " Schritte/s" : "nichts läuft"}</div></div>
        <div class="card stat"><div class="label">Trainierte Schritte</div><div class="value">${fmt.steps(totalSteps)}</div><div class="foot">über ${runs.length} Run${runs.length === 1 ? "" : "s"}</div></div>
        <div class="card stat"><div class="label">Checkpoints</div><div class="value">${checkpoints}</div><div class="foot">spielbare KI-Versionen</div></div>
        <div class="card stat"><div class="label">Echtes Spiel</div><div class="value value-sm">${pill(data.play.state)}</div><div class="foot">${h(data.play.message || "über RLBot, nur offline")}</div></div>
      </div>
      <div class="section"><div class="section-head"><h2>Trainings</h2><a class="btn ghost sm" href="#/training">Alle ansehen</a></div>
        ${runs.length ? `<div class="grid cols-3">${runs.slice(0, 6).map(runCard).join("")}</div>` : `
        <div class="empty"><h3>Noch kein Training</h3><p>Starte mit dem Schnelltest (2 Minuten), um zu prüfen, ob alles läuft.</p>
        <a class="btn primary" href="#/training/new">Erstes Training anlegen</a></div>`}
      </div>
      <div class="section grid cols-2">
        <div class="card"><div class="card-head"><h2>Letzte Replays</h2><a class="btn ghost sm" href="#/arena">Arena</a></div>
          ${data.replays.length ? `<div class="replay-list">${data.replays.map((r) => replayItem(r)).join("")}</div>` : `<p class="faint">Replays entstehen bei jeder Bewertung während des Trainings oder über die Arena.</p>`}</div>
        ${rlCard(app.rl, { compact: true })}
      </div>`);
  };
  page.actions = { ...rlActions, watch: (el) => busy(el, () => startLive(`run:${el.dataset.run}`)) };
  await draw();
  refreshOnSnapshot(draw);
}

const heroStep = (n, title, done, detail) =>
  `<div class="hero-step ${done ? "done" : ""}"><span class="n">${done ? icon("check") : n}</span><div><b>${h(title)}</b><small>${h(detail)}</small></div></div>`;

// ------------------------------------------------------------------ training list

async function pageTraining() {
  const draw = async () => {
    const runs = await api("/api/runs");
    patch(view, `
      <div class="page-head"><div><div class="eyebrow">Training</div><h1>Trainings</h1>
        <p>Jedes Training ist ein eigener Ordner unter <code>runs/</code> mit Einstellungen, Checkpoints, Messwerten und Replays.</p></div>
        <a class="btn primary" href="#/training/new">${icon("plus")}Neues Training</a></div>
      ${runs.length ? `<div class="grid cols-3">${runs.map(runCard).join("")}</div>` : `
        <div class="empty"><h3>Noch kein Training</h3><p>Lege eins an – die Vorlagen sind für den Anfang passend eingestellt.</p>
        <a class="btn primary" href="#/training/new">Training anlegen</a></div>`}`);
  };
  await draw();
  refreshOnSnapshot(draw);
}

// ------------------------------------------------------------------ new run

async function pageNewRun() {
  const [{ presets, defaults, cpu_count: cpus = 2 }, opponents, runs] = await Promise.all([
    api("/api/presets"), api("/api/opponents"), api("/api/runs"),
  ]);
  // Am ehrlichsten ist der gemessene Wert: das schnellste Tempo, das auf diesem
  // Rechner schon einmal erreicht wurde. Sonst eine vorsichtige Schätzung.
  const measured = Math.max(0, ...runs.map((r) => r.last?.steps_per_second || 0));
  const guess = 1400 * Math.max(1, cpus - 1);
  const teacherReady = Boolean(opponents.teacher?.ready);
  const pyCmd = opponents.python || "python3";
  const form = {
    preset: presets.student ? "student" : presets.autopilot ? "autopilot" : "beginner",
    name: "", overrides: {},
  };
  const values = () => ({ ...defaults, ...presets[form.preset].values, ...form.overrides });
  const draw = () => {
    const v = values();
    const workers = v.n_workers || Math.max(1, cpus - 1);
    const sps = measured || guess;
    const speedHint = measured
      ? `gemessen an deinem schnellsten Training`
      : (v.teacher_opponent_prob > 0 ? "grobe Schätzung (mit Lehrer etwas langsamer)" : "grobe Schätzung");
    patch(view, `
      <div class="page-head"><div><div class="eyebrow"><a href="#/training">Training</a> / Neu</div><h1>Neues Training</h1>
        <p>Wähle eine Vorlage. Du kannst ein Training jederzeit stoppen und später fortsetzen – auch mit höherem Ziel.</p></div></div>
      <div class="split wide">
        <div class="card form">
          <div><div class="section-head"><h2>Vorlage</h2></div><div class="choice-grid">
            ${Object.entries(presets).map(([key, p]) => `<button type="button" class="choice ${key === form.preset ? "on" : ""}" data-action="preset" data-preset="${key}"><b>${h(p.label)}</b><span>${h(p.description)}</span></button>`).join("")}
          </div></div>
          <div class="grid cols-2">
            <label class="field"><span>Name</span><input id="name" placeholder="z. B. mein-bot" value="${h(form.name)}" maxlength="64" autocomplete="off" data-change="name"><small>Buchstaben, Ziffern, - _ .</small></label>
            <label class="field"><span>Gesamtschritte</span><input id="total_steps" type="number" min="1000" step="1000000" value="${v.total_steps}" data-change="num" data-key="total_steps"><small>${fmt.steps(v.total_steps)} Schritte</small></label>
          </div>
          <div class="grid cols-2">
            <div class="field"><span class="field-label">Spielmodus</span>${seg("team_size", [[1, "1v1"], [2, "2v2"], [3, "3v3"]], v.team_size)}</div>
            <div class="field"><span class="field-label">${v.auto_curriculum ? "Startstufe" : "Belohnungsstufe"}</span>${seg("reward_stage", [[1, "1 · Ball"], [2, "2 · Tore"], [3, "3 · Komplett"]], v.reward_stage)}</div>
          </div>
          <div class="grid cols-2">
            <div class="field"><span class="field-label">Autopilot</span>${seg("auto_curriculum", [[1, "An"], [0, "Aus"]], v.auto_curriculum ? 1 : 0)}
              <small>Schaltet die Belohnungsstufe selbst hoch, sobald die KI so weit ist (erst Ball sicher treffen, dann Tore schießen).</small></div>
            <div class="field"><span class="field-label">Gegner-Pool</span>${seg("past_opponent_prob", [[0, "Aus"], [0.2, "20 %"], [0.35, "35 %"]], v.past_opponent_prob)}
              <small>Anteil der Spiele gegen ältere eigene Versionen – verhindert, dass die KI Gelerntes wieder vergisst.</small></div>
          </div>
          <div class="teacher-box">
            <div class="row between"><b>Lehrer (Nexto) nutzen</b><span class="faint small">${teacherReady ? "geladen" : "noch nicht geladen"}</span></div>
            <div class="grid cols-2">
              <div class="field"><span class="field-label">Gegen den Lehrer spielen</span>${seg("teacher_opponent_prob", [[0, "Aus"], [0.15, "15 %"], [0.25, "25 %"], [0.4, "40 %"]], v.teacher_opponent_prob)}
                <small>Anteil der Trainingsspiele gegen Nexto (Grand Champion). Stärkt die KI am schnellsten, weil sie gegen einen richtig guten Gegner spielt.</small></div>
              <div class="field"><span class="field-label">Nachahmung</span>${seg("teacher_weight", [[0, "Aus"], [0.5, "50 %"], [1, "100 %"]], v.teacher_weight)}
                <small>Die KI versucht zusätzlich, die Tasten des Lehrers vorherzusagen. Achtung: gemessen nur ein kleiner Zusatzeffekt – der große Hebel ist das Spielen gegen ihn.</small></div>
            </div>
            <div class="row between">
              <p class="faint small">Der Lehrer wird einmalig aus dem Internet geladen (nicht Teil des Projekts, GPL, nur offline nutzen). Ohne Oberfläche: <code>${pyCmd} -m rocketai teacher</code></p>
              <button class="btn sm" data-action="load-teacher">${teacherReady ? "Neu laden" : "Lehrer laden"}</button>
            </div>
          </div>
          <details class="advanced"><summary>Erweiterte Einstellungen</summary>
            <div class="grid cols-3">
              ${numberField("n_workers", "Simulations-Prozesse", v.n_workers, "0 = automatisch. Empfehlung: 12–16 (ein paar Kerne für Windows und den Lernprozess frei lassen)")}
              ${numberField("torch_threads", "Threads des Lernprozesses", v.torch_threads, "0 = automatisch (wenige Threads, damit die Simulationen die Kerne behalten)")}
              ${numberField("envs_per_worker", "Spiele pro Prozess", v.envs_per_worker)}
              ${numberField("steps_per_iteration", "Schritte pro Update", v.steps_per_iteration)}
              ${numberField("learning_rate", "Lernrate", v.learning_rate, "", "0.00001")}
              ${numberField("entropy_coef", "Entropie (Neugier)", v.entropy_coef, "", "0.001")}
              ${numberField("gamma", "Gamma (Weitsicht)", v.gamma, "", "0.001")}
              ${numberField("checkpoint_every_steps", "Checkpoint alle", v.checkpoint_every_steps, "Schritte – so oft gibt es neue Stände für Live")}
              ${numberField("eval_every_steps", "Bewertung alle", v.eval_every_steps, "0 = nie")}
              ${numberField("eval_games", "Spiele pro Bewertung", v.eval_games)}
              ${numberField("past_pool_size", "Größe Gegner-Pool", v.past_pool_size, "so viele ältere Checkpoints spielen mit")}
              ${numberField("teacher_samples", "Lehrer-Antworten pro Update", v.teacher_samples, "0 = für jeden Schritt (langsamer)")}
              ${numberField("teacher_temperature", "Lehrer-Temperatur", v.teacher_temperature, "1 = wie trainiert")}
            </div>
            <label class="field" style="margin-top:16px"><span>Netzgröße (Schichten)</span><input id="hidden_sizes" value="${v.hidden_sizes.join(", ")}" data-change="sizes"><small>Größer lernt mehr, ist aber langsamer. Später nicht mehr änderbar.</small></label>
          </details>
          <div class="row end"><a class="btn ghost" href="#/training">Abbrechen</a><button class="btn primary" data-action="start">${icon("play")}Training starten</button></div>
        </div>
        <div class="card sticky"><div class="card-head"><h2>Schätzung</h2><span class="sub">${cpus} logische CPU-Kerne</span></div>
          <div class="estimate"><span>Tempo ${measured ? "" : "(grob)"}</span><b>~${fmt.int(sps)} Schritte/s</b></div>
          <p class="faint small" style="margin:-4px 0 10px">${speedHint}</p>
          <div class="estimate"><span>Dauer für ${fmt.steps(v.total_steps)}</span><b>~${fmt.duration(v.total_steps / sps)}</b></div>
          <div class="estimate"><span>Erster Checkpoint</span><b>nach ~${fmt.duration(Math.min(v.checkpoint_every_steps, v.total_steps) / sps)}</b></div>
          <div id="new-hints"></div>
          <p class="faint small" style="margin-top:14px"><b>Wichtig:</b> Die Simulation läuft immer auf der CPU; die Grafikkarte beschleunigt nur den Lernschritt. Zu viele Prozesse bremsen: ein paar Kerne für Windows und den Lernprozess frei lassen. ${measured ? "" : "Für eine echte Zeitangabe einmal auf der Einrichtungsseite „Geschwindigkeit messen“ drücken."}</p>
        </div>
      </div>`);
  };
  const refreshHints = async (valuesNow) => {
    const el = view.querySelector("#new-hints");
    if (!el) return;
    try {
      const check = await api("/api/config/check", { method: "POST", body: { preset: form.preset, overrides: form.overrides } });
      const items = [...(check.problems || []), ...(check.hints || [])];
      el.innerHTML = items.length
        ? `<div class="hints section-sm"><b class="small">${check.ok ? "Hinweise" : "So geht das nicht"}</b><ul>${items.map((x) => `<li>${h(x)}</li>`).join("")}</ul></div>`
        : "";
    } catch { /* Hinweise sind Beiwerk */ }
    void valuesNow;
  };

  page.actions = {
    preset: (el) => { form.preset = el.dataset.preset; form.overrides = {}; view.querySelectorAll("[data-dirty]").forEach((i) => { if (i.id !== "name") delete i.dataset.dirty; }); draw(); },
    seg: (el) => {
      const key = el.dataset.key, value = Number(el.dataset.value);
      form.overrides[key] = key === "auto_curriculum" ? Boolean(value) : value;
      draw();
      refreshHints();
    },
    name: (el) => { form.name = el.value.trim(); },
    "load-teacher": (el) => busy(el, async () => {
      const info = await attempt(() => api("/api/teacher/load", { method: "POST" }), "Lehrer konnte nicht geladen werden");
      if (info?.ready) { toast("Lehrer bereit", "Nexto ist geladen."); location.reload(); }
    }),
    num: (el) => { form.overrides[el.dataset.key] = Number(el.value); delete el.dataset.dirty; draw(); },
    sizes: (el) => { form.overrides.hidden_sizes = el.value.split(/[ ,;x×]+/).filter(Boolean).map(Number); },
    start: (el) => busy(el, async () => {
      form.name = view.querySelector("#name").value.trim();
      if (!form.name) return toast("Name fehlt", "Gib dem Training einen Namen.", "error");
      const run = await attempt(() => api("/api/runs", { method: "POST", body: { name: form.name, preset: form.preset, overrides: form.overrides } }), "Training konnte nicht starten");
      if (run) { toast("Training gestartet", run.name); location.hash = `#/training/${encodeURIComponent(run.name)}`; }
    }),
  };
  draw();
  refreshHints();
}

function numberField(id, label, value, hint = "", step = "1") {
  return `<label class="field"><span>${h(label)}</span><input id="${id}" type="number" step="${step}" value="${value}" data-change="num" data-key="${id}">${hint ? `<small>${h(hint)}</small>` : ""}</label>`;
}

// ------------------------------------------------------------------ run detail

const MILESTONES = [
  [30e6, "Fährt gezielt zum Ball und trifft ihn", "20–50 Mio."],
  [150e6, "Schießt Richtung Tor, erste echte Tore", "100–300 Mio."],
  [500e6, "Schlägt Psyonix Rookie/Pro im echten Spiel", "0,3–1 Mrd."],
  [3e9, "Gold/Platin-Niveau", "mehrere Mrd."],
];

const REWARD_LABEL = {
  speed_to_ball: "Tempo zum Ball", face_ball: "Zum Ball schauen", touch: "Ballkontakt", in_air: "In der Luft",
  ball_to_goal: "Ball Richtung Tor", goal: "Tor", boost_keep: "Boost sparen", air_touch: "Luftkontakt",
};

async function pageRun(rawName) {
  const name = decodeURIComponent(rawName);
  const enc = encodeURIComponent(name);
  let selected = null, data = null;

  const load = async () => {
    const [run, metrics, log] = await Promise.all([
      api(`/api/runs/${enc}`), api(`/api/runs/${enc}/metrics`), api(`/api/runs/${enc}/log?lines=200`),
    ]);
    data = { run, metrics, log };
    draw();
  };

  const draw = () => {
    const { run, metrics, log } = data;
    const st = run.status, cfg = run.config;
    const active = ACTIVE.has(st.state);
    const total = st.total_steps || cfg.total_steps;
    const steps = st.steps || 0;
    const pct = Math.min(100, (steps / total) * 100);
    const sps = st.steps_per_second || run.last.steps_per_second;
    const eta = active && sps ? (total - steps) / sps : null;
    const last = metrics[metrics.length - 1] || {};
    const series = (key, window = 5) => {
      const values = smooth(metrics.map((m) => m[key]), window);
      return metrics.map((m, i) => [m.steps, values[i]]);
    };
    const checkpoints = [...run.checkpoints].reverse();
    if ((!selected || !checkpoints.some((c) => c.file === selected)) && checkpoints.length) selected = checkpoints.find((c) => c.file !== "latest.pt")?.file || checkpoints[0].file;
    const stageMarks = [];
    metrics.forEach((m, i) => { if (i && m.stage && metrics[i - 1].stage && m.stage !== metrics[i - 1].stage) stageMarks.push([m.steps, `Stufe ${m.stage}`, "stage"]); });
    const marks = [...checkpoints.map((c) => [c.steps]), ...stageMarks];
    // win rate against the teacher, pooled like the past-version series
    const teacherSeries = metrics.map((m, i) => {
      let games = 0, score = 0;
      for (const x of metrics.slice(Math.max(0, i - 14), i + 1)) {
        if (!x.teacher_games) continue;
        games += x.teacher_games;
        score += x.teacher_wins + 0.5 * (x.teacher_games - x.teacher_wins - x.teacher_losses);
      }
      return [m.steps, games ? (100 * score) / games : null];
    }).filter(([, v]) => v !== null);
    const teacherGoals = metrics.reduce(
      (acc, m) => [acc[0] + (m.teacher_goals_for || 0), acc[1] + (m.teacher_goals_against || 0)], [0, 0],
    );
    // win rate against older versions, pooled over the last 15 updates (single updates have few games)
    const pastSeries = metrics.map((m, i) => {
      let games = 0, score = 0;
      for (const x of metrics.slice(Math.max(0, i - 14), i + 1)) {
        if (!x.past_games) continue;
        games += x.past_games;
        score += x.past_wins + 0.5 * (x.past_games - x.past_wins - x.past_losses);
      }
      return [m.steps, games ? (score / games) * 100 : null];
    }).filter(([, v]) => v != null);
    const stageNow = last.stage || cfg.reward_stage;
    const evals = [...run.evaluations].reverse();
    const parts = last.reward_parts || {};
    const logEl = view.querySelector("#log");
    const atBottom = !logEl || logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 30;
    const lines = [...log.train, ...(st.state === "failed" ? ["", "— Prozessausgabe —", ...log.process] : [])];
    const ckptLabel = (file) => (file === "latest.pt" ? "Neuester" : fmt.steps(Number(file.replace(".pt", ""))));

    patch(view, `
      <div class="page-head"><div><div class="eyebrow"><a href="#/training">Training</a> / ${h(name)}</div>
        <div class="row"><h1>${h(name)}</h1>${pill(st.state)}</div>
        <p>${cfg.auto_curriculum ? `<span class="badge accent">Autopilot</span> ` : ""}${cfg.team_size}v${cfg.team_size} · Stufe ${stageNow}: ${h(STAGES[stageNow])} · Netz ${cfg.hidden_sizes.join("×")} · ${cfg.n_workers || "auto"} Prozesse</p></div>
        <div class="row">
          ${checkpoints.length ? `<button class="btn" data-action="live">${icon("eye")}Live zuschauen</button>` : ""}
          ${active ? `<button class="btn" data-action="stop" ${st.state === "stopping" ? "disabled" : ""}>${icon("stop")}${st.state === "stopping" ? "Stoppt …" : "Stoppen"}</button>`
                   : `<button class="btn primary" data-action="resume">${icon("play")}${steps >= total ? "Weiter trainieren" : "Fortsetzen"}</button>
                      <button class="btn ghost danger" data-action="delete">${icon("trash")}Löschen</button>`}
        </div></div>

      <div class="card"><div class="progress ${active ? "active" : ""}"><i style="width:${pct}%"></i></div>
        <div class="progress-meta"><span>${fmt.int(steps)} / ${fmt.int(total)} Schritte</span>
        <span>${active ? (eta ? `noch ca. ${fmt.duration(eta)}` : "läuft …") : `${pct.toFixed(1)} %`}</span></div></div>

      <div class="grid cols-4 section-sm">
        <div class="card stat"><div class="label">Tempo</div><div class="value">${sps && active ? fmt.int(sps) : "–"}<small>Schritte/s</small></div>
          <div class="foot">${last.realtime_factor ? `≈ ${fmt.int(last.realtime_factor)}× Echtzeit` : "&nbsp;"}${st.device ? ` · lernt auf ${h(String(st.device).toUpperCase())}` : ""}</div></div>
        <div class="card stat"><div class="label">Eigene Ballkontakte</div><div class="value">${fmt.num(last.touches_per_minute, 1)}<small>pro Minute</small></div>
          <div class="foot">Gegner: ${fmt.num(last.touches_against_per_minute, 1)} mal</div></div>
        <div class="card stat"><div class="label">Eigene Tore</div><div class="value">${fmt.num(last.goals_per_minute, 2)}<small>pro Minute</small></div>
          <div class="foot">Gegentore: ${fmt.num(last.goals_against_per_minute, 2)} pro Minute</div></div>
        <div class="card stat"><div class="label">Belohnung</div><div class="value">${fmt.num(last.episode_reward, 1)}<small>pro Episode</small></div>
          <div class="foot">${last.explained_variance != null ? `Kritiker erklärt ${Math.round(last.explained_variance * 100)} %` : "&nbsp;"}</div></div>
      </div>

      ${(data.run.hints || []).length ? `<div class="card section-sm hints"><div class="card-head"><h3>Hinweise zu dieser Einstellung</h3><span class="sub">läuft trotzdem</span></div>
        <ul>${data.run.hints.map((x) => `<li>${h(x)}</li>`).join("")}</ul></div>` : ""}

      <div class="card section-sm"><div class="card-head"><h2>Prognose</h2><span class="sub">${sps ? `bei ${fmt.int(sps)} Schritten/s` : "Tempo unbekannt"} · Erfahrungswerte der RLGym-Community</span></div>
        <div class="milestones">${MILESTONES.map(([at, label, range]) => {
          const done = steps >= at;
          const remaining = sps ? (at - steps) / sps : null;
          return `<div class="milestone ${done ? "done" : ""}"><div class="ms-bar"><i style="width:${Math.min(100, (steps / at) * 100)}%"></i></div>
            <b>${h(label)}</b><small>${range} Schritte · ${done ? "Bereich erreicht" : remaining ? `noch ~${fmt.duration(remaining)}` : "–"}</small></div>`;
        }).join("")}</div></div>

      <div class="grid cols-2 section-sm">
        <div class="card"><div class="card-head"><h3>Ballkontakte pro Minute</h3><span class="sub">geglättet · Striche = Checkpoints</span></div>${lineChart(series("touches_per_minute"), { marks })}</div>
        <div class="card"><div class="card-head"><h3>Belohnung pro Episode</h3><span class="sub">geglättet</span></div>${lineChart(series("episode_reward"), { marks })}</div>
        <div class="card"><div class="card-head"><h3>Tore pro Minute</h3><span class="sub">Selbstspiel</span></div>${lineChart(series("goals_per_minute"), { format: (v) => fmt.num(v, 2), marks })}</div>
        <div class="card"><div class="card-head"><h3>Lernsignal</h3><span class="sub">erklärte Varianz des Kritikers</span></div>${lineChart(series("explained_variance", 3), { format: (v) => fmt.num(v, 2), marks })}</div>
        <div class="card"><div class="card-head"><h3>Siegquote</h3><span class="sub">gegen ältere Versionen · über 50 % = besser als die Vorgänger</span></div>
          ${pastSeries.length > 1 ? lineChart(pastSeries, { format: (v) => `${Math.round(v)} %`, marks }) : `<div class="chart-empty">${cfg.past_opponent_prob ? "Erscheint, sobald der erste Checkpoint im Pool ist und Spiele gegen ihn fertig sind." : "Gegner-Pool ist für dieses Training ausgeschaltet."}</div>`}</div>
        ${teacherSeries.length > 1 ? `<div class="card"><div class="card-head"><h3>Gegen den Lehrer</h3><span class="sub">Siegquote und Tore gegen Nexto · ${fmt.num(teacherSeries.at(-1)[1], 0)} %</span></div>${lineChart(teacherSeries, { format: (v) => `${Math.round(v)} %`, marks })}
          <div class="estimate"><span>Tore für dich</span><b>${teacherGoals[0]}</b></div><div class="estimate"><span>Tore für den Lehrer</span><b>${teacherGoals[1]}</b></div></div>` : ""}
        <div class="card"><div class="card-head"><h3>Neugier</h3><span class="sub">Entropie der Entscheidungen · sinkt, wenn die KI sicherer wird</span></div>${lineChart(series("entropy", 3), { format: (v) => fmt.num(v, 2), marks })}</div>
      </div>

      <div class="section grid cols-2">
        <div class="card"><div class="card-head"><h2>Checkpoints</h2><span class="sub">${checkpoints.length} gespeichert</span></div>
          ${checkpoints.length ? `<div class="table-scroll"><table class="table"><thead><tr><th>Stand</th><th>Gespeichert</th><th class="num">Größe</th></tr></thead><tbody>
            ${checkpoints.map((c) => `<tr class="clickable ${c.file === selected ? "selected" : ""}" data-action="select" data-file="${h(c.file)}" data-key="ck-${h(c.file)}"><td>${c.file === "latest.pt" ? `Neuester <span class="faint">(${fmt.steps(c.steps)})</span>` : `${fmt.steps(c.steps)} Schritte`}</td><td class="faint">${fmt.ago(c.created)}</td><td class="num faint">${(c.size / 1e6).toFixed(1)} MB</td></tr>`).join("")}
          </tbody></table></div>
          <div class="ckpt-actions"><span class="faint">Auswahl: <b class="accent">${selected ? ckptLabel(selected) : "–"}</b></span><span class="spacer"></span>
            <button class="btn sm" data-action="eval">${icon("chart")}Bewerten</button>
            <button class="btn sm" data-action="live-ckpt">${icon("eye")}Live</button>
            <button class="btn sm" data-action="watch">Replay</button>
            <a class="btn sm primary" href="#/play?checkpoint=${encodeURIComponent(`${name}/${selected}`)}">Im echten Spiel</a></div>`
          : `<p class="faint">Der erste Checkpoint erscheint nach ${fmt.steps(cfg.checkpoint_every_steps)} Schritten${sps ? ` (≈ ${fmt.duration(Math.max(0, cfg.checkpoint_every_steps - steps) / sps)})` : ""}.</p>`}
        </div>
        <div class="card"><div class="card-head"><h2>Bewertungen</h2><span class="sub">Sieg/Unentschieden/Niederlage · Tore</span></div>
          ${evals.length ? `<div class="table-scroll"><table class="table"><thead><tr><th>Stand</th><th>Gegner</th><th class="num">S/U/N</th><th class="num">Tore</th><th class="num">Punkte</th></tr></thead><tbody>
            ${evals.slice(0, 10).flatMap((e) => e.results.map((r, i) => `<tr><td>${i ? "" : fmt.steps(e.steps) + (e.manual ? " <span class='faint'>(manuell)</span>" : "")}</td><td>${h(opponentLabel(r.opponent))}</td><td class="num">${r.wins}/${r.draws}/${r.losses}</td><td class="num">${r.goals_for}:${r.goals_against}</td><td class="num"><b class="${r.score >= 0.5 ? "accent" : ""}">${Math.round(r.score * 100)} %</b></td></tr>`)).join("")}
          </tbody></table></div>` : `<p class="faint">${cfg.eval_every_steps ? `Automatisch alle ${fmt.steps(cfg.eval_every_steps)} Schritte – oder einen Checkpoint auswählen und „Bewerten“.` : "Automatische Bewertung ist aus. Wähle einen Checkpoint und klicke „Bewerten“."}</p>`}
        </div>
      </div>

      <div class="section grid cols-2">
        <div class="card"><div class="card-head"><h2>Belohnungsanteile</h2><span class="sub">pro Schritt, letzte Iteration</span></div>
          ${Object.keys(parts).length ? `<div class="bars">${rewardBars(parts)}</div>` : `<p class="faint">Noch keine Daten.</p>`}
        </div>
        <div class="card"><div class="card-head"><h2>Protokoll</h2><span class="sub">train.log · letzte 200 Zeilen</span></div>
          <pre class="log" id="log">${h(lines.join("\n")) || "Noch keine Ausgabe."}</pre></div>
      </div>`);

    const newLog = view.querySelector("#log");
    if (newLog && atBottom) newLog.scrollTop = newLog.scrollHeight;
  };

  page.actions = {
    select: (el) => { selected = el.dataset.file; draw(); },
    live: (el) => busy(el, () => startLive(`run:${name}`)),
    "live-ckpt": (el) => busy(el, () => startLive(selected === "latest.pt" ? `run:${name}` : `${name}/${selected}`)),
    stop: (el) => busy(el, async () => {
      if (await attempt(() => api(`/api/runs/${enc}/stop`, { method: "POST" }))) { toast("Stopp angefordert", "Die aktuelle Runde wird noch beendet und gespeichert."); await load(); }
    }),
    resume: async (el) => {
      const st = data.run.status, total = st.total_steps || data.run.config.total_steps;
      let body = {};
      if ((st.steps || 0) >= total) {
        const answer = await modal({
          title: "Weiter trainieren", body: `<p class="muted">Das Ziel von ${fmt.steps(total)} Schritten ist erreicht. Auf welches neue Gesamtziel soll weiter trainiert werden?</p>`,
          fields: [{ id: "total", label: "Neues Ziel (Gesamtschritte)", type: "number", value: total * 2, min: total + 1000, step: 1000000, hint: "Das Training setzt beim letzten Stand fort." }],
          confirm: "Weiter trainieren",
        });
        if (!answer) return;
        if (!(answer.total > total)) return toast("Ungültiges Ziel", `Bitte mehr als ${fmt.int(total)} Schritte angeben.`, "error");
        body = { total_steps: Math.round(answer.total) };
      }
      await busy(el, async () => {
        if (await attempt(() => api(`/api/runs/${enc}/resume`, { method: "POST", body }))) { toast("Training läuft wieder"); await load(); }
      });
    },
    delete: async (el) => {
      const ok = await modal({ title: "Training löschen?", body: `<p class="muted">„${h(name)}“ wird mit allen Checkpoints, Messwerten und Replays gelöscht. Das kann nicht rückgängig gemacht werden.</p>`, confirm: "Endgültig löschen", danger: true });
      if (!ok) return;
      await busy(el, async () => {
        if (await attempt(() => api(`/api/runs/${enc}`, { method: "DELETE" }))) { toast("Gelöscht", name); location.hash = "#/training"; }
      });
    },
    eval: (el) => busy(el, async () => {
      const job = await attempt(() => api("/api/evaluate", { method: "POST", body: { checkpoint: `${name}/${selected}`, games: 6 } }));
      if (job) { toast("Bewertung läuft", "6 Spiele je Gegner, dauert etwa eine Minute."); waitForJob(job.id, () => { toast("Bewertung fertig"); load(); }); }
    }),
    watch: (el) => busy(el, async () => {
      const job = await attempt(() => api("/api/matches", { method: "POST", body: { blue: `${name}/${selected}`, orange: "chaser", team_size: data.run.config.team_size, seconds: 120 } }));
      if (job) { toast("Match wird simuliert", "Gegner: Balljäger – das Replay öffnet sich gleich."); waitForJob(job.id, (done) => { location.hash = `#/arena/${done.result.replay}`; }); }
    }),
  };

  await load();
  refreshOnSnapshot(load);
  const timer = setInterval(() => { if (data && ACTIVE.has(data.run.status.state) && !document.hidden) load().catch(() => {}); }, 5000);
  page.cleanup.push(() => clearInterval(timer));
}

function rewardBars(parts) {
  const entries = Object.entries(parts);
  const max = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-9);
  return entries.map(([k, v]) => `<div class="bar-row"><span>${h(REWARD_LABEL[k] || k)}</span><div class="bar"><i class="${v < 0 ? "neg" : ""}" style="width:${(Math.abs(v) / max) * 100}%"></i></div><b class="mono">${v.toFixed(4)}</b></div>`).join("");
}

// ------------------------------------------------------------------ live

const CONTROL_NAMES = ["Gas", "Lenken", "Nicken", "Gieren", "Rollen"];

async function pageLive() {
  const [opponents, pads] = await Promise.all([api("/api/opponents"), fieldPads()]);
  const runs = [...new Set(opponents.checkpoints.map((c) => c.run))];
  const form = {
    blue: runs.length ? `run:${runs[0]}` : "chaser",
    orange: "chaser", team_size: 1, speed: 1, match_seconds: 300,
  };
  const meta = { active: false, seq: -1, speed: 1, paused: false };
  const buffer = [];
  const values = new Map(); // car index → recent critic values
  let focusCar = null; // null = first AI car
  let stage = null, pos = null, lastTs = null, lastHud = 0, raf = 0, pollTimer = 0, stopped = false, lastFloor = null;

  const options = (selectedId) => [
    runs.length ? `<optgroup label="Folgt dem Training (immer neuester Stand)">${runs.map((r) => `<option value="run:${h(r)}" ${`run:${r}` === selectedId ? "selected" : ""}>${h(r)} · live</option>`).join("")}</optgroup>` : "",
    `<optgroup label="Eingebaute Gegner">${opponents.scripted.map((o) => `<option value="${o.id}" ${o.id === selectedId ? "selected" : ""}>${h(o.label)}</option>`).join("")}</optgroup>`,
    opponents.teacher?.ready ? `<optgroup label="Lehrer"><option value="teacher" ${selectedId === "teacher" ? "selected" : ""}>Lehrer (Nexto, Grand Champion)</option></optgroup>` : "",
    ...Object.entries(groupBy(opponents.checkpoints.filter((c) => !c.id.endsWith("latest.pt")), (c) => c.run)).map(([run, list]) =>
      `<optgroup label="${h(run)} – fester Stand">${list.slice().reverse().map((c) => `<option value="${h(c.id)}" ${c.id === selectedId ? "selected" : ""}>${h(run)} · ${fmt.steps(c.steps)}</option>`).join("")}</optgroup>`),
  ].join("");

  const setupHtml = () => `
    <div class="card form live-setup">
      <h2>Live-Spiel starten</h2>
      <label class="field"><span><i class="team-dot blue"></i>Blau</span><select data-change="form" data-key="blue">${options(form.blue)}</select></label>
      <label class="field"><span><i class="team-dot orange"></i>Orange</span><select data-change="form" data-key="orange">${options(form.orange)}</select></label>
      <div class="grid cols-2">
        <div class="field"><span class="field-label">Modus</span>${seg("team_size", [[1, "1v1"], [2, "2v2"], [3, "3v3"]], form.team_size, "form-seg")}</div>
        <div class="field"><span class="field-label">Spiellänge</span>${seg("match_seconds", [[120, "2 min"], [300, "5 min"], [1800, "Endlos"]], form.match_seconds, "form-seg")}</div>
      </div>
      <button class="btn primary" data-action="start">${icon("play")}Live-Spiel starten</button>
      <p class="faint small">Läuft in der Simulation (RocketSim) in Echtzeit. „Folgt dem Training“ lädt automatisch jeden neuen Checkpoint, während das Training läuft.</p>
    </div>`;

  const shell = () => `
    <div class="page-head"><div><div class="eyebrow">Live</div><div class="row"><h1>Live-Spiel</h1><span id="live-pill">${pill(meta.active ? "live" : "idle", meta.active ? (meta.paused ? "Pausiert" : "Live") : "Aus")}</span></div>
      <p>Sieh deiner KI in 3D beim Spielen zu – und darunter, was in ihrem neuronalen Netz gerade passiert.</p></div>
      <div class="row" id="live-controls"></div></div>
    <div class="live-layout">
      <div class="stage-wrap" id="stage-wrap">
        <div class="stage-head">
          <div class="scoreboard">
            <div class="team blue"><span id="blue-name">–</span><span class="swatch"></span></div>
            <div class="score-box"><div class="score" id="score">0 : 0</div><div class="clock" id="clock">0:00</div></div>
            <div class="team orange"><span class="swatch"></span><span id="orange-name">–</span></div>
          </div>
          ${stageToolbar()}
        </div>
        <div class="stage-host" id="stage-host" data-static><div class="stage-empty" id="stage-empty"><div>${icon("eye")}<b>Noch kein Live-Spiel</b><span>Rechts einstellen und starten.</span></div></div></div>
        <div class="hud-boost" id="hud-boost" data-static></div>
      </div>
      <aside class="brain card" id="brain-panel"></aside>
    </div>
    <div class="section grid cols-2">
      <div class="card"><div class="card-head"><h2>Ereignisse</h2><span class="sub" id="match-no"></span></div><div class="events" id="events" data-static></div></div>
      <div class="card"><div class="card-head"><h2>So liest du das Gehirn</h2></div>
        <dl class="explain">
          <dt>Entscheidung</dt><dd>Die KI wählt 15-mal pro Sekunde eine von 90 Aktionen – eine Kombination aus Gas, Lenken, Springen, Boost usw.</dd>
          <dt>Sicherheit</dt><dd>Wie eindeutig die Wahl war. Niedrig heißt: Viele Aktionen schienen ihr ähnlich gut. Frühe Trainingsstände sind oft unsicher.</dd>
          <dt>Erwartung</dt><dd>Der „Kritiker“ schätzt, wie viel Belohnung noch kommt. Steigt die Kurve, glaubt die KI, dass die Lage gut für sie ist (z. B. kurz vor einem Ballkontakt).</dd>
          <dt>Was die KI sieht</dt><dd>Die Eingaben des Netzes aus dem Spielzustand: Ballabstand, Höhe, Drehung, Boost, ob der Ball in unserer Hälfte liegt und ob er auf unser Tor zuläuft. Es sind exakt die Zahlen, die auch im Training ankommen – die KI „sieht“ kein Bild, sondern diese Werte.</dd>
          <dt>Alternativen</dt><dd>Die fünf wahrscheinlichsten Aktionen mit ihrer Wahrscheinlichkeit.</dd>
        </dl></div>
    </div>`;

  const controlsHtml = () => meta.active ? `
      <button class="btn" data-action="pause">${meta.paused ? `${icon("play")}Weiter` : `${icon("pause")}Pause`}</button>
      ${seg("speed", [[0.5, "0,5×"], [1, "1×"], [2, "2×"], [4, "4×"]], meta.speed, "speed")}
      <button class="btn ghost danger" data-action="stop">${icon("stop")}Beenden</button>` : "";

  const brainEmpty = () => `<div class="brain-empty">${icon("brain")}<b>Kein KI-Gehirn</b><span>${meta.active ? "In diesem Spiel steuert keine trainierte KI ein Auto. Wähle für Blau oder Orange ein Training." : "Starte ein Spiel mit einer trainierten KI, um ihre Entscheidungen zu sehen."}</span></div>`;

  view.innerHTML = shell();
  const $ = (id) => view.querySelector(`#${id}`);
  const brainPanel = $("brain-panel");

  const renderSide = () => {
    patch($("live-controls"), controlsHtml());
    patch($("live-pill"), pill(meta.active ? "live" : "idle", meta.active ? (meta.paused ? "Pausiert" : "Live") : "Aus"));
    if (!meta.active) {
      patch(brainPanel, setupHtml());
      brainPanel.classList.add("setup");
    } else if (brainPanel.classList.contains("setup") || !brainPanel.querySelector(".brain-live, .brain-empty")) {
      brainPanel.classList.remove("setup");
      brainPanel.innerHTML = meta.hasBrain ? brainSkeleton() : brainEmpty();
    }
  };

  const brainSkeleton = () => `<div class="brain-live">
      <div class="brain-head"><div>${icon("brain")}<b>Gehirn</b></div><span class="faint small" id="brain-who"></span></div>
      <div class="decision"><small>Entscheidung</small><div class="decision-label" id="decision">–</div></div>
      <div class="meter"><div class="meter-head"><span>Sicherheit</span><b id="conf-val">–</b></div><div class="meter-bar"><i id="conf-bar"></i></div></div>
      <div class="controller">
        ${CONTROL_NAMES.map((n, i) => `<div class="axis-row"><span>${n}</span><div class="axis"><i id="ax${i}"></i></div></div>`).join("")}
        <div class="buttons"><span class="btn-led" id="b-jump">Sprung</span><span class="btn-led" id="b-boost">Boost</span><span class="btn-led" id="b-drift">Drift</span></div>
      </div>
      <div><div class="meter-head"><span>Erwartung (Kritiker)</span><b id="value-val">–</b></div><div id="value-spark" class="spark-wrap"></div></div>
      <div><div class="meter-head"><span>Eingaben der letzten 5 s</span><span class="faint small">links alt · rechts jetzt</span></div>
        <div class="history"><div class="history-labels"><span>Gas</span><span>Lenken</span><span>Sprung</span><span>Boost</span><span>Drift</span></div><canvas id="history" width="300" height="90"></canvas></div></div>
      <div><div class="meter-head"><span>Was die KI sieht</span><span class="faint small" id="sees-who">Eingaben des Netzes</span></div>
        <div id="sees" class="sees"></div></div>
      <div><div class="meter-head"><span>Alternativen</span><span class="faint small">Wahrscheinlichkeit</span></div><div id="top5" class="top5"></div></div>
    </div>`;

  // „Was die KI sieht": die Zusatzwerte der Beobachtung (siehe rocketai/obs.py).
  // Die Legende kommt vom Server (/api/live → features), damit Python und
  // Oberfläche nie auseinanderlaufen.
  const seesRows = (features) => (features || []).map((f) => {
    const key = f.key, format = f.format || "signed";
    return `<div class="see-row" data-see="${h(key)}" title="${h(key)}">
      <span class="see-label">${h(f.label)}</span>
      <div class="see-bar"><i></i></div>
      <b class="see-val">–</b></div>`;
  }).join("");

  const updateSees = (brain) => {
    const el = $("sees");
    if (!el || !brain) return;
    if (!el.dataset.ready) { el.innerHTML = seesRows(meta.features); el.dataset.ready = "1"; }
    const sees = brain.sees || {};
    el.querySelectorAll(".see-row").forEach((row) => {
      const value = sees[row.dataset.see];
      const bar = row.querySelector("i"), out = row.querySelector(".see-val");
      if (value == null) { bar.style.width = "0%"; out.textContent = "–"; return; }
      const rowMeta = (meta.features || []).find((f) => f.key === row.dataset.see) || {};
      const format = rowMeta.format || "signed";
      if (format === "flag") {
        bar.style.left = "0%"; bar.style.width = value >= 0.5 ? "100%" : "0%";
        out.textContent = value >= 0.5 ? "ja" : "nein";
      } else if (format === "percent") {
        bar.style.left = "0%"; bar.style.width = `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
        out.textContent = `${Math.round(value * 100)} %`;
      } else {
        bar.style.left = value < 0 ? `${50 + value * 50}%` : "50%";
        bar.style.width = `${Math.abs(Math.max(-1, Math.min(1, value))) * 50}%`;
        out.textContent = value >= 0 ? `+${fmt.num(value, 2)}` : fmt.num(value, 2);
      }
      row.classList.toggle("warn", row.dataset.see === "own_goal_danger" && value > 0.5);
    });
  };

  const updateBrain = (brain, frame) => {
    if (!brain || !brainPanel.querySelector(".brain-live")) return;
    const team = brain.team === 0 ? "Blau" : "Orange";
    $("brain-who").textContent = `${team} · Auto ${brain.car + 1}${meta.brainCount > 1 ? " · Auto unten links wechseln" : ""}`;
    $("decision").textContent = brain.label;
    $("conf-val").textContent = fmt.pct(brain.confidence);
    $("conf-bar").style.width = `${Math.round(brain.confidence * 100)}%`;
    const c = brain.controls;
    // throttle, steer, pitch, yaw, roll are −1..1 → bar from the centre
    [c[0], c[1], c[2], c[3], c[4]].forEach((v, i) => {
      const bar = $(`ax${i}`);
      bar.style.left = v < 0 ? `${50 + v * 50}%` : "50%";
      bar.style.width = `${Math.abs(v) * 50}%`;
    });
    $("b-jump").classList.toggle("on", c[5] > 0);
    $("b-boost").classList.toggle("on", c[6] > 0);
    $("b-drift").classList.toggle("on", c[7] > 0);
    $("value-val").textContent = fmt.num(brain.value, 1);
    $("value-spark").innerHTML = sparkline(values.get(brain.car) || [], { width: 300, height: 54 });
    updateSees(brain);
    drawHistory(brain.car);
    $("top5").innerHTML = brain.top.map(([label, p], i) => `<div class="top-row ${i === 0 && label === brain.label ? "chosen" : ""}"><span>${h(label)}</span><div class="bar"><i style="width:${Math.max(2, p * 100 / Math.max(brain.top[0][1], 0.01))}%"></i></div><b>${fmt.pct(p)}</b></div>`).join("");
    void frame;
  };

  const drawHistory = (car) => {
    const canvas = $("history");
    if (!canvas) return;
    const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height, rows = 5, rowH = H / rows;
    ctx.clearRect(0, 0, W, H);
    const end = Math.floor(pos ?? 0) - (buffer[0]?.seq ?? 0);
    const slice = buffer.slice(Math.max(0, end - 74), end + 1);
    const colW = W / 75;
    slice.forEach((frame, n) => {
      const b = (frame.brains || []).find((x) => x.car === car);
      if (!b) return;
      const x = (75 - slice.length + n) * colW;
      const c = b.controls;
      const cells = [c[0], c[1], c[5], c[6], c[7]];
      cells.forEach((v, r) => {
        if (!v) return;
        const signed = r < 2;
        ctx.fillStyle = signed ? (v > 0 ? `rgba(198,244,50,${0.25 + 0.75 * v})` : `rgba(255,154,77,${0.25 + 0.75 * -v})`) : "rgba(198,244,50,.9)";
        ctx.fillRect(x, r * rowH + 2, Math.max(1, colW - 0.5), rowH - 4);
      });
    });
    ctx.fillStyle = "rgba(255,255,255,.04)";
    for (let r = 1; r < rows; r++) ctx.fillRect(0, r * rowH, W, 1);
  };

  const updateBoost = (scene) => {
    const el = $("hud-boost");
    if (!el) return;
    const aiCars = new Set((meta.lastBrains || []).map((b) => b.car));
    const html = scene.cars.map((car, i) => `<button type="button" class="boost-pill ${car.team ? "orange" : "blue"} ${scene.focus === i ? "focus" : ""} ${aiCars.has(i) ? "ai" : ""}" data-car="${i}" title="${aiCars.has(i) ? "Gehirn dieses Autos anzeigen" : "Eingebauter Bot – kein KI-Gehirn"}"><span>${aiCars.has(i) ? "KI" : "Bot"}</span><b>${Math.round(car.boost)}</b><div><i style="height:${car.boost}%"></i></div></button>`).join("");
    if (el.innerHTML !== html) el.innerHTML = html;
  };

  const addEvents = (events) => {
    const el = $("events");
    for (const e of events) {
      const row = document.createElement("div");
      row.className = `event ${e.kind}`;
      row.innerHTML = `<span class="t">${new Date(e.t * 1000).toLocaleTimeString("de-DE")}</span><span>${h(e.text)}</span>`;
      el.prepend(row);
    }
    while (el.children.length > 40) el.lastChild.remove();
    if (!el.children.length) el.innerHTML = `<p class="faint">Tore, neue Trainingsstände und Spielenden erscheinen hier.</p>`;
    else el.querySelector("p.faint")?.remove();
  };

  const applyState = (state) => {
    const wasActive = meta.active, hadBrain = meta.hasBrain;
    meta.active = state.active;
    if (!state.active && state.seq == null) { renderSide(); return; }
    meta.paused = state.paused; meta.speed = state.speed;
    meta.score = state.score;
    meta.hasBrain = state.blue?.kind !== "scripted" || state.orange?.kind !== "scripted";
    meta.match = state.match;
    meta.features = state.features || meta.features;
    $("blue-name").textContent = state.blue?.label || "–";
    $("orange-name").textContent = state.orange?.label || "–";
    $("match-no").textContent = state.match ? `Spiel ${state.match}${state.history?.length ? ` · bisher ${state.history.map((x) => `${x.blue}:${x.orange}`).join(", ")}` : ""}` : "";
    if (state.events?.length) addEvents(state.events);
    for (const f of state.frames || []) {
      buffer.push(f);
      for (const b of f.brains || []) {
        const list = values.get(b.car) || [];
        list.push(b.value);
        if (list.length > 150) list.shift();
        values.set(b.car, list);
      }
    }
    if (state.seq != null) meta.seq = state.seq;
    if (state.error) toast("Live-Spiel gestoppt", state.error, "error");
    if (wasActive !== meta.active || hadBrain !== meta.hasBrain) {
      if (meta.active) { brainPanel.innerHTML = ""; brainPanel.classList.remove("setup"); }
      renderSide();
    } else patch($("live-controls"), controlsHtml());
    patch($("live-pill"), pill(meta.active ? "live" : "idle", meta.active ? (meta.paused ? "Pausiert" : "Live") : "Aus"));
  };

  const poll = async () => {
    if (stopped) return;
    try {
      const state = await api(`/api/live?since=${meta.seq}`);
      if (!state.active && state.seq != null && state.seq < meta.seq) { meta.seq = -1; buffer.length = 0; pos = null; }
      applyState(state);
    } catch { /* retry */ }
    if (!stopped) pollTimer = setTimeout(poll, meta.active ? 150 : 2000);
  };

  const frame = (ts) => {
    if (stopped) return;
    const dt = lastTs == null ? 0 : Math.min(0.25, (ts - lastTs) / 1000);
    lastTs = ts;
    if (buffer.length && stage) {
      const newest = buffer[buffer.length - 1].seq;
      if (pos == null || pos < buffer[0].seq) pos = Math.max(buffer[0].seq, newest - 3);
      const lag = newest - pos;
      if (!meta.paused) pos += dt * 15 * meta.speed * (1 + Math.max(-0.5, Math.min(0.8, (lag - 4) * 0.1)));
      if (newest - pos > 45) pos = newest - 4;
      pos = Math.min(pos, newest);
      const base = buffer[0].seq;
      const i = Math.floor(pos) - base;
      const A = buffer[Math.max(0, Math.min(buffer.length - 1, i))], B = buffer[Math.max(0, Math.min(buffer.length - 1, i + 1))];
      const scene = blend(A.f, B.f, pos - Math.floor(pos));
      const start = Math.max(0, i - 40);
      scene.trail = buffer.slice(start, i + 1).map((x) => x.f[1]);
      const brains = A.brains || B.brains || [];
      meta.lastBrains = brains;
      meta.brainCount = brains.length;
      const brain = brains.find((b) => b.car === focusCar) || brains[0] || null;
      scene.focus = brain ? brain.car : focusCar ?? 0;
      stage.render(scene);
      // goals between the last and the current playback position
      const floor = Math.floor(pos);
      if (lastFloor != null && floor > lastFloor) {
        buffer.forEach((f, n) => {
          if (f.seq > lastFloor && f.seq <= floor && f.goal != null) {
            stage.goal(f.goal, $(f.goal === 0 ? "blue-name" : "orange-name")?.textContent, buffer[Math.max(0, n - 1)].f[1]);
          }
        });
      }
      lastFloor = floor;
      $("stage-empty")?.remove();
      if (ts - lastHud > 90) {
        lastHud = ts;
        const t = A.f[0];
        $("clock").textContent = meta.active ? `${fmt.clock(t)}${meta.paused ? " · Pause" : ""}` : "beendet";
        $("score").textContent = meta.score ? `${meta.score[0]} : ${meta.score[1]}` : "0 : 0";
        updateBoost(scene);
        updateBrain(brain, A);
      }
      if (buffer.length > 600) buffer.splice(0, buffer.length - 450);
    }
    raf = requestAnimationFrame(frame);
  };

  page.actions = {
    ...rlActions,
    form: (el) => { form[el.dataset.key] = el.value; },
    "form-seg": (el) => { form[el.dataset.key] = Number(el.dataset.value); patch(brainPanel, setupHtml()); },
    start: (el) => busy(el, async () => {
      buffer.length = 0; values.clear(); pos = null; lastFloor = null; focusCar = null; meta.seq = -1;
      $("events").innerHTML = "";
      const result = await attempt(() => api("/api/live/start", { method: "POST", body: form }), "Live-Spiel konnte nicht starten");
      if (result) { applyState(result); toast("Live-Spiel gestartet", `${result.blue.label} gegen ${result.orange.label}`); }
    }),
    pause: (el) => busy(el, async () => { const r = await attempt(() => api("/api/live/control", { method: "POST", body: { paused: !meta.paused } })); if (r) { meta.paused = r.paused; renderSide(); } }),
    speed: (el) => busy(el, async () => { const r = await attempt(() => api("/api/live/control", { method: "POST", body: { speed: Number(el.dataset.value) } })); if (r) { meta.speed = r.speed; renderSide(); } }),
    stop: (el) => busy(el, async () => { const r = await attempt(() => api("/api/live/stop", { method: "POST" })); if (r) { meta.active = false; renderSide(); } }),
  };
  page.cleanup.push(() => { stopped = true; clearTimeout(pollTimer); cancelAnimationFrame(raf); stage?.dispose(); });
  $("hud-boost").addEventListener("click", (e) => {
    const pill = e.target.closest("[data-car]");
    if (!pill) return;
    const car = Number(pill.dataset.car);
    if (!(meta.lastBrains || []).some((b) => b.car === car)) return toast("Kein KI-Gehirn", "Dieses Auto steuert ein eingebauter Bot.");
    focusCar = car;
    lastHud = 0;
  });

  renderSide();
  addEvents([]);
  stage = await createStage($("stage-wrap"), $("stage-host"), { pads });
  await poll();
  raf = requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ arena

async function pageArena(replayId) {
  const [replays, opponents, pads] = await Promise.all([api("/api/replays"), api("/api/opponents"), fieldPads()]);
  const current = replayId || (replays[0] && replays[0].id);
  const options = (selectedId) => [
    `<optgroup label="Eingebaute Gegner">${opponents.scripted.map((o) => `<option value="${o.id}" ${o.id === selectedId ? "selected" : ""}>${h(o.label)}</option>`).join("")}</optgroup>`,
    opponents.teacher?.ready ? `<optgroup label="Lehrer"><option value="teacher" ${selectedId === "teacher" ? "selected" : ""}>Lehrer (Nexto, Grand Champion)</option></optgroup>` : "",
    ...Object.entries(groupBy(opponents.checkpoints, (c) => c.run)).map(([run, list]) =>
      `<optgroup label="${h(run)}">${list.slice().reverse().map((c) => `<option value="${h(c.id)}" ${c.id === selectedId ? "selected" : ""}>${h(run)} · ${c.id.endsWith("latest.pt") ? "neuester" : fmt.steps(c.steps)}</option>`).join("")}</optgroup>`),
  ].join("");
  const latest = opponents.checkpoints.filter((c) => !c.id.endsWith("latest.pt")).at(-1);

  view.innerHTML = `
    <div class="page-head"><div><div class="eyebrow">Arena</div><h1>Arena</h1>
      <p>Simulierte Spiele als Replay – in 3D oder als Draufsicht. Jede Bewertung speichert automatisch ein Replay, oder lass hier zwei Gegner antreten.</p></div>
      <a class="btn" href="#/live">${icon("eye")}Lieber live zuschauen</a></div>
    <div class="arena-layout">
      <div class="stage-wrap" id="stage-wrap">${current ? "" : `<div class="empty"><h3>Kein Replay ausgewählt</h3><p>Starte rechts ein Match.</p></div>`}</div>
      <div class="grid arena-side">
        <div class="card form"><h2>Neues Match</h2>
          <label class="field"><span><i class="team-dot blue"></i>Blau</span><select id="blue">${options(latest ? latest.id : "chaser")}</select></label>
          <label class="field"><span><i class="team-dot orange"></i>Orange</span><select id="orange">${options("chaser")}</select></label>
          <div class="grid cols-2">
            <label class="field"><span>Modus</span><select id="size"><option value="1">1v1</option><option value="2">2v2</option><option value="3">3v3</option></select></label>
            <label class="field"><span>Dauer</span><select id="secs"><option value="60">1 min</option><option value="120" selected>2 min</option><option value="300">5 min</option></select></label>
          </div>
          <button class="btn primary" data-action="go">Simulieren</button></div>
        <div class="card"><div class="card-head"><h2>Replays</h2><span class="sub">${replays.length}</span></div>
          ${replays.length ? `<div class="replay-list">${replays.map((r) => replayItem(r, current)).join("")}</div>` : `<p class="faint">Noch keine Replays.</p>`}</div>
      </div>
    </div>`;
  page.actions = {
    go: (el) => busy(el, async () => {
      const body = { blue: view.querySelector("#blue").value, orange: view.querySelector("#orange").value, team_size: Number(view.querySelector("#size").value), seconds: Number(view.querySelector("#secs").value) };
      const job = await attempt(() => api("/api/matches", { method: "POST", body }));
      if (!job) return;
      toast("Match wird simuliert", "Das dauert nur ein paar Sekunden.");
      await waitForJob(job.id, (done) => { location.hash = `#/arena/${done.result.replay}`; });
    }),
  };
  view.querySelector(".replay-item.on")?.scrollIntoView({ block: "nearest" });
  if (current) {
    const replay = await attempt(() => api(`/api/replays/${current}`), "Replay konnte nicht geladen werden");
    if (replay) await mountReplay(view.querySelector("#stage-wrap"), replay, pads);
  }
}

async function mountReplay(wrap, replay, pads) {
  const frames = replay.frames, meta = replay.meta, fps = replay.fps || 15;
  const duration = frames.length ? frames[frames.length - 1][0] : 0;
  const goals = meta.goal_times || [];
  wrap.innerHTML = `
    <div class="stage-head">
      <div class="scoreboard">
        <div class="team blue"><span>${h(opponentLabel(meta.blue))}</span><span class="swatch"></span></div>
        <div class="score-box"><div class="score" id="score">0 : 0</div><div class="clock" id="clock">0:00</div></div>
        <div class="team orange"><span class="swatch"></span><span>${h(opponentLabel(meta.orange))}</span></div>
      </div>
      ${stageToolbar()}
    </div>
    <div class="stage-host" id="stage-host"></div>
    <div class="player-controls">
      <button class="btn icon-btn" id="toggle" title="Abspielen/Pause (Leertaste)"></button>
      <div class="timeline" id="timeline"><div class="track"></div><div class="fill" id="fill"></div>
        ${goals.map(([t, team]) => `<span class="goal-mark ${team === 0 ? "blue" : "orange"}" style="left:${(t / duration) * 100}%" title="Tor bei ${fmt.clock(t)}"></span>`).join("")}
        <div class="knob" id="knob"></div></div>
      <div class="segmented sm" id="speed">${[0.5, 1, 2, 4].map((s) => `<button data-speed="${s}" class="${s === 1 ? "on" : ""}">${String(s).replace(".", ",")}×</button>`).join("")}</div>
    </div>`;
  const $ = (id) => wrap.querySelector(`#${id}`);
  const stage = await createStage(wrap, $("stage-host"), { pads });
  let t = 0, playing = true, speed = 1, lastTs = null, raf = 0, dragging = false, lastHud = 0;
  const toggle = $("toggle");
  const setPlaying = (p) => { playing = p; toggle.innerHTML = icon(p ? "pause" : "play"); if (p && t >= duration) t = 0; };
  setPlaying(true);
  toggle.addEventListener("click", () => setPlaying(!playing));
  wrap.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => {
    speed = Number(b.dataset.speed);
    wrap.querySelectorAll("[data-speed]").forEach((x) => x.classList.toggle("on", x === b));
  }));
  const timeline = $("timeline");
  const seek = (event) => { const r = timeline.getBoundingClientRect(); t = Math.max(0, Math.min(1, (event.clientX - r.left) / r.width)) * duration; };
  timeline.addEventListener("pointerdown", (e) => { dragging = true; timeline.setPointerCapture(e.pointerId); seek(e); });
  timeline.addEventListener("pointermove", (e) => dragging && seek(e));
  timeline.addEventListener("pointerup", () => { dragging = false; });
  const onKey = (e) => {
    if (e.target.closest("input, select, textarea")) return;
    if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); }
    if (e.key === "ArrowRight") t = Math.min(duration, t + 5);
    if (e.key === "ArrowLeft") t = Math.max(0, t - 5);
  };
  document.addEventListener("keydown", onKey);

  const loop = (ts) => {
    const before = t;
    if (lastTs != null && playing && !dragging) {
      t += ((ts - lastTs) / 1000) * speed;
      if (t >= duration) { t = duration; setPlaying(false); }
      for (const [gt, team] of goals) {
        if (gt > before && gt <= t && team >= 0) {
          const k = Math.max(0, Math.round(gt * fps) - 1);
          stage.goal(team, opponentLabel(team === 0 ? meta.blue : meta.orange), frames[Math.min(frames.length - 1, k)][1]);
        }
      }
    }
    lastTs = ts;
    const f = Math.min(frames.length - 1, Math.max(0, t * fps));
    const i = Math.floor(f);
    const scene = blend(frames[i], frames[Math.min(frames.length - 1, i + 1)], f - i);
    scene.trail = frames.slice(Math.max(0, i - 40), i + 1).map((x) => x[1]);
    stage.render(scene);
    if (ts - lastHud > 60) {
      lastHud = ts;
      const scored = goals.filter(([gt]) => gt <= t + 1e-6);
      $("score").textContent = `${scored.filter((g) => g[1] === 0).length} : ${scored.filter((g) => g[1] === 1).length}`;
      $("clock").textContent = `${fmt.clock(t)} / ${fmt.clock(duration)}`;
      const pct = duration ? (t / duration) * 100 : 0;
      $("fill").style.width = `${pct}%`;
      $("knob").style.left = `${pct}%`;
    }
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
  page.cleanup.push(() => { cancelAnimationFrame(raf); document.removeEventListener("keydown", onKey); stage.dispose(); });
}

// ------------------------------------------------------------------ play

const MODE_HINT = {
  psyonix: "Die eingebauten Bots von Beginner bis All-Star",
  human: "Du spielst selbst gegen deine KI",
  bot: "Jeder RLBot-v5-Bot, z. B. aus dem Botpack",
  self: "Zwei Kopien deiner KI gegeneinander",
};

async function pagePlay(_unused, query) {
  const [opponents] = await Promise.all([api("/api/opponents"), app.rl ? null : pollRocketLeague()]);
  const preselect = query && query.get("checkpoint");
  const usable = opponents.checkpoints;
  const form = {
    checkpoint: preselect || (usable.filter((c) => !c.id.endsWith("latest.pt")).at(-1) || usable.at(-1) || {}).id || "",
    brain: "policy",
    mode: "psyonix", team_size: 1, skill: "rookie", launcher: app.rl?.store === "Steam" ? "steam" : "epic", opponent_bot: "",
  };
  const teacherReady = Boolean(opponents.teacher?.ready);
  const pyCmd = opponents.python || "python3";
  let info = await api("/api/play");

  const draw = () => {
    const rl = app.rl;
    const busyState = info.state === "starting" || info.state === "running";
    const blocked = !info.windows ? "Nur unter Windows möglich." : !info.server_installed ? `RLBotServer fehlt – einmal ${pyCmd} install.py ausführen.` : rl?.game === "normal" ? "Rocket League läuft normal – bitte zuerst schließen." : form.brain === "teacher" ? (!teacherReady ? "Der Lehrer ist noch nicht geladen." : "") : !form.checkpoint ? "Erst ein Training laufen lassen." : "";
    patch(view, `
      <div class="page-head"><div><div class="eyebrow">Spielen</div><h1>Im echten Rocket League</h1>
        <p>RLBot startet Rocket League und ein <b>Offline-Match</b>, in dem deine KI ein Auto steuert – gegen Psyonix-Bots, Community-Bots oder dich selbst.</p></div>
        <div>${pill(info.state)}</div></div>
      <div class="note"><div><b>Nur offline.</b> Rocket League nutzt seit April 2026 Easy Anti-Cheat. Bots laufen ausschließlich in lokalen Matches, die RLBot startet (Spiel ohne Anti-Cheat) – nie online, in Ranked oder privaten Online-Matches.</div></div>
      <div class="split section-sm">
        <div class="grid">
          ${rlCard(rl, { compact: true })}
          <div class="card"><div class="card-head"><h2>Was beim Start passiert</h2></div>
            <ol class="flow">
              <li><b>RLBotServer startet</b><span>lokal auf Port 23234</span></li>
              <li><b>Rocket League öffnet sich</b><span>${form.launcher === "steam" ? "Steam: RL wird direkt mit Bot-Parametern gestartet" : "Epic: RL öffnet kurz über den Launcher (Anmeldung), schließt und startet neu mit Bot-Parametern"}</span></li>
              <li><b>Offline-Match lädt</b><span>ohne Anti-Cheat, eigenes Spielfeld, kein Online-Dienst</span></li>
              <li><b>Deine KI fährt</b><span>bekommt 120× pro Sekunde den Spielzustand und antwortet mit Steuerbefehlen</span></li>
            </ol></div>
        </div>
        <div class="card form"><h2>Match einrichten</h2>
          <div class="field"><span class="field-label">Wer fährt?</span>${seg("brain", [["policy", "Eigene KI"], ["teacher", teacherReady ? "Lehrer (Nexto)" : "Lehrer (nicht geladen)"]], form.brain, "pick")}
            <small>${form.brain === "teacher" ? "Nexto spielt selbst – Grand-Champion-Niveau. Ideal, um zu sehen, wie gut ein Bot wirklich sein kann." : "Dein trainiertes Modell aus dem Training-Tab."}</small></div>
          ${form.brain === "teacher" ? "" : `<label class="field"><span>KI (Checkpoint)</span><select data-change="field" data-key="checkpoint">`}
            ${usable.length ? Object.entries(groupBy(usable, (c) => c.run)).map(([run, list]) => `<optgroup label="${h(run)}">${list.slice().reverse().map((c) => `<option value="${h(c.id)}" ${c.id === form.checkpoint ? "selected" : ""}>${h(run)} · ${c.id.endsWith("latest.pt") ? "neuester Stand" : `${fmt.steps(c.steps)} Schritte`}</option>`).join("")}</optgroup>`).join("") : `<option value="">– noch keine –</option>`}
          ${form.brain === "teacher" ? "" : "</select></label>"}
          <div class="field"><span class="field-label">Gegner</span>
            <div class="choice-grid">${Object.entries(info.modes).map(([key, label]) => `<button type="button" class="choice ${form.mode === key ? "on" : ""}" data-action="pick" data-key="mode" data-value="${key}"><b>${h(label)}</b><span>${MODE_HINT[key]}</span></button>`).join("")}</div></div>
          <div class="grid cols-2">
            <div class="field"><span class="field-label">Modus</span>${seg("team_size", [[1, "1v1"], [2, "2v2"], [3, "3v3"]], form.team_size, "pick")}</div>
            <div class="field"><span class="field-label">Launcher ${rl?.store ? `<span class="faint">(erkannt: ${h(rl.store)})</span>` : ""}</span>${seg("launcher", Object.entries(info.launchers), form.launcher, "pick")}</div>
          </div>
          ${form.mode === "psyonix" || form.mode === "human" ? `<div class="field"><span class="field-label">${form.mode === "human" ? "Stärke deiner Bot-Mitspieler" : "Stärke der Psyonix-Bots"}</span>${seg("skill", Object.entries(info.skills), form.skill, "pick")}</div>` : ""}
          ${form.mode === "bot" ? `<label class="field"><span>bot.toml des Community-Bots</span><input placeholder="C:\\Bots\\Nexto\\bot.toml" value="${h(form.opponent_bot)}" data-change="field" data-key="opponent_bot"><small>Bots für RLBot v5, z. B. aus dem RLBot-Botpack.</small></label>` : ""}
          ${info.message ? `<p class="${info.state === "failed" ? "error-text" : "faint"}">${h(info.message)}</p>` : ""}
          <div class="row end">
            ${busyState ? `<button class="btn" data-action="stop">${icon("stop")}Match beenden</button>` : ""}
            <button class="btn primary" data-action="start" ${blocked || busyState ? "disabled" : ""}>${icon("play")}Match starten</button>
          </div>
          ${blocked ? `<p class="faint small">${h(blocked)}</p>` : ""}
          ${rl && rl.supported && !rl.install ? `<p class="faint small">Rocket League wurde nicht gefunden – bei ungewöhnlichen Installationsorten kannst du trotzdem starten.</p>` : ""}
        </div>
      </div>`);
  };
  page.actions = {
    ...rlActions,
    field: (el) => { form[el.dataset.key] = el.value.trim(); },
    pick: (el) => { form[el.dataset.key] = el.dataset.key === "team_size" ? Number(el.dataset.value) : el.dataset.value; draw(); },
    start: (el) => busy(el, async () => {
      const result = await attempt(() => api("/api/play/start", { method: "POST", body: form }), "Match konnte nicht starten");
      if (result) { info = { ...info, ...result }; toast("Match startet", "Rocket League öffnet sich gleich."); draw(); }
    }),
    stop: (el) => busy(el, async () => { const result = await attempt(() => api("/api/play/stop", { method: "POST" })); if (result) { info = { ...info, ...result }; draw(); } }),
  };
  draw();
  refreshOnSnapshot(async () => { info = await api("/api/play"); draw(); });
  const timer = setInterval(async () => { if (!document.hidden) { await pollRocketLeague(); draw(); } }, 10000);
  page.cleanup.push(() => clearInterval(timer));
}

// ------------------------------------------------------------------ setup

async function pageSetup() {
  const [system, benchmark] = await Promise.all([api("/api/system"), api("/api/benchmark")]);
  const py = system.python || "python3"; // Linux/macOS: python3, Windows: python
  let measured = benchmark?.report || system.benchmark || null;
  const commands = [
    ["Installation (einmalig)", `${py} install.py`],
    ["App starten", `${py} start.py`],
    ["Lehrer laden", `${py} -m rocketai teacher`],
    ["Geschwindigkeit messen", `${py} -m rocketai benchmark`],
    ["Training ohne Oberfläche", `${py} -m rocketai train --preset student --name mein-bot`],
    ["Checkpoint bewerten", `${py} -m rocketai eval runs/mein-bot/checkpoints/latest.pt --opponent chaser teacher`],
    ["Installation prüfen", `${py} -m rocketai doctor`],
  ];
  // Ziele in Schritten und die Zeit, die sie bei diesem (gemessenen) Tempo brauchen.
  let benchmarkBusy = false;
  const milestones = [
    ["Ball treffen", 35e6],
    ["Gezielt schießen", 200e6],
    ["Schlägt Rookie-Bots", 650e6],
    ["Schlägt Pro-Bots", 1.5e9],
  ];
  const sps = measured?.best?.steps_per_second || 0;
  const trainingHours = 4; // Annahme für die 24/7-Spalte
  const benchmarkCard = () => {
    const best = measured?.best;
    const rows = (measured?.scale || []).map((s) =>
      `<tr><td>${s.workers} Prozesse × ${s.envs_per_worker} Spiel(e)</td><td class="num">${fmt.int(s.steps_per_second)}</td><td class="num">${fmt.int(s.decisions_per_second)}</td><td class="num">${fmt.int(s.realtime_factor)}×</td></tr>`).join("");
    return `<div class="card"><div class="card-head"><h2>Geschwindigkeit messen</h2>
        <span class="sub">${measured ? "auf diesem Rechner gemessen" : "noch nicht gemessen"}</span></div>
      <p class="muted small">Misst mit echten RocketSim-Spielen und einem echten PPO-Lernschritt, wie schnell dieser Rechner trainiert. Dauert ein paar Sekunden.</p>
      ${rows ? `<div class="table-scroll"><table class="table"><thead><tr><th>Einstellung</th><th class="num">Schritte/s</th><th class="num">Entscheidungen/s</th><th class="num">Echtzeit</th></tr></thead><tbody>${rows}</tbody></table></div>` : ""}
      ${best ? `<p class="small">Beste Einstellung: <b>${best.workers} Prozesse × ${best.envs_per_worker} Spiel(e)</b> → ${fmt.int(measured.steps_per_day)} Schritte pro Tag (24/7).
        ${measured.update_cuda && measured.update_cpu ? `Lernschritt auf der Grafikkarte ${(measured.update_cuda.steps_per_second / Math.max(1, measured.update_cpu.steps_per_second)).toFixed(1)}× schneller als auf der CPU.` : ""}</p>` : ""}
      ${(measured?.advice || []).map((x) => `<p class="faint small">${h(x)}</p>`).join("")}
      <div class="row end"><button class="btn" data-action="benchmark" ${benchmarkBusy ? "disabled" : ""}>${icon("play")}${benchmarkBusy ? "Messt …" : "Jetzt messen (ca. 20 s)"}</button></div>
    </div>`;
  };
  const draw = () => patch(view, `
    <div class="page-head"><div><div class="eyebrow">Einrichtung</div><h1>Einrichtung</h1>
      <p>Alles, was RocketAI braucht – und wie du es ohne Oberfläche bedienst.</p></div><span class="faint">Version ${h(system.version)}</span></div>
    <div class="grid cols-2">
      <div class="grid">
        <div class="card"><div class="card-head"><h2>Systemprüfung</h2></div><div class="checks">
          ${system.checks.map((c) => checkRow(c.ok ? "ok" : c.required ? "bad" : "off", c.label, c.detail, c.required ? "" : ' <span class="faint">(optional)</span>')).join("")}
        </div></div>
        ${rlCard(app.rl)}
        ${benchmarkCard()}
        <div class="card"><div class="card-head"><h2>Was die KI sieht</h2><span class="sub">${system.observation?.size || 184} Eingabewerte pro Auto</span></div>
          <p class="muted small">Kein Bild, sondern Messwerte aus dem Spiel: Ballposition, Geschwindigkeit, Drehung, Boost-Pads, alle Autos – gespiegelt auf das eigene Tor. Dazu diese Zusatzwerte:</p>
          <div class="feature-list">${(system.observation?.extras || []).map((f) => `<span class="feature">${h(f.label)}</span>`).join("")}</div></div>
      </div>
      <div class="grid">
        <div class="card"><div class="card-head"><h2>Befehle</h2></div><div class="grid gap-sm">
          ${commands.map(([label, cmd]) => `<div><div class="faint small" style="margin-bottom:4px">${h(label)}</div><div class="cmd"><code>${h(cmd)}</code><button class="btn ghost sm" data-action="copy" data-text="${h(cmd)}">${icon("copy")}Kopieren</button></div></div>`).join("")}
        </div></div>
        <div class="card"><div class="card-head"><h2>Ordner</h2></div><p class="muted">Trainings liegen in</p><div class="cmd" style="margin-top:8px"><code>${h(system.runs_folder)}</code><button class="btn ghost sm" data-action="copy" data-text="${h(system.runs_folder)}">${icon("copy")}</button></div></div>
        <div class="card"><div class="card-head"><h2>Tastenkürzel</h2></div>
          <div class="keys"><span><kbd>1</kbd>–<kbd>5</kbd> Kamera</span><span><kbd>V</kbd> 3D/2D</span><span><kbd>F</kbd> Vollbild</span><span><kbd>Leertaste</kbd> Replay Pause</span><span><kbd>←</kbd><kbd>→</kbd> ±5 s</span></div></div>
      </div>
    </div>
        <div class="section card"><div class="card-head"><h2>Wie lange dauert das Training?</h2><span class="sub">${sps ? `gerechnet mit deinem Tempo: ${fmt.int(sps)} Schritte/s` : "Beispielwerte – miss oben dein eigenes Tempo"}</span></div>
      <div class="table-scroll"><table class="table"><thead><tr><th>Ziel</th><th class="num">Schritte (Erfahrungswerte)</th><th class="num">${sps ? "Bei deinem Rechner" : "4 000/s (2 Kerne)"}</th><th class="num">${sps ? "24/7" : "15 000/s (8 Kerne)"}</th></tr></thead><tbody>
        ${milestones.map(([label, steps]) => `<tr><td>${h(label)}</td><td class="num">${fmt.steps(steps)}</td><td class="num">${sps ? fmt.duration(steps / sps) : "-"}</td><td class="num">${sps ? fmt.duration(steps / sps / (24 / (trainingHours || 4))) : "-"}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="faint small">Die Schrittzahlen sind Erfahrungswerte: Wie schnell die KI wirklich lernt, hängt von Belohnung, Einstellungen und Startbedingungen ab – nicht nur von der Rechenleistung.</p></div>`);`);
  page.actions = {
    ...rlActions,
    copy: (el) => copyText(el.dataset.text),
    benchmark: (el) => busy(el, async () => {
      benchmarkBusy = true; draw();
      try {
        const job = await attempt(() => api("/api/benchmark", { method: "POST", body: { seconds: 5, workers: 0, envs_per_worker: 1 } }), "Messung konnte nicht starten");
        if (job) {
          await waitForJob(job.id);
          measured = await api("/api/benchmark").then((r) => r.report);
          toast("Messung fertig", measured?.best ? `${fmt.int(measured.best.steps_per_second)} Schritte/s` : "");
        }
      } finally { benchmarkBusy = false; draw(); }
    }),
  };
  draw();
  refreshOnSnapshot(draw);
}

// ------------------------------------------------------------------ boot

window.addEventListener("hashchange", router);
router();
pollSnapshot();
pollRocketLeague();
setInterval(() => { if (!document.hidden) pollRocketLeague(); }, 30000);
