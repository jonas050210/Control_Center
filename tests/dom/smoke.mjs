// Headless smoke test for the control-center frontend.
//
// It loads web/index.html in jsdom, imports the real ES modules against a
// running server and clicks through every panel. Frames are polled with
// requestAnimationFrame, the DOM contract is asserted, and anything that
// throws inside the page fails the run.
//
//   Terminal 1:  python3 start.py --no-browser
//   Terminal 2:  cd tests/dom && npm install && node smoke.mjs
//
// Environment: BASE_URL (default http://127.0.0.1:8501), ROOT (repo root).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM, VirtualConsole } from 'jsdom';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = process.env.ROOT || path.resolve(HERE, '..', '..');
const BASE = process.env.BASE_URL || 'http://127.0.0.1:8501';
const PACE = Number(process.env.PACE || 700); // settle time after each click
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const problems = [];
const notes = [];

const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (error) => {
  const message = String(error.message || error);
  // jsdom has no canvas backend; chart/scene code is written to survive that.
  if (/getContext|WebGL|Not implemented/.test(message)) {
    notes.push(`jsdom: ${message.split('\n')[0]}`);
    return;
  }
  problems.push(`jsdomError: ${message}`);
});
virtualConsole.on('error', (message) => problems.push(`console.error: ${message}`));
virtualConsole.on('warn', (message) => notes.push(`console.warn: ${message}`));

const html = fs.readFileSync(path.join(ROOT, 'web', 'index.html'), 'utf8');
const dom = new JSDOM(html, { url: `${BASE}/`, pretendToBeVisual: true, virtualConsole });
const { window } = dom;

// The modules use browser globals directly, so expose them to Node ESM first.
global.window = window;
global.document = window.document;
Object.defineProperty(global, 'navigator', { value: window.navigator, configurable: true });
global.location = window.location;
global.history = window.history;
global.Node = window.Node;
global.HTMLElement = window.HTMLElement;
global.Event = window.Event;
global.CustomEvent = window.CustomEvent;
global.getComputedStyle = window.getComputedStyle.bind(window);
global.requestAnimationFrame = (callback) => setTimeout(() => callback(Date.now()), 16);
global.cancelAnimationFrame = (handle) => clearTimeout(handle);

const nodeFetch = globalThis.fetch.bind(globalThis);
global.fetch = (input, init) => nodeFetch(new URL(String(input), BASE).toString(), init);
window.fetch = global.fetch;

window.addEventListener('error', (event) => problems.push(`window.error: ${event.message}`));
process.on('unhandledRejection', (reason) => {
  problems.push(`unhandledRejection: ${reason && reason.message ? reason.message : reason}`);
});

let failures = 0;
function check(label, condition, detail = '') {
  const ok = Boolean(condition);
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`);
  return ok;
}

function buttons(root, text) {
  return [...root.querySelectorAll('button')].filter((node) => node.textContent.includes(text));
}

function click(node) {
  node.dispatchEvent(new window.Event('click', { bubbles: true }));
}

async function api(pathname) {
  const response = await nodeFetch(`${BASE}${pathname}`);
  return response.json();
}

// ------------------------------------------------------------------ boot
await import(path.join(ROOT, 'web', 'js', 'app.js'));
await wait(1500);

const navButtons = [...document.querySelectorAll('#nav button')];
check('acht Panels in der Navigation', navButtons.length === 8,
  navButtons.map((item) => item.textContent.trim()).join(' | '));
check('Serverstatus ONLINE',
  document.getElementById('connection-value').textContent.includes('ONLINE'),
  document.getElementById('connection-value').textContent);

async function open(label) {
  const button = navButtons.find((item) => item.textContent.includes(label));
  check(`Navigation: ${label}`, Boolean(button));
  if (button) click(button);
  await wait(PACE);
}

async function panelOf(label) {
  return document.getElementById(`panel-${label}`);
}

// ----------------------------------------------------------------- arena
// Reset first so the run starts from a defined state (the panel polls the server).
await nodeFetch(`${BASE}/api/arena/reset`, { method: 'POST' }).catch(() => {});
const arena = await panelOf('arena');
await wait(PACE);
check('ARENA: Viewport vorhanden', Boolean(arena.querySelector('.viewport')));
if (arena.querySelector('.err-box')) {
  notes.push(`ARENA ohne WebGL: ${arena.querySelector('.err-box').textContent.trim().slice(0, 90)}`);
}
check('ARENA: Telemetrie gerendert',
  arena.textContent.includes('AGENT 1') && arena.textContent.includes('HP'));
const startButton = buttons(arena, 'Start Match')[0] || buttons(arena, 'Start match')[0];
check('ARENA: Start-Button vorhanden', Boolean(startButton));
if (startButton) {
  click(startButton);
  await wait(PACE);
  const state = await api('/api/arena/state');
  check('ARENA: Match läuft', state.running === true, `frame=${state.frame && state.frame.frame}`);
  const stepButton = buttons(arena, 'Einzelschritt')[0] || buttons(arena, 'Step')[0];
  check('ARENA: Einzelschritt-Button vorhanden', Boolean(stepButton));
  const pause = buttons(arena, 'Pause Match')[0];
  if (pause) click(pause);
  await wait(PACE);
}

// ------------------------------------------------------------ playground
await open('PLAYGROUND');
const playground = document.getElementById('panel-playground');
check('PLAYGROUND: Modus-Umschalter vorhanden', playground.querySelectorAll('select').length >= 1);
const fire = buttons(playground, 'FIRE')[0];
check('PLAYGROUND: Feuer-Button vorhanden', Boolean(fire));
if (fire) {
  click(fire);
  await wait(PACE);
  check('PLAYGROUND: Schuss verarbeitet', playground.textContent.includes('YOUR HEALTH'));
}
const modeSelect = [...playground.querySelectorAll('select')][0];
if (modeSelect) {
  modeSelect.value = 'aim';
  modeSelect.dispatchEvent(new window.Event('change', { bubbles: true }));
  await wait(PACE * 2);
  check('PLAYGROUND: Aim-Modus rendert', playground.textContent.includes('AIM'));
}
const dodgeMode = [...playground.querySelectorAll('select')][0];
if (dodgeMode) {
  dodgeMode.value = 'dodge';
  dodgeMode.dispatchEvent(new window.Event('change', { bubbles: true }));
  await wait(PACE * 2);
  check('PLAYGROUND: Dodge-Modus rendert', playground.textContent.includes('DODGE'));
}

// --------------------------------------------------------------- training
await open('TRAINING');
const training = document.getElementById('panel-training');
check('TRAINING: Start-Button vorhanden', buttons(training, 'Start Training').length >= 1);
check('TRAINING: Metriken und Log rendern', training.textContent.includes('STEPS'));
check('TRAINING: Statusanzeige rendert',
  /STOPPED|COMPLETE|FAILED|RUNNING|PAUSED/.test(training.textContent),
  training.textContent.slice(0, 80).replace(/\s+/g, ' '));

// ------------------------------------------------------------------ stats
await open('STATS');
const stats = document.getElementById('panel-stats');
check('STATS: Kennzahlen rendern',
  stats.textContent.includes('Ø TTK (KILLS)') && stats.textContent.includes('WIN RATE'));
const statsPayload = await api('/api/stats');
check('STATS: API liefert kill_count', Number.isInteger(statsPayload.summary.kill_count),
  `kill_count=${statsPayload.summary.kill_count}`);
check('STATS: Diagramme gezeichnet', stats.querySelectorAll('canvas.chart').length >= 5,
  `${stats.querySelectorAll('canvas.chart').length} Canvas-Charts`);

// -------------------------------------------------------------- benchmark
await open('BENCHMARK');
const benchmark = document.getElementById('panel-benchmark');
check('BENCHMARK: Ansicht rendert',
  benchmark.textContent.includes('BENCHMARK') || benchmark.textContent.includes('Start'));

// -------------------------------------------------------------------- ttk
await open('TTK-TESTER');
const ttk = document.getElementById('panel-ttk');
const sliders = ttk.querySelectorAll('input[type="range"]');
check('TTK: Slider gerendert', sliders.length >= 16, `${sliders.length} Slider`);
check('TTK: Duel-Button vorhanden', buttons(ttk, 'Duels simulieren').length === 1);

// ------------------------------------------------------------------- maps
await open('MAPS');
const maps = document.getElementById('panel-maps');
check('MAPS: Statistik rendert',
  maps.textContent.includes('OBJEKTE') || maps.textContent.includes('DECKUNG'));
check('MAPS: Kartenansicht vorhanden', Boolean(maps.querySelector('.viewport')));

// ---------------------------------------------------------------- heatmap
await open('HEATMAP');
const heatmap = document.getElementById('panel-heatmap');
check('HEATMAP: Kennzahlen rendern',
  heatmap.textContent.includes('KILLS') || heatmap.textContent.includes('Kills'));

// ---------------------------------------------------------------- summary
console.log('\n--- Notizen (erwartete jsdom-Limitierungen) ---');
console.log(notes.length ? [...new Set(notes)].slice(0, 8).join('\n') : 'keine');
console.log('\n--- gefundene Probleme ---');
if (problems.length === 0) {
  console.log('keine');
} else {
  [...new Set(problems)].forEach((problem) => console.log(`- ${problem}`));
}
console.log(failures === 0 && problems.length === 0 ? '\nERGEBNIS: OK' : `\nERGEBNIS: ${failures} Fehler`);
process.exit(failures === 0 && problems.length === 0 ? 0 : 1);
