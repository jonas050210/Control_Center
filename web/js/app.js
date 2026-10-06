// Application shell: navigation, lazy panel mounting, and status polling.

import { api } from './api.js';
import { store } from './store.js';
import { arenaPanel } from './panels/arena.js';
import { playgroundPanel } from './panels/playground.js';
import { trainingPanel } from './panels/training.js';
import { statsPanel } from './panels/stats.js';
import { benchmarkPanel } from './panels/benchmark.js';
import { ttkPanel } from './panels/ttk.js';
import { mapsPanel } from './panels/maps.js';
import { heatmapPanel } from './panels/heatmap.js';
import { mount, setText, statusClass, toast, errorBox, h } from './dom.js';

const PANELS = [arenaPanel, playgroundPanel, trainingPanel, statsPanel,
  benchmarkPanel, ttkPanel, mapsPanel, heatmapPanel];

const navHost = document.getElementById('nav');
const host = document.getElementById('panel-host');
const connectionValue = document.getElementById('connection-value');
const trainingBadge = document.getElementById('training-badge-value');

let currentPanel = null;

function activate(panelId) {
  const panel = PANELS.find((entry) => entry.id === panelId) || PANELS[0];
  if (currentPanel && currentPanel !== panel) currentPanel.onHide?.();
  for (const entry of PANELS) {
    const section = document.getElementById(`panel-${entry.id}`);
    if (section) section.classList.toggle('active', entry === panel);
  }
  for (const node of navHost.querySelectorAll('button')) {
    node.classList.toggle('active', node.dataset.panel === panel.id);
  }
  if (!panel.mounted) {
    const section = document.getElementById(`panel-${panel.id}`);
    try {
      panel.mount(section);
      panel.mounted = true;
    } catch (error) {
      console.error(error);
      mount(section, errorBox(`Panel ${panel.id} konnte nicht geladen werden: ${error.message}`));
    }
  }
  panel.onShow?.();
  currentPanel = panel;
  if (window.location.hash !== `#${panel.id}`) {
    history.replaceState(null, '', `#${panel.id}`);
  }
}

function buildNav() {
  mount(navHost, ...PANELS.map((panel) => {
    const node = h('button', { text: panel.label, dataset: { panel: panel.id } });
    node.addEventListener('click', () => activate(panel.id));
    return node;
  }));
}

function buildSections() {
  mount(host, ...PANELS.map((panel) => h('section', { id: `panel-${panel.id}`, class: 'panel' })));
}

async function boot() {
  buildNav();
  buildSections();
  try {
    const [health, meta] = await Promise.all([api.health(), api.meta()]);
    store.meta = meta;
    setText(connectionValue, '● ONLINE');
    connectionValue.className = 'status-good';
    const deps = meta.dependencies || {};
    if (deps.streamlit || deps.plotly) {
      toast('Streamlit/Plotly/pandas sind noch installiert und werden nicht mehr gebraucht: '
        + 'python3 install.py --remove-legacy entfernt sie.', 'warn', 9000);
    }
    if (!meta.training_ready) {
      toast('PPO-Training ist deaktiviert: Stable-Baselines3/PyTorch fehlen. install.py ausführen.', 'warn', 9000);
    }
    if (health.app) {
      document.title = `NEURAL ARENA · ${meta.cpu.physical} Kerne · ${health.version}`;
    }
  } catch (error) {
    setText(connectionValue, '● OFFLINE');
    connectionValue.className = 'status-bad';
    mount(host, errorBox(`Server nicht erreichbar: ${error.message}. start.py ausführen und diese Seite neu laden.`));
    return;
  }
  activate((window.location.hash || '').replace('#', '') || 'arena');
  window.addEventListener('hashchange', () => {
    const id = (window.location.hash || '').replace('#', '');
    if (id && id !== currentPanel?.id) activate(id);
  });
  setInterval(async () => {
    try {
      const snapshot = await api.trainingStatus();
      const status = snapshot.status || 'stopped';
      setText(trainingBadge, `● ${status.toUpperCase()}`);
      trainingBadge.className = statusClass(status);
      setText(connectionValue, '● ONLINE');
      connectionValue.className = 'status-good';
    } catch (error) {
      setText(connectionValue, '● VERBINDUNG VERLOREN');
      connectionValue.className = 'status-bad';
    }
  }, 3000);
}

boot();

// Surface unexpected module errors instead of failing silently.
window.addEventListener('unhandledrejection', (event) => {
  console.error(event.reason);
  if (event.reason && event.reason.message) toast(event.reason.message, 'error');
});
