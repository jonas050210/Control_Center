// ARENA panel: two policies duel in a WebGL-rendered 3D map.

import { api } from '../api.js';
import { store } from '../store.js';
import { createScene } from '../scene.js';
import {
  button, card, feed, field, fmt, h, metric, mount, setText, toast, warnBox,
} from '../dom.js';

function bandLabel(value, labels) {
  if (!value) return 'unbekannt';
  const index = Math.min(labels.length - 1, Math.max(0, Math.round((value + 1) / 2 * labels.length - 0.5)));
  return labels[index];
}

export const arenaPanel = {
  id: 'arena',
  label: '🎮 ARENA',

  mount(root) {
    const meta = store.meta;
    this.state = {
      map: 'Dust',
      weaponA: meta.weapons[0],
      weaponB: meta.weapons[2] || meta.weapons[1],
      modelA: meta.models[0],
      modelB: meta.models[0],
      detail: 'Balanced',
      vision: meta.default_vision_mode || 'coarse_los',
      configured: false,
      sceneKey: null,
      errors: [],
      running: false,
    };

    this.mapSelect = h('select');
    this.weaponA = h('select');
    this.weaponB = h('select');
    this.modelA = h('select');
    this.modelB = h('select');
    this.detailSelect = h('select');
    this.visionSelect = h('select');
    for (const [node, values, key] of [
      [this.mapSelect, meta.maps, 'map'],
      [this.weaponA, meta.weapons, 'weaponA'],
      [this.weaponB, meta.weapons, 'weaponB'],
      [this.modelA, meta.models, 'modelA'],
      [this.modelB, meta.models, 'modelB'],
      [this.detailSelect, Object.keys(meta.detail_presets), 'detail'],
      [this.visionSelect, meta.vision_modes || ['coarse_los'], 'vision'],
    ]) {
      for (const value of values) node.appendChild(h('option', { value, text: value }));
      node.value = this.state[key];
      node.addEventListener('change', () => {
        this.state[key] = node.value;
        this.applyConfig();
      });
    }

    this.startButton = button('▶ Start Match', () => this.toggleRun(), { className: 'grow' });
    this.stepButton = button('⏭ Step', () => this.step(1));
    this.resetButton = button('🔄 Reset', () => this.reset());
    this.viewportHost = h('div', { class: 'grow', style: { minWidth: '320px' } });
    this.telemetry = h('div', { class: 'grid cols-2' });
    this.statsRow = h('div', { class: 'grid cols-4' });
    this.feedNode = h('div', { class: 'feed' });
    this.errorHost = h('div');
    this.scoreHost = h('div', { class: 'legend' });
    this.perceptionHost = h('div');

    const settings = card('MATCH SETUP',
      h('div', { class: 'row' },
        field('ARENA MAP', this.mapSelect),
        field('AGENT 1 WEAPON', this.weaponA),
        field('AGENT 2 WEAPON', this.weaponB)),
      h('div', { class: 'row', style: { marginTop: '10px' } },
        field('AGENT 1 POLICY', this.modelA),
        field('AGENT 2 POLICY', this.modelB),
        field('3D DETAIL', this.detailSelect),
        field('SICHT DER AI', this.visionSelect)),
      h('div', { class: 'row', style: { marginTop: '10px' } },
        this.startButton, this.stepButton, this.resetButton));
    this.perceptionCard = card('👁 WAS DIE AI SIEHT (AGENT 1)', this.perceptionHost);

    mount(root,
      h('h2', { text: '🎮 3D ARENA · LIVE MATCH CONTROL' }),
      h('p', { class: 'hint', text: 'Die Simulation läuft headless in Python; die 3D-Szene wird direkt im Browser mit WebGL gerendert (kein Plotly, kein Streamlit). Maus: drehen · Rad: zoomen · Shift+Ziehen: verschieben.' }),
      settings,
      this.perceptionCard,
      this.errorHost,
      h('hr', { class: 'sep' }),
      h('div', { class: 'row', style: { alignItems: 'flex-start' } },
        h('div', { class: 'grow', style: { flexBasis: '620px' } }, this.viewportHost),
        h('div', { class: 'grow', style: { flexBasis: '330px' } },
          card('MATCH TELEMETRY', this.telemetry, this.scoreHost),
          h('div', { style: { height: '12px' } }),
          card('KILL FEED', this.feedNode))));

    this.scene = createScene(this.viewportHost, { height: 560 });
    this.applyConfig();
    this.loopTimer = null;
    this.active = true;
    this.runLoop();
  },

  onShow() {
    this.active = true;
    if (!this.loopTimer) this.runLoop();
  },

  onHide() {
    this.active = false;
    if (this.loopTimer) {
      clearTimeout(this.loopTimer);
      this.loopTimer = null;
    }
  },

  async runLoop() {
    if (!this.active) return;
    if (this.state.running) {
      await this.step(3, true);
    }
    if (!this.active) return;
    this.loopTimer = setTimeout(() => this.runLoop(), this.state.running ? 110 : 400);
  },

  async applyConfig() {
    const payload = {
      map: this.state.map,
      weapon_a: this.state.weaponA,
      weapon_b: this.state.weaponB,
      model_a: this.state.modelA,
      model_b: this.state.modelB,
      detail: this.state.detail,
      vision: this.state.vision,
    };
    try {
      const response = await api.arenaConfig(payload);
      this.state.configured = true;
      this.state.sceneKey = response.scene_key;
      this.state.running = Boolean(response.running);
      store.arena.sceneKey = response.scene_key;
      store.arena.detail = this.state.detail;
      store.arena.models = { a: response.models.a, b: response.models.b };
      this.scene.setStatic(response.static);
      this.applyFrame(response.frame, response.perception);
      this.renderErrors(response.load_errors);
      this.renderMessages(response.messages);
      this.updateRunButton();
    } catch (error) {
      this.renderErrors([error.message]);
    }
  },

  async step(count = 1, quiet = false) {
    if (!this.state.configured) return;
    try {
      const response = await api.arenaStep(count);
      this.state.running = Boolean(response.running) && (!quiet || response.running);
      this.applyFrame(response.frame, response.perception);
      this.renderMessages(response.messages);
      if (response.new_messages?.length) {
        this.trackResults(response.new_messages);
      }
      this.renderErrors(response.load_errors);
      this.updateRunButton();
    } catch (error) {
      if (!quiet) toast(error.message, 'error');
    }
  },

  async toggleRun() {
    if (!this.state.configured) return;
    const next = !this.state.running;
    try {
      const response = await api.arenaRun(next);
      this.state.running = Boolean(response.running) && next;
      this.applyFrame(response.frame);
      this.updateRunButton();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async reset() {
    try {
      const response = await api.arenaReset();
      this.state.running = false;
      this.applyFrame(response.frame);
      this.renderMessages([]);
      this.updateRunButton();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  updateRunButton() {
    setText(this.startButton, this.state.running ? '⏸ Pause Match' : '▶ Start Match');
    this.startButton.classList.toggle('active', this.state.running);
  },

  renderPerception(perception) {
    if (!perception || !perception.player) return;
    const p = perception.player;
    // Sector centres sit at multiples of 30°, so the sector index is simply the
    // quantised bearing divided by that step - no extra rounding fudge.
    const sectorStep = Math.PI / 6;
    const sector = ((Math.round(bearing / sectorStep) % 12) + 12) % 12;
    const direction = p.memory
      ? `Sektor ${sector + 1}/12 (${fmt.fixed((bearing * 180) / Math.PI, 0)}°)`
      : 'keine Richtung bekannt';
    const since = p.seconds_since_seen === null || p.seconds_since_seen === undefined
      ? 'nie gesehen'
      : `vor ${fmt.fixed(p.seconds_since_seen, 1)} s`;
    const bands = ['<5 m', '5–10 m', '10–20 m', '20–35 m', '35–60 m', '>60 m'];
    const health = ['<25 %', '25–50 %', '50–75 %', '>75 %'];
    mount(this.perceptionHost,
      h('div', { class: 'grid cols-4' },
        metric('SICHTKONTAKT', p.visible ? 'JA' : 'NEIN',
          p.visible ? 'Ziel im 120°-Kegel, Sichtlinie frei' : 'verdeckt oder außerhalb des Kegels'),
        metric('RICHTUNG (GROB)', direction,
          p.visible ? 'frisch gesehen' : since),
        metric('ENTFERNUNG (BAND)', bandLabel(p.distance_band, bands),
          `Sichtfeld ${perception.field_of_view_degrees || 120}°`),
        metric('GEGNER-HP (GROB)', bandLabel(p.hp_band, health),
          `Modus: ${perception.vision_mode}`)),
      h('p', { class: 'hint', text: 'Die AI bekommt keine Koordinaten: nur 12 Richtungs-Sektoren, '
        + 'Entfernungs- und HP-Bänder sowie ein Gedächtnis („zuletzt gesehen vor X s"). '
        + 'Deckung blendet sie wirklich aus.' }));
  },

  applyFrame(frame, perception) {
    if (!frame) return;
    this.scene.setFrame(frame);
    if (perception) this.renderPerception(perception);
    const [first, second] = frame.agents || [];
    mount(this.telemetry,
      this.agentCard(first, 'AGENT 1', 'green'),
      this.agentCard(second, 'AGENT 2', 'red'));
    mount(this.statsRow);
    const wins = store.arena.stats;
    mount(this.scoreHost,
      h('span', { class: 'chip', text: `MATCHES ${wins.matches}` }),
      h('span', { class: 'chip', text: `A1 ${wins.winsA}` }),
      h('span', { class: 'chip', text: `A2 ${wins.winsB}` }),
      h('span', { class: 'chip', text: `DRAW ${wins.draws}` }));
    const totals = `MATCH TIME ${fmt.fixed(frame.elapsed, 1)}s · SIM FRAME ${fmt.num(frame.frame)}`
      + (frame.done ? ' · MATCH BEENDET' : '');
    this.scoreHost.appendChild(h('span', { class: 'chip', text: totals }));
  },

  agentCard(agent, label, tone = 'green') {
    if (!agent) return metric(label, '—');
    const accuracy = agent.hits / Math.max(1, agent.bullets_fired || agent.shots_fired || 1);
    return h('div', { class: 'metric' },
      h('div', { class: 'agent-row' },
        h('span', { class: 'm-label', text: `${label} · ${agent.weapon}` }),
        h('span', { class: 'm-value', text: `${agent.hp.toFixed(0)} HP` })),
      h('div', { class: `hp-bar ${tone}` },
        h('span', { style: { width: `${Math.max(0, Math.min(100, agent.hp))}%` } })),
      h('span', { class: 'm-sub', text: `AMMO ${agent.ammo}/${agent.mag_size} · SHOTS ${agent.shots_fired} · HITS ${agent.hits} · ACC ${fmt.percent(accuracy)}` }));
  },

  renderErrors(errors) {
    const unique = [...new Set([...(errors || []), ...this.state.errors])].filter(Boolean).slice(0, 4);
    this.state.errors = unique;
    mount(this.errorHost, ...unique.map((message) => warnBox(message)));
  },

  renderMessages(messages) {
    feed(this.feedNode, messages || [], { limit: 14 });
  },

  trackResults(messages) {
    for (const message of messages) {
      if (message.includes('MATCH COMPLETE')) {
        store.arena.stats.matches += 1;
        if (message.includes('AGENT 1 WINS')) store.arena.stats.winsA += 1;
        else if (message.includes('AGENT 2 WINS')) store.arena.stats.winsB += 1;
        else store.arena.stats.draws += 1;
      }
    }
  },

  unmount() {
    this.onHide();
    this.scene?.dispose();
  },
};
