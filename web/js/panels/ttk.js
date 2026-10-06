// TTK panel: Monte-Carlo weapon balance lab with Lua export.

import { api } from '../api.js';
import { chartCanvas, drawDonut, drawHistogram } from '../charts.js';
import { button, card, field, fmt, h, metric, mount, table, toast } from '../dom.js';

export const ttkPanel = {
  id: 'ttk',
  label: '🎯 TTK-TESTER',

  mount(root) {
    this.defaults = null;
    this.result = null;
    this.fields = { a: {}, b: {} };
    this.weapons = { a: 'Pistol', b: 'AK-47' };
    this.trials = 5000;
    this.distance = 20;

    this.histogramA = chartCanvas((ctx, w, hgt) => drawHistogram(ctx, w, hgt, {
      counts: this.result?.weapons.a.histogram || [],
      edges: this.result?.histogram_edges || [],
      color: '#00ff41',
      xLabel: 'TTK Waffe 1',
      emptyMessage: 'Noch keine Simulation.',
    }), { height: 280 });
    this.histogramB = chartCanvas((ctx, w, hgt) => drawHistogram(ctx, w, hgt, {
      counts: this.result?.weapons.b.histogram || [],
      edges: this.result?.histogram_edges || [],
      color: '#ff0040',
      xLabel: 'TTK Waffe 2',
      emptyMessage: 'Noch keine Simulation.',
    }), { height: 280 });
    this.winChart = chartCanvas((ctx, w, hgt) => drawDonut(ctx, w, hgt, {
      parts: this.result ? [
        { label: this.result.names.a, value: this.result.win_rate_a, color: '#00ff41' },
        { label: this.result.names.b, value: this.result.win_rate_b, color: '#ff0040' },
        { label: 'Draw / Timeout', value: this.result.draw_rate, color: '#616a64' },
      ] : [],
      centerLabel: 'DUELS',
      emptyMessage: 'Noch keine Simulation.',
    }), { height: 280 });

    this.editorA = h('div');
    this.editorB = h('div');
    this.resultHost = h('div');
    this.tableHost = h('div');
    this.luaHost = h('pre', { class: 'code' });
    this.verdictHost = h('div');

    this.trialsRange = rangeInput('SIMULATION COUNT', 1000, 100000, 1000, this.trials,
      (value) => { this.trials = value; });
    this.distanceRange = rangeInput('DUEL DISTANZ (m)', 5, 80, 1, this.distance,
      (value) => { this.distance = value; });

    mount(root,
      h('h2', { text: '🎯 TTK TESTER · WAFFEN-BALANCING LABOR' }),
      h('p', { class: 'hint', text: 'Monte-Carlo-Duelle mit Streuung, Schrot-Pellets, Headshots, Distanz-Falloff, Magazinen und Nachladezeiten. Die Simulation läuft im Server-Thread (NumPy-vektorisiert).' }),
      h('div', { class: 'grid cols-2' },
        card('WAFFE 1', this.editorA),
        card('WAFFE 2', this.editorB)),
      h('div', { class: 'row', style: { marginTop: '12px' } },
        h('div', { class: 'grow' }, this.trialsRange.node),
        h('div', { class: 'grow' }, this.distanceRange.node),
        button('⚔️ Duels simulieren', () => this.simulate())),
      h('hr', { class: 'sep' }),
      this.verdictHost,
      this.resultHost,
      h('div', { class: 'grid cols-2', style: { marginTop: '12px' } },
        card('TTK VERTEILUNG · WAFFE 1', this.histogramA.canvas),
        card('TTK VERTEILUNG · WAFFE 2', this.histogramB.canvas)),
      h('div', { class: 'grid cols-2', style: { marginTop: '12px' } },
        card('DUEL WIN-RATE', this.winChart.canvas),
        card('WAFFEN-VERGLEICH', this.tableHost)),
      h('h4', { text: 'ROBLOX STUDIO EXPORT' }),
      h('div', { class: 'row' },
        button('📋 Lua kopieren', () => this.copyLua(), { className: 'ghost' })),
      this.luaHost);

    this.loadDefaults();
  },

  async loadDefaults() {
    try {
      this.defaults = await api.ttkDefaults();
      if (this.defaults.state?.a?.weapon) this.weapons.a = this.defaults.state.a.weapon;
      if (this.defaults.state?.b?.weapon) this.weapons.b = this.defaults.state.b.weapon;
      if (this.defaults.state?.trials) this.trials = this.defaults.state.trials;
      if (this.defaults.state?.distance) this.distance = this.defaults.state.distance;
      this.trialsRange.set(this.trials);
      this.distanceRange.set(this.distance);
      this.fields.a = { ...this.defaults.specs[this.weapons.a], ...(this.defaults.state.a.fields || {}) };
      this.fields.b = { ...this.defaults.specs[this.weapons.b], ...(this.defaults.state.b.fields || {}) };
      this.renderEditors();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  renderEditors() {
    this.renderEditor('a', this.editorA);
    this.renderEditor('b', this.editorB);
  },

  renderEditor(side, host) {
    const weapons = this.defaults.weapons;
    const presetSelect = h('select');
    for (const name of weapons) presetSelect.appendChild(h('option', { value: name, text: name }));
    presetSelect.value = this.weapons[side];
    presetSelect.addEventListener('change', () => {
      this.weapons[side] = presetSelect.value;
      this.fields[side] = { ...this.defaults.specs[presetSelect.value] };
      this.renderEditors();
    });

    const sliders = this.defaults.fields.map((name) => {
      const [min, max, step] = this.defaults.limits[name];
      const value = Number(this.fields[side][name]);
      const readout = h('span', { class: 'hint', text: formatField(name, value) });
      const input = h('input', { type: 'range', min, max, step, value });
      input.addEventListener('input', () => {
        this.fields[side][name] = Number(input.value);
        readout.textContent = formatField(name, Number(input.value));
      });
      return h('div', null,
        h('label', { class: 'field' }, `${name.toUpperCase()} `, readout), input);
    });

    mount(host,
      field('PRESET', presetSelect),
      ...sliders);
  },

  async simulate() {
    try {
      const response = await api.ttkSimulate({
        weapon_a: this.weapons.a,
        weapon_b: this.weapons.b,
        fields_a: this.fields.a,
        fields_b: this.fields.b,
        trials: this.trials,
        distance: this.distance,
      });
      this.result = response;
      this.render();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  render() {
    if (!this.result) return;
    const result = this.result;
    mount(this.verdictHost, h('div', { class: 'ok-box', text: `VERDICT · ${result.verdict}` }));
    mount(this.resultHost,
      h('div', { class: 'grid cols-4', style: { marginTop: '12px' } },
        metric(`${result.names.a} WIN RATE`, fmt.percent(result.win_rate_a)),
        metric(`${result.names.b} WIN RATE`, fmt.percent(result.win_rate_b)),
        metric('Ø TTK WAFFE 1', result.weapons.a.mean_ttk === null ? 'Kein Kill' : `${fmt.fixed(result.weapons.a.mean_ttk, 2)}s`),
        metric('Ø TTK WAFFE 2', result.weapons.b.mean_ttk === null ? 'Kein Kill' : `${fmt.fixed(result.weapons.b.mean_ttk, 2)}s`)),
      h('p', { class: 'hint', text: `${fmt.num(result.trials)} Duelle auf ${result.distance}m Distanz · Draw/Timeout ${fmt.percent(result.draw_rate)}` }));

    mount(this.tableHost, table([
      { label: 'KENNZAHL', value: (row) => row.label },
      { label: result.names.a, value: (row) => row.a, align: 'right' },
      { label: result.names.b, value: (row) => row.b, align: 'right' },
    ], [
      { label: 'Kill-Rate', a: fmt.percent(result.weapons.a.kill_rate), b: fmt.percent(result.weapons.b.kill_rate) },
      { label: 'Ø Schüsse bis Kill', a: fmt.fixed(result.weapons.a.mean_shots, 1), b: fmt.fixed(result.weapons.b.mean_shots, 1) },
      { label: 'Treffer-Wahrscheinlichkeit', a: fmt.percent(result.weapons.a.accuracy), b: fmt.percent(result.weapons.b.accuracy) },
      { label: 'Erwarteter DPS', a: fmt.fixed(result.weapons.a.expected_dps, 1), b: fmt.fixed(result.weapons.b.expected_dps, 1) },
      { label: 'Basisschaden', a: String(result.weapons.a.spec.damage), b: String(result.weapons.b.spec.damage) },
      { label: 'Magazin', a: String(result.weapons.a.spec.mag_size), b: String(result.weapons.b.spec.mag_size) },
      { label: 'Nachladen', a: `${result.weapons.a.spec.reload_time}s`, b: `${result.weapons.b.spec.reload_time}s` },
      { label: 'Streuung', a: `${result.weapons.a.spec.spread}°`, b: `${result.weapons.b.spec.spread}°` },
    ]));

    this.histogramA.render();
    this.histogramB.render();
    this.winChart.render();
    this.luaHost.textContent = result.lua;
  },

  copyLua() {
    if (!this.result) return;
    navigator.clipboard.writeText(this.result.lua)
      .then(() => toast('Lua-Tabelle in die Zwischenablage kopiert.'))
      .catch(() => toast('Kopieren nicht möglich – Text manuell markieren.', 'warn'));
  },
};

function rangeInput(labelText, min, max, step, value, onChange) {
  const readout = h('span', { class: 'hint', text: String(value) });
  const input = h('input', { type: 'range', min, max, step, value });
  input.addEventListener('input', () => {
    onChange(Number(input.value));
    readout.textContent = String(input.value);
  });
  const label = h('label', { class: 'field' }, `${labelText} `, readout);
  return {
    node: h('div', null, label, input),
    set(next) {
      input.value = String(next);
      readout.textContent = String(next);
      onChange(Number(next));
    },
  };
}

function formatField(name, value) {
  if (name === 'mag_size' || name === 'pellets' || name === 'damage') return String(Math.round(value));
  return Number(value).toFixed(2);
}
