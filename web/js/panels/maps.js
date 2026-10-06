// MAP LAB panel: 3D geometry preview, stats, randomization, and Custom editor.

import { api } from '../api.js';
import { store } from '../store.js';
import { createScene } from '../scene.js';
import { button, card, field, fmt, h, metric, mount, table, toast, warnBox } from '../dom.js';

const OBJECT_TYPES = ['wall', 'crate', 'barrel', 'ramp', 'pillar', 'shelf', 'platform'];

export const mapsPanel = {
  id: 'maps',
  label: '🗺️ MAPS',

  mount(root) {
    this.meta = store.meta;
    this.mapName = this.meta.maps[0];
    this.detail = 'Balanced';
    this.payload = null;

    this.mapSelect = h('select');
    for (const name of this.meta.maps) this.mapSelect.appendChild(h('option', { value: name, text: name }));
    this.mapSelect.addEventListener('change', () => {
      this.mapName = this.mapSelect.value;
      this.load();
      this.renderEditor();
    });
    this.detailSelect = h('select');
    for (const name of Object.keys(this.meta.detail_presets)) {
      this.detailSelect.appendChild(h('option', { value: name, text: name }));
    }
    this.detailSelect.value = this.detail;
    this.detailSelect.addEventListener('change', () => {
      this.detail = this.detailSelect.value;
      this.load();
    });

    this.statsRow = h('div', { class: 'grid cols-4' });
    this.noticeHost = h('div');
    this.breakdownHost = h('div');
    this.editorHost = h('div');
    this.sceneHost = h('div', { class: 'grow', style: { flexBasis: '660px', minWidth: '320px' } });

    mount(root,
      h('h2', { text: '🗺️ MAP LAB · 3D GEOMETRIE-VORSCHAU' }),
      h('p', { class: 'hint', text: 'Wähle eine Map, randomisiere Deckung oder baue eine eigene Custom-Arena. Das Custom-Layout liegt in data/custom_map.json und wird von Arena und Training mitbenutzt.' }),
      card('MAP AUSWAHL',
        h('div', { class: 'row' },
          field('SELECT MAP', this.mapSelect),
          field('3D DETAIL', this.detailSelect),
          button('🎲 Randomize Cover', () => this.randomize()),
          button('↺ Reset Layout', () => this.reset(), { className: 'ghost' }))),
      this.noticeHost,
      this.statsRow,
      h('hr', { class: 'sep' }),
      h('div', { class: 'row', style: { alignItems: 'flex-start' } },
        this.sceneHost,
        h('div', { class: 'grow', style: { flexBasis: '340px' } },
          card('OBJEKT-TYPEN', this.breakdownHost))),
      this.editorHost);

    this.scene = createScene(this.sceneHost, { height: 620 });
    this.load();
    this.renderEditor();
  },

  unmount() {
    this.scene?.dispose();
  },

  async load() {
    try {
      const payload = await api.mapScene(this.mapName, this.meta.detail_presets[this.detail]);
      this.payload = payload;
      this.scene.setStatic(payload);
      const stats = payload.stats;
      mount(this.statsRow,
        metric('MAP-GRÖSSE', `${fmt.fixed(stats.width, 0)} × ${fmt.fixed(stats.depth, 0)} m`),
        metric('OBJEKTE', String(stats.objects)),
        metric('DECKUNGSDICHTE', `${fmt.fixed(stats.cover_density, 1)}%`),
        metric('Ø SICHTLINIE', `${fmt.fixed(stats.average_sightline, 1)} m`));
      mount(this.noticeHost,
        payload.load_error ? warnBox(payload.load_error) : null,
        h('p', { class: 'hint', text: stats.description }));
      const rows = Object.entries(payload.breakdown || {}).map(([kind, count]) => ({
        label: kind, count,
      }));
      mount(this.breakdownHost, rows.length
        ? table([
          { label: 'OBJEKT-TYP', value: (row) => row.label },
          { label: 'ANZAHL', value: (row) => row.count, align: 'right' },
        ], rows)
        : h('p', { class: 'hint', text: 'Diese Map hat noch keine Deckungsobjekte.' }));
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async randomize() {
    try {
      const payload = await api.mapRandomize(this.mapName);
      this.payload = payload;
      this.scene.setStatic(payload);
      toast(`Deckung für ${this.mapName} neu generiert.`);
      this.load();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async reset() {
    try {
      await api.mapReset(this.mapName);
      this.load();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  renderEditor() {
    if (this.mapName !== 'Custom') {
      mount(this.editorHost, null);
      return;
    }
    const inputs = {};
    const numeric = (label, key, value, step) => {
      const node = h('input', { type: 'number', value, step });
      inputs[key] = node;
      return field(label, node);
    };
    const typeSelect = h('select');
    for (const kind of OBJECT_TYPES) typeSelect.appendChild(h('option', { value: kind, text: kind }));

    const addCard = card('CUSTOM OBJEKT HINZUFÜGEN',
      h('div', { class: 'row' },
        field('TYP', typeSelect),
        numeric('X MITTE', 'x', 0, 1),
        numeric('Y MITTE', 'y', 0, 1),
        numeric('BREITE', 'width', 2, 0.5)),
      h('div', { class: 'row', style: { marginTop: '8px' } },
        numeric('HÖHE', 'height', 1.5, 0.25),
        numeric('TIEFE', 'depth', 2, 0.5),
        numeric('YAW (rad)', 'yaw', 0, 0.1),
        h('div', { class: 'grow' }, button('＋ Objekt hinzufügen', () => this.addObject(typeSelect.value, inputs)))),
      h('p', { class: 'hint', text: 'Alle Angaben in Metern. Das Layout wird sofort in data/custom_map.json gespeichert.' }));

    const objects = this.payload?.serialized?.objects || [];
    const listRows = objects.map((item, index) => h('div', { class: 'list-row' },
      h('span', { text: `${index + 1}. ${item.name || item.kind} · ${fmt.fixed(item.width, 1)}×${fmt.fixed(item.depth, 1)}×${fmt.fixed(item.height, 1)} m` }),
      button('🗑', () => this.deleteObject(index), { className: 'tiny danger' })));

    const importInput = h('input', { type: 'file', accept: '.json,application/json' });
    importInput.addEventListener('change', () => this.importFile(importInput.files[0]));

    const editorCard = card('CUSTOM OBJECT EDITOR',
      h('div', { class: 'row' },
        h('div', { class: 'grow' },
          h('label', { class: 'field', text: 'EXPORT' }),
          h('a', { class: 'button-link ghost', href: api.mapExportUrl, download: 'custom_map.json' },
            '⬇ Export Custom JSON')),
        h('div', { class: 'grow' }, field('IMPORT JSON', importInput))),
      h('div', { class: 'scroll-list', style: { marginTop: '10px' } },
        listRows.length ? listRows : h('p', { class: 'hint', text: 'Custom ist leer. Objekte hinzufügen oder Cover randomisieren.' })));

    mount(this.editorHost, addCard, h('div', { style: { height: '12px' } }), editorCard);
  },

  async addObject(kind, inputs) {
    try {
      const payload = await api.mapAddObject({
        kind,
        x: Number(inputs.x.value),
        y: Number(inputs.y.value),
        width: Number(inputs.width.value),
        height: Number(inputs.height.value),
        depth: Number(inputs.depth.value),
        yaw: Number(inputs.yaw.value),
      });
      this.payload = payload;
      this.scene.setStatic(payload);
      toast('Objekt hinzugefügt und gespeichert.');
      this.renderEditor();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async deleteObject(index) {
    try {
      const payload = await api.mapDeleteObject(index);
      this.payload = payload;
      this.scene.setStatic(payload);
      this.renderEditor();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async importFile(file) {
    if (!file) return;
    try {
      const text = await file.text();
      const response = await fetch('/api/maps/custom/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: text,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      this.payload = payload;
      this.scene.setStatic(payload);
      toast('Custom-Map importiert und gespeichert.');
      this.renderEditor();
    } catch (error) {
      toast(`Import fehlgeschlagen: ${error.message}`, 'error');
    }
  },
};
