// HEATMAP panel: death / safe / kill zones from logged episodes.

import { api } from '../api.js';
import { chartCanvas, drawHeatGrid } from '../charts.js';
import { store } from '../store.js';
import { button, card, field, fmt, h, metric, mount, poller, toast } from '../dom.js';

export const heatmapPanel = {
  id: 'heatmap',
  label: '🔥 HEATMAP',

  mount(root) {
    this.meta = store.meta;
    this.payload = null;
    this.filters = {
      map: 'Dust',
      weapon: 'All weapons',
      mode: 'Both',
      episodeMin: 1,
      episodeMax: 0,
      distanceMin: 0,
      distanceMax: 100,
    };

    this.chart = chartCanvas((ctx, w, hgt) => drawHeatGrid(ctx, w, hgt, this.payload), { height: 560 });

    this.mapSelect = h('select');
    for (const name of this.meta.maps) this.mapSelect.appendChild(h('option', { value: name, text: name }));
    this.mapSelect.value = this.filters.map;
    this.mapSelect.addEventListener('change', () => {
      this.filters.map = this.mapSelect.value;
      this.refresh();
    });
    this.weaponSelect = h('select');
    this.modeSelect = h('select');
    for (const mode of ['Both', 'Deaths', 'Kills']) {
      this.modeSelect.appendChild(h('option', { value: mode, text: mode }));
    }
    this.modeSelect.addEventListener('change', () => {
      this.filters.mode = this.modeSelect.value;
      this.refresh();
    });
    const numberInput = (key) => {
      const node = h('input', { type: 'number', value: this.filters[key] });
      node.addEventListener('change', () => {
        this.filters[key] = Number(node.value);
        this.refresh();
      });
      this[key] = node;
      return node;
    };

    this.statsRow = h('div', { class: 'grid cols-4' });

    mount(root,
      h('h2', { text: '🔥 HEATMAP · DEATH / SAFE / KILL ZONEN' }),
      h('p', { class: 'hint', text: 'Zonenwerte kommen aus logs/heatmap_events.csv (Training) und aus den Browser-Matches dieser Sitzung. Rot = Tode, gelb = Kills, grün = unauffällige Bereiche.' }),
      card('FILTER',
        h('div', { class: 'row' },
          field('MAP', this.mapSelect),
          field('WEAPON', this.weaponSelect),
          field('ANZEIGE', this.modeSelect),
          field('EPISODE MIN', numberInput('episodeMin')),
          field('EPISODE MAX', numberInput('episodeMax'))),
        h('div', { class: 'row', style: { marginTop: '10px' } },
          field('DISTANZ MIN (m)', numberInput('distanceMin')),
          field('DISTANZ MAX (m)', numberInput('distanceMax')),
          button('🔄 Neu laden', () => this.refresh(), { className: 'ghost' }))),
      this.statsRow,
      h('hr', { class: 'sep' }),
      card('ZONENKARTE', this.chart.canvas));

    this.poller = poller(() => this.refresh(), 6000);
    this.refresh();
    this.poller.start();
  },

  onShow() { this.poller?.start(); },
  onHide() { this.poller?.stop(); },
  unmount() { this.poller?.stop(); },

  async refresh() {
    try {
      const payload = await api.heatmap({
        map: this.filters.map,
        weapon: this.filters.weapon,
        mode: this.filters.mode,
        episode_range: [this.filters.episodeMin, Math.max(this.filters.episodeMin, this.filters.episodeMax)],
        distance_range: [this.filters.distanceMin, Math.max(this.filters.distanceMin, this.filters.distanceMax)],
      });
      this.payload = payload;
      if (!this.weaponSelect.options.length && payload.weapons) {
        for (const weapon of payload.weapons) {
          this.weaponSelect.appendChild(h('option', { value: weapon, text: weapon }));
        }
        this.weaponSelect.value = 'All weapons';
        this.weaponSelect.addEventListener('change', () => {
          this.filters.weapon = this.weaponSelect.value;
          this.refresh();
        });
      }
      if (payload.max_episode && this.episodeMax.value === '0') {
        this.episodeMax.value = String(payload.max_episode);
        this.filters.episodeMax = payload.max_episode;
      }
      mount(this.statsRow,
        metric('TREFFER IM FILTER', fmt.num(payload.events), 'geloggte Episoden-Events'),
        metric('KILLS', fmt.num(payload.kills), 'im Raster'),
        metric('TODE', fmt.num(payload.deaths), 'im Raster'),
        metric('MAP-DECKUNG', `${fmt.fixed(payload.width, 0)} × ${fmt.fixed(payload.depth, 0)} m`,
          payload.hottest_death ? `Todes-Hotspot bei X ${fmt.fixed(payload.hottest_death.x, 0)} / Y ${fmt.fixed(payload.hottest_death.y, 0)}` : 'keine Tode geloggt'));
      this.chart.render();
    } catch (error) {
      toast(error.message, 'error');
    }
  },
};
