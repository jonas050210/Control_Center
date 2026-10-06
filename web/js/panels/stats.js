// STATS panel: training telemetry charts and system resources (canvas, no Plotly).

import { api } from '../api.js';
import { chartCanvas, drawDonut, drawHistogram, drawLine } from '../charts.js';
import { card, fmt, h, metric, mount, poller } from '../dom.js';

export const statsPanel = {
  id: 'stats',
  label: '📊 STATS',

  mount(root) {
    this.data = null;
    this.rewardChart = chartCanvas((ctx, w, hgt) => drawLine(ctx, w, hgt, {
      series: [{ y: this.data?.series.reward || [], color: '#00ff41', fill: true, width: 2.4 }],
      xLabel: 'Timesteps (Index)', yLabel: 'Reward',
      emptyMessage: 'Reward-Daten erscheinen nach dem ersten Trainings-Update.',
    }), { height: 300 });
    this.winChart = chartCanvas((ctx, w, hgt) => drawLine(ctx, w, hgt, {
      series: [
        { y: this.data?.series.win_rate || [], color: '#6cff88', width: 2.4, markers: true },
        { y: this.data?.series.kill_rate || [], color: '#ffaa00', width: 2.4 },
      ],
      xLabel: 'Timesteps (Index)', yLabel: 'Prozent', yMin: 0, yMax: 100,
      emptyMessage: 'Win-Rate erscheint nach den ersten abgeschlossenen Episoden.',
    }), { height: 300 });
    this.ttkChart = chartCanvas((ctx, w, hgt) => {
      const histogram = this.data?.histogram || {};
      drawHistogram(ctx, w, hgt, {
        counts: histogram.counts || countsFromEdges(histogram),
        edges: histogram.edges || [],
        color: '#00cc33',
        xLabel: 'Zeit bis Kill',
      });
    }, { height: 300 });
    this.fpsChart = chartCanvas((ctx, w, hgt) => drawLine(ctx, w, hgt, {
      series: [{ y: this.data?.series.fps || [], color: '#00aaff', width: 2, fill: true }],
      xLabel: 'Timesteps (Index)', yLabel: 'Steps/s',
      emptyMessage: 'FPS-Verlauf erscheint nach dem ersten Log-Intervall.',
    }), { height: 300 });
    this.weaponChart = chartCanvas((ctx, w, hgt) => drawDonut(ctx, w, hgt, {
      parts: Object.entries(this.data?.weapons || {}).map(([label, value], index) => ({
        label, value, color: ['#00ff41', '#00aaff', '#ffaa00', '#ff0040', '#a855f7', '#55ddaa'][index % 6],
      })),
      centerLabel: 'WAFFEN',
      emptyMessage: 'Waffen-Nutzung wird nach Runden gesammelt.',
    }), { height: 300 });

    this.summaryTop = h('div', { class: 'grid cols-4' });
    this.summaryMid = h('div', { class: 'grid cols-4', style: { marginTop: '12px' } });
    this.resources = h('div', { class: 'grid cols-3' });
    this.statusChip = h('div', { class: 'legend' });

    mount(root,
      h('h2', { text: '📊 TRAINING TELEMETRIE · LIVE PERFORMANCE' }),
      h('p', { class: 'hint', text: 'Alle Kennzahlen kommen aus logs/training_metrics.csv und logs/heatmap_events.csv; die Diagramme zeichnet der Browser auf Canvas (kein Plotly).' }),
      this.statusChip,
      this.summaryTop,
      this.summaryMid,
      h('hr', { class: 'sep' }),
      h('div', { class: 'grid cols-2' },
        card('REWARD OVER TIME', this.rewardChart.canvas),
        card('WIN-RATE (grün) vs. KILL-RATE (orange)', this.winChart.canvas)),
      h('div', { class: 'grid cols-2', style: { marginTop: '12px' } },
        card('TTK VERTEILUNG', this.ttkChart.canvas),
        card('WAFFEN-NUTZUNG', this.weaponChart.canvas)),
      h('div', { class: 'grid cols-1', style: { marginTop: '12px' } },
        card('STEPS PRO SEKUNDE', this.fpsChart.canvas)),
      h('h4', { text: 'SYSTEM RESOURCE MONITOR' }),
      this.resources);

    this.poller = poller(() => this.refresh(), 2200);
    this.poller.start();
  },

  onShow() { this.poller.start(); },
  onHide() { this.poller.stop(); },
  unmount() { this.poller.stop(); },

  async refresh() {
    try {
      const data = await api.stats();
      this.data = data;
      const summary = data.summary;
      mount(this.statusChip,
        h('span', { class: `chip ${data.status === 'running' ? 'status-good' : ''}`, text: `STATUS ${(data.status || 'stopped').toUpperCase()}` }),
        h('span', { class: 'chip', text: `TRAININGSZEIT ${fmt.clock(summary.training_time)}` }),
        h('span', { class: 'chip', text: `GELOGGTE EPISODEN ${fmt.num(summary.logged_episodes)}` }),
        data.latest_checkpoint ? h('span', { class: 'chip', text: `CHECKPOINT ${data.latest_checkpoint.split(/[\\/]/).pop()}` }) : null);
      mount(this.summaryTop,
        metric('CURRENT FPS', fmt.num(summary.fps)),
        metric('AVG FPS', fmt.num(summary.avg_fps)),
        metric('TOTAL STEPS', fmt.num(summary.steps)),
        metric('EPISODEN', fmt.num(summary.episodes)));
      mount(this.summaryMid,
        metric('WIN RATE', fmt.percent(summary.win_rate),
          `Kills: ${fmt.percent(summary.kill_rate || 0)}`),
        metric('Ø REWARD', fmt.fixed(summary.avg_reward, 2)),
        metric('Ø TTK (KILLS)', `${fmt.fixed(summary.avg_ttk, 2)}s`,
          `${fmt.num(summary.kill_count || 0)} bestätigte Kills von ${fmt.num(summary.logged_episodes)} Episoden`),
        metric('HEADSHOT-ANTEIL', fmt.percent(summary.headshot_pct)));
      const resources = data.resources || {};
      mount(this.resources,
        metric('CPU AUSLASTUNG', resources.cpu_percent === null ? 'n/a' : `${resources.cpu_percent.toFixed(0)}%`, 'systemweit'),
        metric('PROZESS-RAM', resources.process_mb === null ? 'n/a' : fmt.mb(resources.process_mb),
          resources.children === null ? '' : `${resources.children} Kindprozesse`),
        metric('Ø KILL-DISTANZ', summary.avg_kill_distance === null ? '—' : `${fmt.fixed(summary.avg_kill_distance, 1)}m`,
          `Top-Waffe: ${summary.top_weapon} · Schwäche: ${summary.worst_matchup}`));
      this.rewardChart.render();
      this.winChart.render();
      this.ttkChart.render();
      this.fpsChart.render();
      this.weaponChart.render();
    } catch (error) {
      console.warn(error);
    }
  },
};

// The API already returns histogram counts; this keeps the panel resilient.
function countsFromEdges(histogram) {
  if (histogram && Array.isArray(histogram.counts)) return histogram.counts;
  return [];
}

