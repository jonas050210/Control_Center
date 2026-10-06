// BENCHMARK panel: worker × envs throughput sweep with a canvas heat matrix.

import { api } from '../api.js';
import { chartCanvas, drawHeatGrid } from '../charts.js';
import { store, rememberBest } from '../store.js';
import { button, card, field, fmt, h, metric, mount, poller, table, toast, warnBox } from '../dom.js';

export const benchmarkPanel = {
  id: 'benchmark',
  label: '🔧 BENCHMARK',

  mount(root) {
    this.meta = store.meta;
    this.snapshot = null;
    this.matrixChart = chartCanvas((ctx, w, hgt) => {
      const data = this.snapshot;
      if (!data || !data.results.length) {
        drawHeatGrid(ctx, w, hgt, null);
        return;
      }
      const completed = data.results.filter((row) => row.status === 'complete');
      const workers = [...new Set(data.results.map((row) => row.workers))].sort((a, b) => a - b);
      const envs = [...new Set(data.results.map((row) => row.envs_per_worker))].sort((a, b) => a - b);
      const grid = envs.map((env) => workers.map((worker) => {
        const row = completed.find((item) => item.workers === worker && item.envs_per_worker === env);
        return row ? row.steps_per_second : 0;
      }));
      drawMatrix(ctx, w, hgt, { workers, envs, grid, best: data.best });
    }, { height: 320 });

    this.mapSelect = h('select');
    for (const name of this.meta.maps) this.mapSelect.appendChild(h('option', { value: name, text: name }));
    this.tableHost = h('div');
    this.summary = h('div', { class: 'grid cols-3' });
    this.progress = h('div', { class: 'bar' }, h('span', { style: { width: '0%' } }));
    this.statusLine = h('p', { class: 'hint' });
    this.warnHost = h('div');
    this.startButton = button('🚀 Run Full Benchmark', () => this.start());
    this.stopButton = button('⏹ Stop', () => this.stop(), { className: 'danger', disabled: true });
    this.applyButton = button('💾 Beste Config ins Training übernehmen', () => this.applyBest(), { className: 'ghost', disabled: true });

    if (!this.meta.training_ready) {
      mount(this.warnHost, warnBox('Benchmark benötigt Stable-Baselines3. Bitte install.py ausführen.'));
    }

    const configs = this.meta.benchmark_configs.length;
    mount(root,
      h('h2', { text: '🔧 CPU BENCHMARK · WORKERS × ENVS' }),
      h('p', { class: 'hint', text: `Jede der ${configs} gültigen Kombinationen läuft 20 Sekunden nach dem Worker-Start (Produkt auf 24 Envs begrenzt). Voller Durchlauf ≈ ${(configs * 20 / 60).toFixed(1)} Minuten plus Prozessstart.` }),
      this.warnHost,
      card('BENCHMARK',
        h('div', { class: 'row' },
          field('BENCHMARK MAP', this.mapSelect),
          this.startButton, this.stopButton, this.applyButton)),
      h('hr', { class: 'sep' }),
      this.statusLine,
      this.progress,
      h('div', { style: { height: '12px' } }),
      this.summary,
      h('div', { style: { height: '12px' } }),
      card('WORKER × ENV DURCHSATZ', this.matrixChart.canvas),
      h('div', { style: { height: '12px' } }),
      card('ERGEBNISSE', this.tableHost));

    this.poller = poller(() => this.refresh(), 2000);
    this.poller.start();
  },

  onShow() { this.poller.start(); },
  onHide() { this.poller.stop(); },
  unmount() { this.poller.stop(); },

  async start() {
    try {
      this.snapshot = await api.benchmarkStart({ map: this.mapSelect.value, seconds_per_combo: 20 });
      this.render();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async stop() {
    try {
      this.snapshot = await api.benchmarkStop();
      this.render();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async refresh() {
    try {
      this.snapshot = await api.benchmark();
      this.render();
    } catch (error) {
      console.warn(error);
    }
  },

  applyBest() {
    const best = this.snapshot?.best;
    if (!best) return;
    rememberBest(best);
    toast(`Training erhält ${best.workers} Worker × ${best.envs_per_worker} Envs – im Training-Tab prüfen.`);
  },

  render() {
    const data = this.snapshot;
    if (!data) return;
    const running = ['running', 'stopping'].includes(data.status);
    this.startButton.disabled = running;
    this.stopButton.disabled = !running;
    this.applyButton.disabled = !data.best;
    const current = data.current ? `${data.current[0]} Worker × ${data.current[1]} Envs` : data.status.toUpperCase();
    this.statusLine.textContent = `STATUS: ${current} · ${data.completed}/${data.total} Kombinationen abgeschlossen`;
    this.progress.firstChild.style.width = `${(data.progress * 100).toFixed(1)}%`;

    const best = data.best;
    mount(this.summary,
      metric('BESTE STEPS / SEK', best ? fmt.num(best.steps_per_second) : '—',
        best ? `${best.workers} Worker × ${best.envs_per_worker} Envs` : 'noch keine Messung'),
      metric('TOTAL ENVS', best ? String(best.total_envs) : '—', best ? `CPU ${fmt.fixed(best.cpu_percent, 0)}%` : ''),
      metric('RAM (PEAK)', best ? fmt.mb(best.ram_mb) : '—', 'inkl. Kindprozesse'));

    if (data.error) {
      mount(this.statusLine, h('div', { class: 'err-box', text: data.error }));
    }
    this.matrixChart.render();

    const completed = data.results.filter((row) => row.status === 'complete');
    const bestIndex = completed.length && best
      ? data.results.findIndex((row) => row.workers === best.workers && row.envs_per_worker === best.envs_per_worker)
      : -1;
    mount(this.tableHost, data.results.length
      ? table([
        { label: 'WORKERS', value: (row) => row.workers },
        { label: 'ENVS / WORKER', value: (row) => row.envs_per_worker },
        { label: 'TOTAL ENVS', value: (row) => row.total_envs },
        { label: 'STEPS / S', value: (row) => fmt.num(row.steps_per_second), align: 'right' },
        { label: 'CPU', value: (row) => `${fmt.fixed(row.cpu_percent, 0)}%`, align: 'right' },
        { label: 'RAM', value: (row) => fmt.mb(row.ram_mb), align: 'right' },
        { label: 'DAUER', value: (row) => `${fmt.fixed(row.seconds, 1)}s`, align: 'right' },
        { label: 'STATUS', value: (row) => (row.status === 'complete' ? '✅ COMPLETE' : row.status.toUpperCase()) },
      ], data.results, { bestIndex })
      : h('p', { class: 'hint', text: 'Noch keine Ergebnisse. Benchmark starten, um die CPU-Durchsätze zu messen.' }));
  },
};

function drawMatrix(ctx, w, h, { workers, envs, grid, best }) {
  const box = { x0: 54, y0: 18, x1: w - 18, y1: h - 30 };
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#141414';
  ctx.fillRect(0, 0, w, h);
  if (!workers.length || !envs.length) {
    ctx.fillStyle = '#6f9a79';
    ctx.font = '12px "Cascadia Code", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Noch keine Messwerte.', w / 2, h / 2);
    return;
  }
  const cellW = (box.x1 - box.x0) / workers.length;
  const cellH = (box.y1 - box.y0) / envs.length;
  const peak = Math.max(1, ...grid.flat());
  envs.forEach((envValue, rowIndex) => {
    workers.forEach((workerValue, columnIndex) => {
      const value = grid[rowIndex][columnIndex];
      const intensity = value / peak;
      ctx.fillStyle = value > 0
        ? `rgba(0, 255, 65, ${0.12 + 0.72 * intensity})`
        : 'rgba(40, 60, 45, 0.35)';
      const x = box.x0 + columnIndex * cellW;
      const y = box.y0 + (envs.length - 1 - rowIndex) * cellH;
      ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
      if (value > 0) {
        ctx.fillStyle = intensity > 0.55 ? '#04140a' : '#d6ffdd';
        ctx.font = 'bold 11px "Cascadia Code", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(Math.round(value).toLocaleString('de-DE'), x + cellW / 2, y + cellH / 2);
      }
      if (best && best.workers === workerValue && best.envs_per_worker === envValue) {
        ctx.strokeStyle = '#ffaa00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 1, y + 1, cellW - 2, cellH - 2);
      }
    });
  });
  ctx.fillStyle = '#8dbf96';
  ctx.font = '10px "Cascadia Code", monospace';
  ctx.textAlign = 'center';
  workers.forEach((value, index) => {
    ctx.fillText(`${value}`, box.x0 + index * cellW + cellW / 2, box.y1 + 12);
  });
  ctx.textAlign = 'right';
  envs.forEach((value, rowIndex) => {
    ctx.fillText(`${value}`, box.x0 - 8, box.y0 + (envs.length - 1 - rowIndex) * cellH + cellH / 2);
  });
  ctx.textAlign = 'left';
  ctx.fillText('Workers →', box.x0, box.y1 + 22);
  ctx.fillText('Envs/worker ↑', box.x0, box.y0 - 8);
}
