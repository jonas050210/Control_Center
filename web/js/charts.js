// Small canvas chart library (line, bars, histogram, donut, zone heatmap).
// Replaces Plotly: the browser draws directly, so there is no 3 MB JS payload.

import { h } from './dom.js';

const AXIS = '#31533a';
const GRID = '#233c30';
const TEXT = '#8dbf96';
const NEON = '#00ff41';

export function chartCanvas(draw, { height = 260 } = {}) {
  const canvas = h('canvas', { class: 'chart' });
  canvas.style.height = `${height}px`;
  const render = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(220, canvas.clientWidth || canvas.parentElement?.clientWidth || 320);
    const hgt = Math.max(120, canvas.clientHeight || height);
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(hgt * dpr));
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, hgt);
    ctx.fillStyle = '#141414';
    ctx.fillRect(0, 0, width, hgt);
    try {
      draw(ctx, width, hgt);
    } catch (error) {
      ctx.fillStyle = '#ff0040';
      ctx.font = '11px monospace';
      ctx.fillText(`Chart-Fehler: ${error.message}`, 10, 20);
    }
  };
  if (typeof ResizeObserver !== 'undefined') {
    const observer = new ResizeObserver(() => render());
    observer.observe(canvas);
  } else {
    window.addEventListener('resize', render);
  }
  requestAnimationFrame(render);
  return { canvas, render };
}

function frame(ctx, w, h, { left = 46, right = 14, top = 16, bottom = 28 } = {}) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#141414';
  ctx.fillRect(0, 0, w, h);
  return { x0: left, y0: top, x1: w - right, y1: h - bottom, width: w - left - right, height: h - top - bottom };
}

function text(ctx, value, x, y, { align = 'left', color = TEXT, size = 10, bold = false } = {}) {
  ctx.fillStyle = color;
  ctx.font = `${bold ? 'bold ' : ''}${size}px "Cascadia Code", Consolas, monospace`;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillText(String(value), x, y);
}

function empty(ctx, w, h, message) {
  text(ctx, message, w / 2, h / 2, { align: 'center', color: '#6f9a79', size: 12 });
}

function gridAndAxis(ctx, box, { yMin, yMax, yTicks = 4, xTicks = 5, xLabels = null, yFormat = (v) => v.toFixed(1) }) {
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  for (let index = 0; index <= yTicks; index += 1) {
    const ratio = index / yTicks;
    const y = box.y1 - ratio * box.height;
    ctx.beginPath();
    ctx.moveTo(box.x0, y);
    ctx.lineTo(box.x1, y);
    ctx.stroke();
    text(ctx, yFormat(yMin + ratio * (yMax - yMin)), box.x0 - 6, y, { align: 'right' });
  }
  for (let index = 0; index <= xTicks; index += 1) {
    const ratio = index / xTicks;
    const x = box.x0 + ratio * box.width;
    if (index > 0 && index < xTicks) {
      ctx.beginPath();
      ctx.moveTo(x, box.y0);
      ctx.lineTo(x, box.y1);
      ctx.stroke();
    }
    if (xLabels) text(ctx, xLabels[index], x, box.y1 + 12, { align: 'center' });
  }
  ctx.strokeStyle = AXIS;
  ctx.beginPath();
  ctx.moveTo(box.x0, box.y0);
  ctx.lineTo(box.x0, box.y1);
  ctx.lineTo(box.x1, box.y1);
  ctx.stroke();
}

function niceBounds(values, { includeZero = true } = {}) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (!finite.length) return { min: 0, max: 1 };
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (includeZero) {
    min = Math.min(0, min);
    max = Math.max(0, max);
  }
  if (max - min < 1e-9) {
    max = min + 1;
  }
  const padding = (max - min) * 0.08;
  return { min: min - (min < 0 ? padding : 0), max: max + padding };
}

// ------------------------------------------------------------------ line plot

export function drawLine(ctx, w, h, data) {
  const { series = [], xLabel = '', yLabel = '', yMin = null, yMax = null, emptyMessage = 'Keine Daten' } = data || {};
  const usable = series.filter((entry) => entry && entry.y && entry.y.some((value) => Number.isFinite(value)));
  if (!usable.length) {
    frame(ctx, w, h);
    empty(ctx, w, h, emptyMessage);
    return;
  }
  const box = frame(ctx, w, h, { left: 52, bottom: 30, top: 14 });
  const allY = usable.flatMap((entry) => entry.y.map((value) => (Number.isFinite(value) ? value : null)))
    .filter((value) => value !== null);
  const bounds = niceBounds(allY, { includeZero: true });
  const lower = yMin === null ? bounds.min : yMin;
  const upper = yMax === null ? bounds.max : yMax;
  const span = upper - lower || 1;
  const length = Math.max(...usable.map((entry) => entry.y.length));
  const xAt = (index) => box.x0 + (length <= 1 ? 0 : (index / (length - 1)) * box.width);
  const yAt = (value) => box.y1 - ((value - lower) / span) * box.height;

  const step = Math.max(1, Math.round(length / 5));
  const xLabels = [];
  for (let index = 0; index < 6; index += 1) {
    const position = Math.min(length - 1, index * step);
    xLabels.push(length > 999 ? `${Math.round(position / 1000)}k` : String(position));
  }
  gridAndAxis(ctx, box, {
    yMin: lower,
    yMax: upper,
    xLabels,
    yFormat: (value) => (Math.abs(value) >= 1000 ? `${Math.round(value / 1000)}k` : value.toFixed(Math.abs(value) < 10 ? 1 : 0)),
  });

  for (const entry of usable) {
    const color = entry.color || NEON;
    ctx.strokeStyle = color;
    ctx.lineWidth = entry.width || 2;
    ctx.beginPath();
    let started = false;
    entry.y.forEach((value, index) => {
      if (!Number.isFinite(value)) {
        started = false;
        return;
      }
      const x = xAt(index);
      const y = yAt(value);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    if (entry.fill) {
      const gradient = ctx.createLinearGradient(0, box.y0, 0, box.y1);
      gradient.addColorStop(0, `${color}33`);
      gradient.addColorStop(1, `${color}05`);
      ctx.fillStyle = gradient;
      ctx.beginPath();
      let startedFill = false;
      entry.y.forEach((value, index) => {
        if (!Number.isFinite(value)) return;
        const x = xAt(index);
        const y = yAt(value);
        if (!startedFill) {
          ctx.moveTo(x, box.y1);
          ctx.lineTo(x, y);
          startedFill = true;
        } else {
          ctx.lineTo(x, y);
        }
      });
      const lastIndex = entry.y.map(Number.isFinite).lastIndexOf(true);
      if (startedFill && lastIndex >= 0) {
        ctx.lineTo(xAt(lastIndex), box.y1);
        ctx.closePath();
        ctx.fill();
      }
    }
    if (entry.markers) {
      ctx.fillStyle = color;
      entry.y.forEach((value, index) => {
        if (!Number.isFinite(value)) return;
        ctx.beginPath();
        ctx.arc(xAt(index), yAt(value), 2, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }
  if (xLabel) text(ctx, xLabel, box.x1, box.y1 + 20, { align: 'right' });
  if (yLabel) text(ctx, yLabel, box.x0 - 40, box.y0 - 6);
}

// ------------------------------------------------------------------ bar chart

export function drawBars(ctx, w, h, data) {
  const { labels = [], values = [], colors = [], valueFormat = (value) => value.toFixed(2), emptyMessage = 'Keine Daten' } = data || {};
  if (!labels.length) {
    frame(ctx, w, h);
    empty(ctx, w, h, emptyMessage);
    return;
  }
  const box = frame(ctx, w, h, { left: 52, bottom: 32, top: 18 });
  const numeric = values.map((value) => (Number.isFinite(value) ? value : 0));
  const bounds = niceBounds([...numeric, 0]);
  const span = bounds.max - bounds.min || 1;
  gridAndAxis(ctx, box, {
    yMin: bounds.min,
    yMax: bounds.max,
    xTicks: Math.max(1, labels.length - 1),
    xLabels: labels,
    yFormat: (value) => value.toFixed(Math.abs(value) < 10 ? 1 : 0),
  });
  const slot = box.width / labels.length;
  const barWidth = Math.min(78, slot * 0.6);
  labels.forEach((label, index) => {
    const value = numeric[index];
    const x = box.x0 + slot * index + slot / 2;
    const zeroY = box.y1 - ((0 - bounds.min) / span) * box.height;
    const y = box.y1 - ((value - bounds.min) / span) * box.height;
    ctx.fillStyle = colors[index] || NEON;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(x - barWidth / 2, Math.min(y, zeroY), barWidth, Math.max(1, Math.abs(zeroY - y)));
    ctx.globalAlpha = 1;
    text(ctx, valueFormat(value), x, Math.min(y, zeroY) - 10, { align: 'center', color: '#d6ffdd', size: 11, bold: true });
  });
}

// ------------------------------------------------------------------ histogram

export function drawHistogram(ctx, w, h, data) {
  const { counts = [], edges = [], color = NEON, xLabel = '', yLabel = 'Runden', emptyMessage = 'Noch keine Runden geloggt.' } = data || {};
  const total = counts.reduce((sum, value) => sum + value, 0);
  if (!counts.length || total === 0) {
    frame(ctx, w, h);
    empty(ctx, w, h, emptyMessage);
    return;
  }
  const box = frame(ctx, w, h, { left: 52, bottom: 30, top: 16 });
  const maxCount = Math.max(...counts);
  gridAndAxis(ctx, box, {
    yMin: 0,
    yMax: maxCount,
    xTicks: 4,
    xLabels: [0, 1, 2, 3, 4].map((index) => {
      const position = Math.round((index / 4) * (edges.length - 1));
      const value = edges[position] ?? 0;
      return `${Number(value).toFixed(1)}s`;
    }),
    yFormat: (value) => value.toFixed(0),
  });
  const slot = box.width / counts.length;
  counts.forEach((count, index) => {
    const height = (count / maxCount) * box.height;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.8;
    ctx.fillRect(box.x0 + slot * index + 1, box.y1 - height, Math.max(1, slot - 2), height);
    ctx.globalAlpha = 1;
  });
  if (xLabel) text(ctx, xLabel, box.x1, box.y1 + 20, { align: 'right' });
  text(ctx, `Ø ${(edges.reduce((sum, value) => sum + value, 0) / Math.max(1, edges.length)).toFixed(2)}s`, box.x0, box.y0 - 6);
}

// ------------------------------------------------------------------- donut

export function drawDonut(ctx, w, h, data) {
  const { parts = [], centerLabel = '', emptyMessage = 'Keine Daten' } = data || {};
  const usable = parts.filter((part) => Number(part.value) > 0);
  if (!usable.length) {
    frame(ctx, w, h);
    empty(ctx, w, h, emptyMessage);
    return;
  }
  frame(ctx, w, h);
  const total = usable.reduce((sum, part) => sum + Number(part.value), 0);
  const cx = w * 0.34;
  const cy = h / 2;
  const radius = Math.min(w * 0.24, h * 0.36);
  let angle = -Math.PI / 2;
  for (const part of usable) {
    const slice = (Number(part.value) / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = part.color || NEON;
    ctx.globalAlpha = 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#0a0a0a';
    ctx.lineWidth = 2;
    ctx.stroke();
    angle += slice;
  }
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 0.58, 0, Math.PI * 2);
  ctx.fillStyle = '#141414';
  ctx.fill();
  if (centerLabel) text(ctx, centerLabel, cx, cy, { align: 'center', color: '#d6ffdd', size: 12, bold: true });
  usable.forEach((part, index) => {
    const y = 22 + index * 16;
    const x = w * 0.62;
    ctx.fillStyle = part.color || NEON;
    ctx.fillRect(x, y - 5, 9, 9);
    text(ctx, `${part.label} · ${Number(part.value).toLocaleString('de-DE', { maximumFractionDigits: 1 })}`,
      x + 14, y, { color: '#b5e8bd', size: 11 });
  });
}

// -------------------------------------------------------------- zone heatmap

export function drawHeatGrid(ctx, w, h, payload) {
  const box = frame(ctx, w, h, { left: 40, right: 16, top: 16, bottom: 40 });
  if (!payload || !payload.grid) {
    empty(ctx, w, h, 'Keine Heatmap-Daten.');
    return;
  }
  const { grid, width, depth, objects = [], hottest_kill: hottestKill, hottest_death: hottestDeath } = payload;
  const scale = Math.min(box.width / width, box.height / depth);
  const mapWidth = width * scale;
  const mapHeight = depth * scale;
  const originX = box.x0 + (box.width - mapWidth) / 2;
  const originY = box.y0 + (box.height - mapHeight) / 2;
  const toX = (x) => originX + (x + width / 2) * scale;
  const toY = (y) => originY + mapHeight - (y + depth / 2) * scale;

  ctx.save();
  ctx.fillStyle = '#0c1a10';
  ctx.fillRect(originX, originY, mapWidth, mapHeight);
  const rows = grid.length;
  const cols = rows ? grid[0].length : 0;
  const cellW = mapWidth / Math.max(1, cols);
  const cellH = mapHeight / Math.max(1, rows);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < cols; column += 1) {
      const value = Number(grid[row][column]) || 0;
      if (Math.abs(value) < 1e-4) continue;
      const intensity = Math.min(1, Math.abs(value));
      ctx.fillStyle = value > 0
        ? `rgba(255, 170, 0, ${0.16 + 0.74 * intensity})`
        : `rgba(255, 0, 64, ${0.16 + 0.74 * intensity})`;
      // grid rows run along y, columns along x.
      ctx.fillRect(originX + column * cellW, originY + (rows - 1 - row) * cellH, cellW + 0.6, cellH + 0.6);
    }
  }
  for (const item of objects) {
    ctx.fillStyle = 'rgba(150, 160, 152, 0.22)';
    ctx.strokeStyle = item.color || '#8d795d';
    ctx.lineWidth = 1;
    const x = toX(item.x0);
    const y = toY(item.y1);
    const objectWidth = Math.max(1.5, (item.x1 - item.x0) * scale);
    const objectHeight = Math.max(1.5, (item.y1 - item.y0) * scale);
    ctx.fillRect(x, y, objectWidth, objectHeight);
    ctx.strokeRect(x, y, objectWidth, objectHeight);
  }
  ctx.strokeStyle = payload.theme || NEON;
  ctx.lineWidth = 1.4;
  ctx.strokeRect(originX, originY, mapWidth, mapHeight);

  const marker = (zone, color, label) => {
    if (!zone) return;
    const x = toX(zone.x);
    const y = toY(zone.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.stroke();
    text(ctx, label, x + 9, y, { color, size: 10, bold: true });
  };
  marker(hottestKill, '#ffaa00', `KILL-HOTSPOT ×${hottestKill ? hottestKill.count : 0}`);
  marker(hottestDeath, '#ff0040', `DEATH-HOTSPOT ×${hottestDeath ? hottestDeath.count : 0}`);
  ctx.restore();

  // axes in metres
  const steps = 4;
  for (let index = 0; index <= steps; index += 1) {
    const ratio = index / steps;
    const xValue = -width / 2 + ratio * width;
    const yValue = -depth / 2 + ratio * depth;
    text(ctx, `${xValue.toFixed(0)}m`, originX + ratio * mapWidth, originY + mapHeight + 12, { align: 'center' });
    text(ctx, `${yValue.toFixed(0)}m`, originX - 6, toY(yValue), { align: 'right' });
  }
  text(ctx, 'X →', originX + mapWidth / 2, originY + mapHeight + 26, { align: 'center' });
  text(ctx, payload.map || '', originX, originY - 8, { color: '#d6ffdd', size: 11, bold: true });

  // legend
  const legendY = h - 12;
  const legendX = box.x1 - 250;
  const gradient = ctx.createLinearGradient(legendX, 0, legendX + 160, 0);
  gradient.addColorStop(0, 'rgba(255, 0, 64, 0.9)');
  gradient.addColorStop(0.5, 'rgba(0, 168, 51, 0.55)');
  gradient.addColorStop(1, 'rgba(255, 170, 0, 0.9)');
  ctx.fillStyle = gradient;
  ctx.fillRect(legendX, legendY - 6, 160, 8);
  text(ctx, 'Death', legendX - 4, legendY - 2, { align: 'right', size: 10 });
  text(ctx, 'Safe', legendX + 80, legendY - 2, { align: 'center', size: 10 });
  text(ctx, 'Kill', legendX + 164, legendY - 2, { size: 10 });
}
