// Thin fetch wrapper around the FastAPI backend.

async function request(path, { method = 'GET', body, timeout = 120000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (error) {
        payload = { detail: text.slice(0, 400) };
      }
    }
    if (!response.ok) {
      const message = payload && payload.detail ? payload.detail : `HTTP ${response.status}`;
      throw new Error(message);
    }
    return payload;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error(`Zeitüberschreitung nach ${timeout / 1000}s für ${path}`);
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  request,
  health: () => request('/api/health'),
  meta: () => request('/api/meta'),

  arenaConfig: (payload) => request('/api/arena/config', { method: 'POST', body: payload }),
  arenaState: () => request('/api/arena/state'),
  arenaRun: (running) => request('/api/arena/run', { method: 'POST', body: { running } }),
  arenaStep: (steps = 1) => request('/api/arena/step', { method: 'POST', body: { steps } }),
  arenaReset: () => request('/api/arena/reset', { method: 'POST' }),

  playgroundConfig: (payload) => request('/api/playground/config', { method: 'POST', body: payload }),
  playgroundAction: (payload) => request('/api/playground/action', { method: 'POST', body: payload }),
  playgroundReset: () => request('/api/playground/reset', { method: 'POST' }),
  playgroundRecording: (payload) => request('/api/playground/recording', { method: 'POST', body: payload }),
  saveDemos: () => request('/api/playground/demos/save', { method: 'POST' }),
  demosUrl: '/api/playground/demos.csv',

  aimStart: (duration_seconds) => request('/api/aim/start', { method: 'POST', body: { duration_seconds } }),
  aimTap: (cell) => request('/api/aim/tap', { method: 'POST', body: { cell } }),
  aimState: () => request('/api/aim/state'),
  aimScene: () => request('/api/aim/scene'),

  dodgeStart: () => request('/api/dodge/start', { method: 'POST' }),
  dodgeStep: (dx = 0, dy = 0) => request('/api/dodge/step', { method: 'POST', body: { dx, dy } }),
  dodgeState: () => request('/api/dodge/state'),
  dodgeScene: () => request('/api/dodge/scene'),

  trainingStatus: () => request('/api/training/status'),
  trainingStart: (payload) => request('/api/training/start', { method: 'POST', body: payload }),
  trainingCommand: (command) => request(`/api/training/${command}`, { method: 'POST' }),

  stats: () => request('/api/stats'),
  heatmap: (payload) => request('/api/heatmap', { method: 'POST', body: payload }),

  benchmark: () => request('/api/benchmark'),
  benchmarkStart: (payload) => request('/api/benchmark/start', { method: 'POST', body: payload }),
  benchmarkStop: () => request('/api/benchmark/stop', { method: 'POST' }),

  ttkDefaults: () => request('/api/ttk'),
  ttkSimulate: (payload) => request('/api/ttk/simulate', { method: 'POST', body: payload, timeout: 180000 }),

  maps: () => request('/api/maps'),
  mapScene: (name, detail, spawns = true) =>
    request(`/api/maps/${encodeURIComponent(name)}/scene?detail=${detail}&spawns=${spawns ? 1 : 0}`),
  mapRandomize: (name) => request(`/api/maps/${encodeURIComponent(name)}/randomize`, { method: 'POST' }),
  mapReset: (name) => request(`/api/maps/${encodeURIComponent(name)}/reset`, { method: 'POST' }),
  mapAddObject: (payload) => request('/api/maps/custom/objects', { method: 'POST', body: payload }),
  mapDeleteObject: (index) =>
    request(`/api/maps/custom/objects/${index}`, { method: 'DELETE' }),
  mapExportUrl: '/api/maps/custom/export',
};
