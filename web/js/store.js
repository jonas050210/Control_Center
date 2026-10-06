// Shared client state between panels (no persistence, single browser session).

export const store = {
  meta: null,
  arena: {
    running: false,
    detail: 'Balanced',
    sceneKey: null,
    models: { a: 'Heuristic AI', b: 'Heuristic AI' },
    stats: { winsA: 0, winsB: 0, draws: 0, matches: 0 },
  },
  playground: {
    mode: 'shooter',
    recording: false,
    sceneKey: null,
  },
  training: {
    defaults: { workers: null, envsPerWorker: 1 },
  },
  benchmark: {
    best: null,
  },
};

export function rememberBest(best) {
  store.benchmark.best = best;
  if (best) {
    store.training.defaults.workers = best.workers;
    store.training.defaults.envsPerWorker = best.envs_per_worker;
  }
}
