// TRAINING panel: background PPO controller with live metrics and worker log.

import { api } from '../api.js';
import { store } from '../store.js';
import {
  button, card, checkboxField, errorBox, field, fmt, h, metric, mount, poller, setText, statusClass, toast,
} from '../dom.js';

const DURATIONS = [
  { label: '10 Minuten', minutes: 10 },
  { label: '30 Minuten', minutes: 30 },
  { label: '1 Stunde', minutes: 60 },
  { label: '2 Stunden', minutes: 120 },
  { label: '5 Stunden', minutes: 300 },
  { label: 'Custom', minutes: null },
];

const METHODS = ['Imitation → RL', 'Pure RL', 'Resume Checkpoint'];

export const trainingPanel = {
  id: 'training',
  label: '🏋️ TRAINING',

  mount(root) {
    this.meta = store.meta;
    this.state = {
      durationMinutes: 10,
      customMinutes: 45,
      workers: store.training.defaults.workers ?? this.meta.cpu.auto_workers,
      envsPerWorker: store.training.defaults.envsPerWorker ?? 1,
      method: METHODS[1],
      curriculum: true,
      selfPlay: true,
      map: 'Dust',
      vision: store.meta.default_vision_mode || 'coarse_los',
      episodeSeconds: store.training.defaults.episodeSeconds ?? 60,
      gate: store.training.defaults.gate ?? 40,
      checkpoint: null,
      running: false,
    };

    const durationSelect = h('select');
    for (const entry of DURATIONS) durationSelect.appendChild(h('option', { value: entry.label, text: entry.label }));
    durationSelect.addEventListener('change', () => {
      const entry = DURATIONS.find((item) => item.label === durationSelect.value);
      this.state.durationMinutes = entry.minutes;
      this.customField.style.display = entry.minutes === null ? '' : 'none';
      this.renderEstimate();
    });
    const customInput = h('input', { type: 'number', min: 1, max: 7200, value: this.state.customMinutes });
    customInput.addEventListener('change', () => {
      this.state.customMinutes = Number(customInput.value);
      this.renderEstimate();
    });
    this.customField = field('Custom Minuten', customInput);
    this.customField.style.display = 'none';

    this.workersRange = h('input', { type: 'range', min: 1, max: 20, value: this.state.workers });
    this.workersValue = h('span', { class: 'hint', text: String(this.state.workers) });
    this.workersRange.addEventListener('input', () => {
      this.state.workers = Number(this.workersRange.value);
      this.workersValue.textContent = String(this.state.workers);
      this.renderEstimate();
    });
    this.envsRange = h('input', { type: 'range', min: 1, max: 8, value: this.state.envsPerWorker });
    this.envsValue = h('span', { class: 'hint', text: String(this.state.envsPerWorker) });
    this.envsRange.addEventListener('input', () => {
      this.state.envsPerWorker = Number(this.envsRange.value);
      this.envsValue.textContent = String(this.state.envsPerWorker);
      this.renderEstimate();
    });

    this.methodSelect = h('select');
    for (const method of METHODS) this.methodSelect.appendChild(h('option', { value: method, text: method }));
    this.methodSelect.value = this.state.method;
    this.methodSelect.addEventListener('change', () => {
      this.state.method = this.methodSelect.value;
      this.checkpointField.style.display = this.state.method === 'Resume Checkpoint' ? '' : 'none';
    });
    this.checkpointField = field('CHECKPOINT', h('select'));
    this.checkpointField.style.display = 'none';

    this.mapSelect = h('select');
    for (const name of this.meta.maps) this.mapSelect.appendChild(h('option', { value: name, text: name }));
    this.mapSelect.addEventListener('change', () => { this.state.map = this.mapSelect.value; });

    // Short episodes finish more fights per minute - the single biggest
    // throughput lever for PPO in a 1v1 arena.
    this.episodeInput = h('input', {
      type: 'number', min: 15, max: 300, step: 5, value: this.state.episodeSeconds,
    });
    this.episodeInput.addEventListener('change', () => {
      this.state.episodeSeconds = Math.min(300, Math.max(15, Number(this.episodeInput.value) || 60));
      this.episodeInput.value = this.state.episodeSeconds;
      this.renderEstimate();
    });

    // The curriculum only moves on once the current phase is actually beaten;
    // 0 % disables the gate (pure time-based ramp, the old behaviour).
    this.gateInput = h('input', {
      type: 'number', min: 0, max: 100, step: 5, value: this.state.gate,
    });
    this.gateInput.addEventListener('change', () => {
      this.state.gate = Math.min(100, Math.max(0, Number(this.gateInput.value) || 0));
      this.gateInput.value = this.state.gate;
    });

    // Perception: how much the policy is allowed to see of the opponent.
    const VISION_LABELS = {
      coarse_los: 'Realistisch: Deckung + Sichtkegel',
      coarse: 'Grob, aber immer verfolgt',
      noisy: 'Exakt mit Rauschen',
      exact: 'Exakt (alt, kennt Position durch Wände)',
    };
    this.visionSelect = h('select');
    for (const mode of this.meta.vision_modes || ['coarse_los']) {
      this.visionSelect.appendChild(h('option', { value: mode, text: VISION_LABELS[mode] || mode }));
    }
    this.visionSelect.value = this.state.vision;
    this.visionSelect.addEventListener('change', () => { this.state.vision = this.visionSelect.value; });

    this.estimate = h('p', { class: 'hint' });
    this.form = h('div');

    this.statusCard = h('div');
    this.metricsRow = h('div', { class: 'grid cols-4' });
    this.metricsRow2 = h('div', { class: 'grid cols-4', style: { marginTop: '12px' } });
    this.logNode = h('pre', { class: 'log' });
    this.progressBar = h('div', { class: 'bar' }, h('span', { style: { width: '0%' } }));
    this.progressText = h('p', { class: 'hint' });
    this.errorHost = h('div');

    mount(root,
      h('h2', { text: '🏋️ PPO TRAINING · CPU WORKER CONTROL' }),
      h('p', { class: 'hint', text: `CPU-only Training · ${this.meta.cpu.physical} physische / ${this.meta.cpu.logical} logische Kerne erkannt · SB3 VecNormalize · Training läuft in einem Server-Thread weiter, auch wenn der Tab gewechselt wird.` }),
      this.errorHost,
      card('LAUF KONFIGURIEREN',
        h('div', { class: 'row' },
          field('TRAININGSDauer', durationSelect),
          this.customField,
          field('TRAINING MAP', this.mapSelect),
          field('METHODE', this.methodSelect)),
        h('div', { class: 'row', style: { marginTop: '10px' } },
          h('div', { class: 'grow' },
            h('label', { class: 'field' }, 'WORKERS ', this.workersValue), this.workersRange),
          h('div', { class: 'grow' },
            h('label', { class: 'field' }, 'ENVS PRO WORKER ', this.envsValue), this.envsRange)),
        h('div', { class: 'row', style: { marginTop: '10px' } },
          checkboxField('4-Phasen Curriculum', true, (checked) => { this.state.curriculum = checked; }),
          checkboxField('Frozen Self-Play ab Phase 3', true, (checked) => { this.state.selfPlay = checked; }),
          this.checkpointField),
        h('div', { class: 'row', style: { marginTop: '10px' } },
          field('SICHT DER AI (WAHRNEHMUNG)', this.visionSelect),
          field('EPISODENLÄNGE (SEKUNDEN)', this.episodeInput),
          field('FREISCHALTUNG: KILL-RATE %', this.gateInput)),
        this.estimate),
      h('div', { class: 'row', style: { marginTop: '12px' } },
        this.startButton = button('🚀 Start Training', () => this.start()),
        this.pauseButton = button('⏸ Pause / Resume', () => this.command('pause-or-resume'), { disabled: true }),
        this.stopButton = button('⏹ Stop', () => this.command('stop'), { className: 'danger', disabled: true }),
        this.saveButton = button('💾 Save Checkpoint', () => this.command('save'), { disabled: true })),
      h('hr', { class: 'sep' }),
      card('LIVE STATUS', this.statusCard, this.progressText, this.progressBar,
        this.metricsRow, this.metricsRow2),
      h('div', { style: { height: '12px' } }),
      card('LIVE WORKER LOG', this.logNode));

    this.renderEstimate();
    this.loadCheckpoints();
    this.poller = poller(() => this.refresh(), 1200);
    this.active = true;
    this.poller.start();
  },

  onShow() {
    this.active = true;
    const defaults = store.training.defaults;
    if (defaults.workers && defaults.workers !== this.state.workers) {
      this.state.workers = defaults.workers;
      this.state.envsPerWorker = defaults.envsPerWorker || this.state.envsPerWorker;
      this.workersRange.value = String(this.state.workers);
      this.workersValue.textContent = String(this.state.workers);
      this.envsRange.value = String(this.state.envsPerWorker);
      this.envsValue.textContent = String(this.state.envsPerWorker);
      this.renderEstimate();
    }
    this.poller.start();
  },

  onHide() {
    this.active = false;
    this.poller.stop();
  },

  renderEstimate() {
    const minutes = this.state.durationMinutes === null ? this.state.customMinutes : this.state.durationMinutes;
    const requested = this.state.workers * this.state.envsPerWorker;
    const effective = Math.min(requested, this.meta.max_envs);
    const steps = Math.max(50_000, Math.round(minutes * 60 * 1_500));
    mount(this.estimate,
      h('span', { class: 'chip', text: `${effective} Vektor-Envs aktiv` }),
      requested > this.meta.max_envs
        ? warnChip(`angefragt: ${requested} → auf ${this.meta.max_envs} begrenzt`)
        : h('span', { class: 'chip', text: 'ein SubprocVecEnv-Kind pro Env' }),
      h('span', { class: 'chip', text: `Rollout-Horizont ≈ ${fmt.num(steps)} Timesteps` }),
      h('span', { class: 'chip', text: 'Wall-Clock-Dauer ist der harte Stopp' }));
  },

  async loadCheckpoints() {
    try {
      const meta = await api.meta();
      store.meta = meta;
      const select = this.checkpointField.querySelector('select');
      if (!select) return;
      const options = meta.models.filter((name) => name !== meta.heuristic);
      select.replaceChildren(...(options.length
        ? options.map((name) => h('option', { value: name, text: name }))
        : [h('option', { value: '', text: 'Kein Checkpoint in models/' })]));
      this.state.checkpoint = options[0] || null;
    } catch (error) {
      console.warn(error);
    }
  },

  async start() {
    const minutes = this.state.durationMinutes === null ? this.state.customMinutes : this.state.durationMinutes;
    const payload = {
      duration_minutes: minutes,
      workers: this.state.workers,
      envs_per_worker: this.state.envsPerWorker,
      map: this.state.map,
      method: this.state.method,
      curriculum: this.state.curriculum,
      self_play: this.state.selfPlay,
      resume_checkpoint: this.state.checkpoint,
      max_envs: this.meta.max_envs,
      vision: this.state.vision,
      episode_seconds: this.state.episodeSeconds,
      curriculum_min_win_rate: this.state.gate / 100,
    };
    try {
      const response = await api.trainingStart(payload);
      toast(`Training gestartet · ${fmt.num(response.estimated_steps)} Timesteps geplant`);
      this.refresh();
    } catch (error) {
      toast(error.message, 'error');
      mount(this.errorHost, errorBox(error.message));
    }
  },

  async command(action) {
    try {
      if (action === 'pause-or-resume') {
        const status = await api.trainingStatus();
        await api.trainingCommand(status.status === 'paused' ? 'resume' : 'pause');
      } else {
        await api.trainingCommand(action);
      }
      this.refresh();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async refresh() {
    try {
      const snapshot = await api.trainingStatus();
      this.render(snapshot);
    } catch (error) {
      console.warn(error);
    }
  },

  render(snapshot) {
    const status = snapshot.status || 'stopped';
    const metrics = snapshot.metrics || {};
    const running = ['starting', 'running', 'paused', 'stopping'].includes(status);
    this.pauseButton.disabled = !running;
    this.stopButton.disabled = !running;
    this.saveButton.disabled = !running;
    this.startButton.disabled = running;
    setText(this.pauseButton, status === 'paused' ? '▶ Resume' : '⏸ Pause');

    const label = {
      running: '● TRAINING', starting: '● STARTING', paused: '● PAUSED',
      stopping: '● STOPPING', complete: '● COMPLETE', error: '● ERROR', stopped: '● STOPPED',
    }[status] || `● ${status.toUpperCase()}`;

    const config = snapshot.config;
    const remaining = config ? Math.max(0, config.duration_seconds - (metrics.elapsed || 0)) : 0;
    const eta = config && config.duration_seconds
      ? `ETA ${fmt.clock(remaining)}`
      : 'Schritt-Ziel';
    mount(this.statusCard,
      h('span', { class: `card-label-top ${statusClass(status)}`, text: label }),
      h('span', { class: 'hint', text: ` ${fmt.num(metrics.effective_envs || 0)} Vektor-Envs · ${eta}`
        + (config ? ` · ${config.n_workers}×${config.envs_per_worker} auf ${config.map_name} (${config.method})` : '') }));

    const progress = Math.max(0, Math.min(1, Number(metrics.progress) || 0));
    this.progressBar.firstChild.style.width = `${(progress * 100).toFixed(1)}%`;
    setText(this.progressText, `PPO Rollout-Fortschritt · ${(progress * 100).toFixed(1)}%`
      + (snapshot.latest_checkpoint ? ` · letzter Checkpoint: ${snapshot.latest_checkpoint.split(/[\\/]/).pop()}` : ''));

    mount(this.metricsRow,
      metric('TOTAL STEPS', fmt.num(metrics.timesteps || 0)),
      metric('STEPS / SEK', fmt.num(metrics.fps || 0)),
      metric('EPISODEN', fmt.num(metrics.episodes || 0)),
      metric('WIN RATE', fmt.percent(metrics.win_rate || 0)),
      metric('DAVON KILLS', fmt.percent(metrics.kill_rate || 0)));
    mount(this.metricsRow2,
      metric('Ø EPISODE REWARD', fmt.fixed(metrics.avg_reward || 0, 2)),
      metric('Ø TTK', `${fmt.fixed(metrics.avg_ttk || 0, 2)}s`),
      metric('ACCURACY', fmt.percent(metrics.accuracy || 0)),
      metric('HEADSHOT-ANTEIL', fmt.percent(metrics.headshot_pct || 0)));

    const logs = (snapshot.logs || []).slice(-40).join('\n') || 'Warte auf den ersten PPO-Callback…';
    setText(this.logNode, logs);
    this.logNode.scrollTop = this.logNode.scrollHeight;
    mount(this.errorHost, snapshot.error ? errorBox(snapshot.error) : null);
  },

  unmount() {
    this.poller?.stop();
  },
};

function warnChip(text) {
  return h('span', { class: 'chip', style: { borderColor: '#ffaa00', color: '#ffaa00' }, text });
}
