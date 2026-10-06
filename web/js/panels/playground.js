// PLAYGROUND panel: human-vs-bot shooter, aim trainer, and dodge survival.

import { api } from '../api.js';
import { store } from '../store.js';
import { createScene } from '../scene.js';
import {
  button, card, checkboxField, clear, feed, field, fmt, h, metric, mount, toast, warnBox,
} from '../dom.js';

const MODES = [
  { id: 'shooter', label: '🔫 Real Shooter' },
  { id: 'aim', label: '🎯 Aim Trainer' },
  { id: 'dodge', label: '⚡ Dodge Survival' },
];

const STANCES = ['Stand', 'Crouch', 'Prone'];

export const playgroundPanel = {
  id: 'playground',
  label: '🔫 PLAYGROUND',

  mount(root) {
    this.meta = store.meta;
    this.mode = store.playground.mode || 'shooter';
    this.detail = 'Balanced';
    this.stance = 0;
    this.sprint = false;
    this.recording = false;
    this.shooter = null;
    this.aim = null;
    this.dodge = null;
    this.timer = null;

    this.modeSelect = h('select');
    for (const mode of MODES) this.modeSelect.appendChild(h('option', { value: mode.id, text: mode.label }));
    this.modeSelect.value = this.mode;
    this.modeSelect.addEventListener('change', () => {
      this.mode = this.modeSelect.value;
      store.playground.mode = this.mode;
      this.buildMode();
    });
    this.detailSelect = h('select');
    for (const name of Object.keys(this.meta.detail_presets)) {
      this.detailSelect.appendChild(h('option', { value: name, text: name }));
    }
    this.detailSelect.value = this.detail;
    this.detailSelect.addEventListener('change', () => {
      this.detail = this.detailSelect.value;
      if (this.mode === 'shooter') this.applyShooterConfig();
    });

    this.body = h('div');

    mount(root,
      h('h2', { text: '🕹️ PLAYGROUND · MENSCH GEGEN KI + DRILLS' }),
      h('p', { class: 'hint', text: 'Headless-Simulation im Server, 3D-Darstellung im Browser. Tastatur: WASD bewegen, J/L drehen, I/K zielen, Leertaste feuern, R nachladen, X sprinten, C crouchen, P hinlegen, Q beenden.' }),
      card('MODUS',
        h('div', { class: 'row' },
          field('PLAY MODE', this.modeSelect),
          field('3D DETAIL', this.detailSelect))),
      this.body);

    this.bindKeys();
    this.buildMode();
  },

  onShow() {
    this.active = true;
    if (this.mode === 'aim') this.startAimPolling();
  },

  onHide() {
    this.active = false;
    this.stopPolling();
  },

  unmount() {
    this.onHide();
    this.scene?.dispose();
  },

  // ------------------------------------------------------------------ keybind
  bindKeys() {
    this.keyHandler = (event) => {
      if (!this.active || this.mode !== 'shooter' || !this.shooter) return;
      const tag = (event.target.tagName || '').toLowerCase();
      if (['input', 'select', 'textarea'].includes(tag)) return;
      const map = {
        w: 'forward', s: 'back', a: 'strafe_left', d: 'strafe_right',
        j: 'turn_left', l: 'turn_right', i: 'look_up', k: 'look_down',
        ' ': 'fire', r: 'reload', e: 'jump', x: 'sprint_forward',
      };
      if (event.key === 'q') {
        this.modeSelect.value = 'dodge';
        this.modeSelect.dispatchEvent(new Event('change'));
        return;
      }
      const press = map[event.key.toLowerCase()];
      if (!press) return;
      event.preventDefault();
      this.shooter.send(press);
    };
    window.addEventListener('keydown', this.keyHandler);
  },

  // -------------------------------------------------------------------- modes
  buildMode() {
    this.stopPolling();
    this.scene?.dispose();
    this.scene = null;
    clear(this.body);
    if (this.mode === 'shooter') this.buildShooter();
    else if (this.mode === 'aim') this.buildAim();
    else this.buildDodge();
  },

  // ----------------------------------------------------------- real shooter
  buildShooter() {
    const meta = this.meta;
    this.shooter = {
      map: 'Dust',
      weapon: meta.weapons[0],
      enemyWeapon: meta.weapons[2] || meta.weapons[1],
      bot: meta.bot_behaviors[0],
      configured: false,
      sceneKey: null,
      reward: 0,
      result: null,
      demoCount: 0,
      busy: false,
    };

    const selects = {};
    const makeSelect = (key, values) => {
      const node = h('select');
      for (const value of values) node.appendChild(h('option', { value, text: value }));
      node.value = this.shooter[key];
      node.addEventListener('change', () => {
        this.shooter[key] = node.value;
        this.applyShooterConfig();
      });
      selects[key] = node;
      return node;
    };
    const botOptions = [...meta.bot_behaviors, ...meta.models.filter((name) => name !== meta.heuristic)];

    this.shooterTelemetry = h('div', { class: 'grid cols-2' });
    this.shooterResult = h('div');
    this.shooterFeed = h('div', { class: 'feed' });
    this.shooterSceneHost = h('div', { class: 'grow', style: { flexBasis: '620px', minWidth: '320px' } });
    this.demoInfo = h('div', { class: 'hint' });
    this.warnHost = h('div');

    const padMove = h('div', { class: 'pad-grid' },
      h('div'), button('▲ Forward', () => this.press('forward'), { className: 'pad' }), h('div'),
      button('◀ Strafe', () => this.press('strafe_left'), { className: 'pad' }),
      button('▼ Back', () => this.press('back'), { className: 'pad' }),
      button('Strafe ▶', () => this.press('strafe_right'), { className: 'pad' }),
      button('🏃 Sprint', () => this.press('sprint_forward'), { className: 'pad' }), h('div'), h('div'));

    const padLook = h('div', { class: 'pad-grid' },
      h('div'), button('⬆ Look', () => this.press('look_up'), { className: 'pad' }), h('div'),
      button('⟲ Turn', () => this.press('turn_left'), { className: 'pad' }),
      button('⬇ Look', () => this.press('look_down'), { className: 'pad' }),
      button('Turn ⟳', () => this.press('turn_right'), { className: 'pad' }));

    const padCombat = h('div', { class: 'pad-grid' },
      button('🔴 FIRE', () => this.press('fire'), { className: 'pad', style: { gridColumn: 'span 3' } }),
      button('↻ Reload', () => this.press('reload'), { className: 'pad' }),
      button('↗ Move+Fire', () => this.press('move_fire'), { className: 'pad' }),
      button('⬆ Jump', () => this.press('jump'), { className: 'pad' }));

    const stanceSelect = h('select');
    for (const name of STANCES) stanceSelect.appendChild(h('option', { value: name, text: name }));
    stanceSelect.addEventListener('change', () => {
      this.stance = STANCES.indexOf(stanceSelect.value);
    });

    this.recordToggle = checkboxField('Imitation-Demos aufzeichnen', this.recording, async (checked) => {
      this.recording = checked;
      try {
        await api.playgroundRecording({ recording: checked });
        toast(checked ? 'Aufzeichnung aktiv.' : 'Aufzeichnung gestoppt.');
      } catch (error) {
        toast(error.message, 'error');
      }
    });
    this.sprintToggle = checkboxField('Sprint', this.sprint, (checked) => { this.sprint = checked; });

    mount(this.body,
      card('MATCH SETUP',
        h('div', { class: 'row' },
          field('MAP', makeSelect('map', meta.maps)),
          field('YOUR WEAPON', makeSelect('weapon', meta.weapons)),
          field('BOT WEAPON', makeSelect('enemyWeapon', meta.weapons))),
        h('div', { class: 'row', style: { marginTop: '10px' } },
          field('BOT BEHAVIOR / PPO MODEL', makeSelect('bot', botOptions)),
          field('STANCE', stanceSelect),
          h('div', { class: 'grow' }, this.recordToggle, this.sprintToggle)),
        h('div', { class: 'row', style: { marginTop: '10px' } },
          button('🔄 New Round', () => this.resetShooter()),
          button('⬇ Demo-CSV laden', () => window.open(api.demosUrl, '_blank'), { className: 'ghost' }),
          button('💾 Demos in data/demos.csv speichern', () => this.saveDemos(), { className: 'ghost' }))),
      this.warnHost,
      h('hr', { class: 'sep' }),
      h('div', { class: 'row', style: { alignItems: 'flex-start' } },
        this.shooterSceneHost,
        h('div', { class: 'grow', style: { flexBasis: '360px' } },
          card('TELEMETRIE', this.shooterTelemetry, this.shooterResult),
          h('div', { style: { height: '12px' } }),
          card('STEUERUNG',
            h('div', { class: 'row' }, h('div', { class: 'grow' }, h('h4', { text: 'MOVE' }), padMove),
              h('div', { class: 'grow' }, h('h4', { text: 'LOOK' }), padLook)),
            h('h4', { text: 'COMBAT' }), padCombat),
          h('div', { style: { height: '12px' } }),
          card('KAMPF-FEED', this.shooterFeed),
          h('div', { style: { height: '12px' } }),
          card('IMITATION-DATEN', this.demoInfo))));

    this.scene = createScene(this.shooterSceneHost, { height: 540 });
    this.applyShooterConfig();
  },

  async applyShooterConfig() {
    const state = this.shooter;
    try {
      const response = await api.playgroundConfig({
        map: state.map,
        weapon: state.weapon,
        enemy_weapon: state.enemyWeapon,
        bot: state.bot,
        detail: this.detail,
      });
      state.configured = true;
      state.sceneKey = response.scene_key;
      state.reward = 0;
      state.result = null;
      this.scene.setStatic(response.static);
      this.scene.setFrame(response.frame);
      this.renderShooterTelemetry(response.frame, 0, null);
      this.renderShooterWarnings([]);
      state.demoCount = response.demo_count;
      this.renderDemoInfo();
      feed(this.shooterFeed, []);
    } catch (error) {
      this.renderShooterWarnings([error.message]);
    }
  },

  async press(press) {
    const state = this.shooter;
    if (!state || !state.configured || state.busy) return;
    state.busy = true;
    try {
      const response = await api.playgroundAction({
        press,
        stance: this.stance,
        sprint: this.sprint,
        recording: this.recording,
      });
      state.reward = response.reward;
      state.result = response.result;
      state.demoCount = response.demo_count;
      state.done = response.frame.done;
      this.scene.setFrame(response.frame);
      this.renderShooterTelemetry(response.frame, response.reward, response.result);
      this.appendMessages(response.new_messages);
      this.renderShooterWarnings(response.load_errors);
      this.renderDemoInfo();
    } catch (error) {
      toast(error.message, 'error');
    } finally {
      state.busy = false;
    }
  },

  async resetShooter() {
    try {
      const response = await api.playgroundReset();
      this.shooter.reward = 0;
      this.shooter.result = null;
      this.scene.setFrame(response.frame);
      this.renderShooterTelemetry(response.frame, 0, null);
      feed(this.shooterFeed, []);
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  appendMessages(messages) {
    this.messages = [...(this.messages || []), ...(messages || [])].slice(-40);
    feed(this.shooterFeed, this.messages, { limit: 10 });
  },

  renderShooterTelemetry(frame, reward, result) {
    const [player, bot] = frame.agents || [];
    mount(this.shooterTelemetry,
      player ? metric('YOUR HEALTH', `${player.hp.toFixed(0)} HP`, `${player.ammo}/${player.mag_size} ${player.weapon}`) : metric('YOUR HEALTH', '—'),
      bot ? metric('BOT HEALTH', `${bot.hp.toFixed(0)} HP`, `${bot.ammo}/${bot.mag_size} ${bot.weapon}`) : metric('BOT HEALTH', '—'));
    const accuracy = player ? player.hits / Math.max(1, player.bullets_fired || 1) : 0;
    mount(this.shooterResult,
      h('div', { class: 'legend' },
        h('span', { class: 'chip', text: `ROUND ${fmt.fixed(frame.elapsed, 1)}s` }),
        h('span', { class: 'chip', text: `REWARD ${reward >= 0 ? '+' : ''}${fmt.fixed(reward, 2)}` }),
        h('span', { class: 'chip', text: `ACC ${fmt.percent(accuracy)}` }),
        h('span', { class: 'chip', text: `SHOTS ${player ? player.shots_fired : 0}` })),
      result
        ? h('div', {
          class: result.win ? 'ok-box' : 'err-box',
          text: `ROUND RESULT · ${result.win ? 'VICTORY' : (result.draw ? 'DRAW' : 'DEFEAT')} · TTK ${fmt.fixed(result.ttk, 1)}s`,
        })
        : null);
  },

  renderShooterWarnings(errors) {
    const unique = [...new Set(errors || [])].slice(0, 3);
    mount(this.warnHost, ...unique.map((message) => warnBox(message)));
  },

  renderDemoInfo() {
    mount(this.demoInfo,
      h('span', { class: 'chip', text: `${fmt.num(this.shooter.demoCount)} / 25.000 Samples` }),
      h('span', { class: 'hint', text: 'Aufzeichnung aktivieren, dann bewegen/feuern: jeder Policy-Schritt wird als (state, action) gespeichert.' }));
  },

  async saveDemos() {
    try {
      const response = await api.saveDemos();
      toast(response.saved ? `${response.saved} Demos in ${response.path} gespeichert.` : 'Keine neuen Demos zum Speichern.', response.saved ? 'info' : 'warn');
      this.shooter.demoCount = 0;
      this.renderDemoInfo();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  // -------------------------------------------------------------- aim trainer
  buildAim() {
    this.aim = { duration: 30, game: null, best: 0, hits: 0, misses: 0, lastResult: null };
    this.aimSceneHost = h('div', { class: 'grow', style: { flexBasis: '640px', minWidth: '320px' } });
    this.aimMetrics = h('div', { class: 'grid cols-4' });
    this.aimGrid = h('div');
    this.aimStatus = h('div', { class: 'hint' });

    const durationSelect = h('select');
    for (const value of [15, 30, 60]) durationSelect.appendChild(h('option', { value, text: `${value} Sekunden` }));
    durationSelect.value = '30';
    durationSelect.addEventListener('change', () => { this.aim.duration = Number(durationSelect.value); });

    mount(this.body,
      h('h3', { text: '🎯 AIM TRAINER · 3D PRECISION RANGE' }),
      h('p', { class: 'hint', text: 'Triff das aktive Bullseye in der 3D-Schießbahn. Treffer, Quote und Reaktionszeit werden live gemessen – die 9 Tore sind auch direkt im 3D-Bild hervorgehoben.' }),
      card('DRILL',
        h('div', { class: 'row' },
          field('RUNDE', durationSelect),
          button('▶ Start / Restart Drill', () => this.startAim()),
          h('div', { class: 'grow' }, this.aimStatus))),
      h('hr', { class: 'sep' }),
      h('div', { class: 'row', style: { alignItems: 'flex-start' } },
        this.aimSceneHost,
        h('div', { class: 'grow', style: { flexBasis: '340px' } },
          card('MESSWERTE', this.aimMetrics), h('div', { style: { height: '12px' } }),
          card('TARGET GRID', this.aimGrid))));

    this.scene = createScene(this.aimSceneHost, { height: 540 });
    this.loadAimScene();
  },

  async loadAimScene() {
    try {
      const response = await api.aimScene();
      this.scene.setStatic(response.static);
      this.aim.best = response.best;
      if (response.game) {
        this.aim.game = response.game;
        this.renderAim();
      } else {
        this.renderAim();
      }
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async startAim() {
    try {
      const response = await api.aimStart(this.aim.duration);
      this.aim.game = response.game;
      this.aim.best = response.best;
      this.renderAim();
      this.startAimPolling();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async tapCell(cell) {
    if (!this.aim.game || !this.aim.game.active) return;
    try {
      const response = await api.aimTap(cell);
      this.aim.game = response.game;
      this.aim.best = response.best;
      this.renderAim();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  startAimPolling() {
    this.stopPolling();
    this.timer = setInterval(async () => {
      if (!this.active) return;
      try {
        const response = await api.aimState();
        this.aim.game = response.game;
        this.aim.best = response.best;
        this.renderAim();
      } catch (error) {
        console.warn(error);
      }
    }, 900);
  },

  stopPolling() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  },

  renderAim() {
    const game = this.aim.game;
    const states = [];
    for (let index = 0; index < 9; index += 1) {
      states.push({ cell: index, active: Boolean(game && game.active && game.target === index) });
    }
    this.scene?.setOverlay({ type: 'aim', states });
    if (!game) {
      mount(this.aimMetrics, metric('STATUS', 'BEREIT'), metric('HITS', '0'), metric('ACCURACY', '—'), metric('BEST', String(this.aim.best)));
      mount(this.aimGrid, h('p', { class: 'hint', text: 'Drill starten, um die Ziele zu aktivieren.' }));
      mount(this.aimStatus, h('span', { class: 'chip', text: `SESSION BEST ${this.aim.best}` }));
      return;
    }
    mount(this.aimMetrics,
      metric('HITS', String(game.hits), game.active ? `${fmt.fixed(game.remaining, 1)}s verbleibend` : 'Runde beendet'),
      metric('ACCURACY', fmt.percent(game.accuracy), `${game.misses} Fehlschüsse`),
      metric('BEST STREAK', String(game.best_streak), `Streak ${game.streak}`),
      metric('REAKTION', game.avg_reaction_ms ? `${game.avg_reaction_ms.toFixed(0)} ms` : '—', `BEST ${this.aim.best} Hits`));

    const rows = [];
    for (let row = 0; row < 3; row += 1) {
      const cells = [];
      for (let column = 0; column < 3; column += 1) {
        const cell = row * 3 + column;
        const isTarget = game.active && game.target === cell;
        cells.push(button(isTarget ? `🎯 ${cell + 1}` : String(cell + 1).padStart(2, '0'),
          () => this.tapCell(cell), { className: `pad ${isTarget ? 'active' : ''}` }));
      }
      rows.push(h('div', { class: 'pad-grid', style: { marginBottom: '6px' } }, ...cells));
    }
    if (game.active) {
      rows.push(h('div', { class: 'bar', style: { marginTop: '10px' } },
        h('span', { style: { width: `${Math.max(0, Math.min(100, (game.remaining / game.duration) * 100))}%` } })));
    } else {
      rows.push(h('div', { class: 'ok-box', text: `DRILL COMPLETE · ${game.hits} Ziele · ${game.misses} Fehlschüsse · Best Streak ${game.best_streak}` }));
    }
    mount(this.aimGrid, ...rows);
    mount(this.aimStatus, h('span', { class: 'chip', text: `SESSION BEST ${this.aim.best} Hits` }));
  },

  // ----------------------------------------------------------- dodge survival
  buildDodge() {
    this.dodge = { game: null, best: 0, busy: false };
    this.dodgeSceneHost = h('div', { class: 'grow', style: { flexBasis: '620px', minWidth: '320px' } });
    this.dodgeMetrics = h('div', { class: 'grid cols-4' });
    this.dodgePad = h('div');
    this.dodgeStatus = h('div');

    mount(this.body,
      h('h3', { text: '⚡ DODGE SURVIVAL · PROJEKTILEN AUSWEICHEN' }),
      h('p', { class: 'hint', text: 'Bewege dich auf dem 5×5-Grid. Jeder D-Pad-Tap lässt eine Salve weiterlaufen; die Feuerlinien werden mit steigendem Score dichter.' }),
      card('RUN',
        h('div', { class: 'row' },
          button('▶ New Run', () => this.newDodgeRun()),
          h('div', { class: 'grow' }, this.dodgeStatus))),
      h('hr', { class: 'sep' }),
      h('div', { class: 'row', style: { alignItems: 'flex-start' } },
        this.dodgeSceneHost,
        h('div', { class: 'grow', style: { flexBasis: '340px' } },
          card('MESSWERTE', this.dodgeMetrics), h('div', { style: { height: '12px' } }),
          card('DODGE PAD', this.dodgePad))));

    this.scene = createScene(this.dodgeSceneHost, { height: 520 });
    this.loadDodgeScene();
  },

  async loadDodgeScene() {
    try {
      const response = await api.dodgeScene();
      this.scene.setStatic(response.static);
      this.dodge.best = response.best;
      this.dodge.game = response.game;
      this.renderDodge();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async newDodgeRun() {
    try {
      const response = await api.dodgeStart();
      this.dodge.game = response.game;
      this.dodge.best = response.best;
      this.renderDodge();
    } catch (error) {
      toast(error.message, 'error');
    }
  },

  async dodgeMove(dx, dy) {
    if (!this.dodge.game || !this.dodge.game.active || this.dodge.busy) return;
    this.dodge.busy = true;
    try {
      const response = await api.dodgeStep(dx, dy);
      this.dodge.game = response.game;
      this.dodge.best = response.best;
      this.renderDodge();
    } catch (error) {
      toast(error.message, 'error');
    } finally {
      this.dodge.busy = false;
    }
  },

  renderDodge() {
    const game = this.dodge.game;
    this.scene?.setOverlay({ type: 'dodge', game });
    mount(this.dodgeMetrics,
      metric('SCORE', String(game ? game.score : 0), `Tick ${game ? game.tick : 0}`),
      metric('SALVEN EVADED', String(game ? game.dodged : 0), 'Projektile passiert'),
      metric('PERSONAL BEST', String(this.dodge.best)),
      metric('STATUS', game && game.active ? 'AKTIV' : 'BEENDET', game && game.hit ? 'Getroffen' : ''));

    if (!game) {
      mount(this.dodgePad, h('p', { class: 'hint', text: 'Run starten – die ersten Salven sind harmlos, danach wird es dichter.' }));
      mount(this.dodgeStatus, null);
      return;
    }
    if (game.active) {
      mount(this.dodgePad,
        h('div', { class: 'pad-grid' }, h('div'), button('▲', () => this.dodgeMove(0, -1), { className: 'pad' }), h('div')),
        h('div', { class: 'pad-grid' },
          button('◀', () => this.dodgeMove(-1, 0), { className: 'pad' }),
          button('WAIT', () => this.dodgeMove(0, 0), { className: 'pad' }),
          button('▶', () => this.dodgeMove(1, 0), { className: 'pad' })),
        h('div', { class: 'pad-grid' }, h('div'), button('▼', () => this.dodgeMove(0, 1), { className: 'pad' }), h('div')));
      mount(this.dodgeStatus, h('span', { class: 'chip', text: 'Läuft - Taps zählen als Tick' }));
    } else {
      mount(this.dodgePad, h('p', { class: 'hint', text: 'Run beendet – New Run startet eine neue Salve.' }));
      mount(this.dodgeStatus, h('div', { class: 'err-box', text: `RUN OVER · Score ${game.score} · ${game.dodged} Salven evaded` }));
    }
  },
};
