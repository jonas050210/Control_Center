// WebGL 3D renderer for arena, map lab, and mini-games.
//
// Replaces the Plotly figures the Streamlit app used to build server-side.
// The backend sends compact JSON in simulation coordinates (x/y ground plane,
// z up); this module maps it to Three.js space (x, z, y) with y up and keeps
// hundreds of cover props in a handful of instanced draw calls.

import { h, clear } from './dom.js';

let THREE = null;
let threeSource = '';

export async function loadThree() {
  if (THREE) return THREE;
  const attempts = [];
  try {
    // Preferred: the import map in index.html points at web/vendor/three.module.min.js.
    THREE = await import('three');
    threeSource = 'vendor (importmap)';
    return THREE;
  } catch (error) {
    attempts.push(`importmap: ${error.message}`);
  }
  try {
    // Works even without an import map (older browsers, unusual base paths).
    THREE = await import('../vendor/three.module.min.js');
    threeSource = 'vendor (relativ)';
    return THREE;
  } catch (error) {
    attempts.push(`vendor: ${error.message}`);
  }
  const cdn = 'https://unpkg.com/three@0.160.1/build/three.module.min.js';
  try {
    THREE = await import(/* webpackIgnore: true */ cdn);
    threeSource = 'CDN';
    return THREE;
  } catch (error) {
    attempts.push(`CDN: ${error.message}`);
  }
  throw new Error(
    'Three.js ist nicht verfügbar. Bitte install.py ausführen (lädt web/vendor/three.module.min.js) '
    + `oder online gehen. [${attempts.join(' | ')}]`,
  );
}

export function threeInfo() {
  return threeSource;
}

const GRID_COLOR = 0x233c30;

function clamp(value, lower, upper) {
  return Math.min(upper, Math.max(lower, value));
}

function lerpAngle(current, target, factor) {
  let delta = (target - current) % (Math.PI * 2);
  if (delta > Math.PI) delta -= Math.PI * 2;
  if (delta < -Math.PI) delta += Math.PI * 2;
  return current + delta * factor;
}

export class Arena3D {
  constructor(container, { height = 520, interactive = true, three = null, rendererFactory = null } = {}) {
    this.container = container;
    this.height = height;
    this.interactive = interactive;
    // Injectable for headless tests: a preloaded Three.js module and a renderer stub.
    this.injectedThree = three;
    this.rendererFactory = rendererFactory;
    this.ready = false;
    this.disposed = false;
    this.staticPayload = null;
    this.frame = null;
    this.overlay = null;
    this.cameraState = { theta: 0.7, phi: 1.06, radius: 34, target: [0, 1.2, 0] };
    this.agentVisuals = [];
    this.smoothing = 0.24;
    this.viewport = h('div', { class: 'viewport', style: { '--viewport-height': `${height}px` } });
    this.badge = h('div', { class: 'badge', text: 'Maus: drehen · Rad: zoomen · Shift+Ziehen: verschieben' });
    this.overlayNode = h('div', { class: 'overlay' });
    this.viewport.appendChild(this.overlayNode);
    this.viewport.appendChild(this.badge);
    container.appendChild(this.viewport);
  }

  async init() {
    try {
      const T = this.injectedThree || await loadThree();
      this.T = T;
      this.renderer = this.rendererFactory
        ? this.rendererFactory(T)
        : new T.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.renderer.domElement.style.width = '100%';
      this.renderer.domElement.style.height = '100%';
      this.viewport.insertBefore(this.renderer.domElement, this.overlayNode);

      this.scene = new T.Scene();
      this.scene.background = new T.Color(0x070b08);
      this.scene.fog = new T.FogExp2(0x070b08, 0.0075);

      this.camera = new T.PerspectiveCamera(52, 1, 0.1, 800);
      const hemisphere = new T.HemisphereLight(0x9dffb8, 0x101a12, 1.05);
      const key = new T.DirectionalLight(0xe8fff0, 0.95);
      key.position.set(14, 24, 10);
      const fill = new T.DirectionalLight(0x66ff99, 0.35);
      fill.position.set(-16, 12, -14);
      this.scene.add(hemisphere, key, fill);

      this.root = new T.Group();
      this.staticGroup = new T.Group();
      this.dynamicGroup = new T.Group();
      this.overlayGroup = new T.Group();
      this.root.add(this.staticGroup, this.dynamicGroup, this.overlayGroup);
      this.scene.add(this.root);

      this._buildTrails();
      this._bindControls();
      if (typeof ResizeObserver !== 'undefined') {
        this.observer = new ResizeObserver(() => this._resize());
        this.observer.observe(this.viewport);
      }
      this._resize();
      this.renderer.setAnimationLoop(() => this._tick());
      this.ready = true;
      if (this.staticPayload) this.setStatic(this.staticPayload);
      if (this.frame) this.setFrame(this.frame);
      if (this.overlay) this.setOverlay(this.overlay);
    } catch (error) {
      clear(this.overlayNode);
      this.overlayNode.appendChild(h('div', { class: 'err-box', style: { margin: '14px' },
        text: `3D-Renderer Fehler: ${error.message}` }));
      this.badge.textContent = '3D nicht verfügbar';
      console.warn(error);
    }
    return this;
  }

  setBadge(text) {
    this.badge.textContent = text;
  }

  // ------------------------------------------------------------- static scene
  setStatic(payload) {
    this.staticPayload = payload;
    if (!this.ready) return;
    const T = this.T;
    this._disposeGroup(this.staticGroup);
    const width = Number(payload.width) || 40;
    const depth = Number(payload.depth) || 40;
    const height = Math.max(width, depth);

    const floor = new T.Mesh(
      new T.PlaneGeometry(width, depth),
      new T.MeshLambertMaterial({ color: new T.Color(payload.floor_color || '#101815') }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.08;
    this.staticGroup.add(floor);

    const gridSize = Math.max(2, Math.round(height / (payload.grid_step || 2)) * (payload.grid_step || 2));
    const grid = new T.GridHelper(gridSize, Math.max(2, Math.round(gridSize / (payload.grid_step || 2))), GRID_COLOR, GRID_COLOR);
    grid.material.transparent = true;
    grid.material.opacity = 0.42;
    grid.scale.set(width / gridSize, 1, depth / gridSize);
    grid.position.y = 0.004;
    this.staticGroup.add(grid);

    // Cover, set dressing, and props: one instanced mesh per kind + color.
    const buckets = new Map();
    for (const item of payload.objects || []) {
      const key = `${item.kind}|${item.color}|${item.detail ? 'd' : 's'}`;
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(item);
    }
    for (const [key, items] of buckets) {
      const kind = key.split('|')[0];
      const geometry = kind === 'barrel'
        ? new T.CylinderGeometry(0.5, 0.5, 1, 10)
        : new T.BoxGeometry(1, 1, 1);
      const material = new T.MeshLambertMaterial({
        color: new T.Color(items[0].color || '#66727c'),
        flatShading: true,
      });
      const mesh = new T.InstancedMesh(geometry, material, items.length);
      const matrix = new T.Matrix4();
      const quaternion = new T.Quaternion();
      const position = new T.Vector3();
      const scale = new T.Vector3();
      items.forEach((item, index) => {
        position.set(item.x, item.z + item.h / 2, item.y);
        quaternion.setFromEuler(new T.Euler(0, item.yaw || 0, 0));
        if (kind === 'barrel') {
          const radius = (item.w + item.d) / 2;
          scale.set(radius, item.h, radius);
        } else {
          scale.set(item.w, item.h, item.d);
        }
        matrix.compose(position, quaternion, scale);
        mesh.setMatrixAt(index, matrix);
      });
      mesh.instanceMatrix.needsUpdate = true;
      mesh.frustumCulled = false;
      this.staticGroup.add(mesh);
    }

    // Scattered floor detail (non-colliding decoration).
    const details = payload.details || {};
    const detailCount = (details.x || []).length;
    if (detailCount) {
      const geometry = new T.BoxGeometry(1, 1, 1);
      const material = new T.MeshLambertMaterial({ flatShading: true });
      const mesh = new T.InstancedMesh(geometry, material, detailCount);
      const matrix = new T.Matrix4();
      const quaternion = new T.Quaternion();
      const position = new T.Vector3();
      const scale = new T.Vector3();
      const color = new T.Color();
      for (let index = 0; index < detailCount; index += 1) {
        const size = Number(details.size[index]) || 0.5;
        position.set(details.x[index], Number(details.z[index]) || 0.04, details.y[index]);
        quaternion.setFromEuler(new T.Euler(0, index * 0.7, 0));
        scale.set(size, 0.06, size * 0.7);
        matrix.compose(position, quaternion, scale);
        mesh.setMatrixAt(index, matrix);
        mesh.setColorAt(index, color.set(details.color[index] || '#324d3b'));
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      mesh.frustumCulled = false;
      this.staticGroup.add(mesh);
    }

    // Range markings, perimeter strip, beacons, spawn pads.
    const theme = new T.Color(payload.theme || '#00ff41');
    const ringRadius = Number(payload.ring_radius) || 5;
    for (const factor of [1, 0.72, 0.45]) {
      const ring = new T.Mesh(
        new T.RingGeometry(ringRadius * factor - 0.06, ringRadius * factor + 0.06, 64),
        new T.MeshBasicMaterial({ color: theme, transparent: true, opacity: 0.14, side: T.DoubleSide }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.012;
      this.staticGroup.add(ring);
    }
    const stripPoints = [
      new T.Vector3(-width / 2, 0.02, -depth / 2),
      new T.Vector3(width / 2, 0.02, -depth / 2),
      new T.Vector3(width / 2, 0.02, depth / 2),
      new T.Vector3(-width / 2, 0.02, depth / 2),
    ];
    const strip = new T.LineLoop(
      new T.BufferGeometry().setFromPoints(stripPoints),
      new T.LineBasicMaterial({ color: theme, transparent: true, opacity: 0.55 }),
    );
    this.staticGroup.add(strip);

    for (const [x, z] of [[-width * 0.46, -depth * 0.46], [width * 0.46, -depth * 0.46],
      [width * 0.46, depth * 0.46], [-width * 0.46, depth * 0.46]]) {
      const beacon = new T.Mesh(
        new T.CylinderGeometry(0.1, 0.1, 2.4, 8),
        new T.MeshBasicMaterial({ color: theme, transparent: true, opacity: 0.38 }),
      );
      beacon.position.set(x, 1.2, z);
      this.staticGroup.add(beacon);
    }

    for (const [x, y] of payload.spawns || []) {
      const pad = new T.Mesh(
        new T.RingGeometry(0.45, 0.62, 24),
        new T.MeshBasicMaterial({ color: 0x00aaff, transparent: true, opacity: 0.85, side: T.DoubleSide }),
      );
      pad.rotation.x = -Math.PI / 2;
      pad.position.set(x, 0.02, y);
      this.staticGroup.add(pad);
    }

    this._buildOverlayGeometry(payload);

    const topObject = Math.max(4, ...(payload.objects || []).map((item) => item.z + item.h));
    this.cameraState.target = [0, Math.min(topObject * 0.5, 6), 0];
    this.cameraState.radius = Math.max(width, depth) * (payload.camera_radius ?? 1.1);
    if (payload.camera_target) this.cameraState.target = [...payload.camera_target];
    if (payload.camera_theta !== undefined) this.cameraState.theta = payload.camera_theta;
    if (payload.camera_phi !== undefined) this.cameraState.phi = payload.camera_phi;
    this._applyCamera();
  }

  _buildOverlayGeometry(payload) {
    const T = this.T;
    this._disposeGroup(this.overlayGroup);
    this.overlayMeshes = {};

    if (payload.targets) {
      const discs = [];
      for (const target of payload.targets) {
        const ring = new T.Mesh(
          new T.RingGeometry(0.52, 0.66, 28),
          new T.MeshBasicMaterial({ color: 0x00ff41, transparent: true, opacity: 0.55, side: T.DoubleSide }),
        );
        const disc = new T.Mesh(
          new T.CircleGeometry(0.5, 28),
          new T.MeshBasicMaterial({ color: 0x0f2f18, side: T.DoubleSide }),
        );
        for (const mesh of [ring, disc]) {
          mesh.position.set(target.x, target.z, target.y);
          mesh.rotation.x = Math.PI / 2;
          this.overlayGroup.add(mesh);
        }
        discs.push({ ring, disc, target });
      }
      this.overlayMeshes.targets = discs;
    }

    if (payload.cells) {
      const spacing = Number(payload.spacing) || 3.4;
      const tiles = [];
      const geometry = new T.BoxGeometry(spacing * 0.92, 0.12, spacing * 0.92);
      for (const cell of payload.cells) {
        const tile = new T.Mesh(geometry, new T.MeshLambertMaterial({ color: 0x14351d, flatShading: true }));
        tile.position.set(cell.x, 0.06, cell.y);
        this.overlayGroup.add(tile);
        tiles.push(tile);
      }
      const player = new T.Mesh(
        new T.ConeGeometry(0.85, 1.9, 14),
        new T.MeshLambertMaterial({ color: 0x00ff41, flatShading: true }),
      );
      player.position.set(0, 1.05, 0);
      this.overlayGroup.add(player);
      const projectileGeometry = new T.SphereGeometry(0.42, 12, 10);
      const projectileMaterial = new T.MeshBasicMaterial({ color: 0xff0040 });
      const projectiles = new T.InstancedMesh(projectileGeometry, projectileMaterial, 40);
      projectiles.count = 0;
      projectiles.frustumCulled = false;
      this.overlayGroup.add(projectiles);

      for (const turret of payload.turrets || []) {
        const barrel = new T.Mesh(
          new T.CylinderGeometry(0.24, 0.34, 1.4, 10),
          new T.MeshLambertMaterial({ color: 0x8fa39a, flatShading: true }),
        );
        barrel.position.set(turret.x, 0.7, turret.y);
        this.overlayGroup.add(barrel);
      }
      this.overlayMeshes.dodge = { tiles, player, projectiles, spacing, cells: payload.cells };
    }
  }

  // ----------------------------------------------------------- dynamic scene
  setFrame(frame) {
    this.frame = frame;
    if (!this.ready) return;
    const agents = frame.agents || [];
    while (this.agentVisuals.length < agents.length) {
      this.agentVisuals.push(this._createAgent(this.agentVisuals.length));
    }
    agents.forEach((agent, index) => {
      const visual = this.agentVisuals[index];
      visual.target = agent;
      visual.group.visible = true;
      try {
        this._updateAgentLabel(visual, agent);
      } catch (error) {
        console.warn('Label-Update fehlgeschlagen', error);
      }
    });
    for (let index = agents.length; index < this.agentVisuals.length; index += 1) {
      this.agentVisuals[index].group.visible = false;
    }
    this._updateTrails(frame.trails || []);
  }

  setOverlay(overlay) {
    this.overlay = overlay;
    if (!this.ready || !overlay || !this.overlayMeshes) return;
    if (overlay.type === 'aim' && this.overlayMeshes.targets) {
      const states = overlay.states || [];
      this.overlayMeshes.targets.forEach((entry, index) => {
        const state = states[index] || {};
        const active = Boolean(state.active);
        entry.ring.material.color.set(active ? '#ffaa00' : '#00ff41');
        entry.ring.material.opacity = active ? 0.95 : 0.4;
        entry.disc.material.color.set(active ? '#5a2d00' : (state.hit ? '#123f1c' : '#0f2f18'));
        entry.disc.scale.setScalar(active ? 1.16 : 1);
      });
    }
    if (overlay.type === 'dodge' && this.overlayMeshes.dodge) {
      const { tiles, player, projectiles, spacing, cells } = this.overlayMeshes.dodge;
      const T = this.T;
      const grid = Math.round(Math.sqrt(cells.length)) || 5;
      const half = (grid - 1) / 2;
      const color = new T.Color();
      const active = Boolean(overlay.game && overlay.game.active);
      tiles.forEach((tile, index) => {
        const cell = cells[index];
        const occupied = (overlay.game?.projectiles || []).some(
          ([px, py]) => px === cell.column && py === cell.row,
        );
        const isPlayer = overlay.game && overlay.game.player
          && overlay.game.player[0] === cell.column && overlay.game.player[1] === cell.row;
        color.set(occupied ? '#5a0d1c' : isPlayer ? '#1c5f2c' : '#14351d');
        tile.material.color.copy(color);
        tile.material.emissive?.setRGB?.(0, 0, 0);
      });
      if (overlay.game && overlay.game.player) {
        const [column, row] = overlay.game.player;
        player.position.set((column - half) * spacing, 1.05, (row - half) * spacing);
        player.visible = active;
      }
      const list = (overlay.game?.projectiles || []).slice(0, 40);
      const matrix = new T.Matrix4();
      const position = new T.Vector3();
      const quaternion = new T.Quaternion();
      const scale = new T.Vector3(1, 1, 1);
      list.forEach(([column, row], index) => {
        position.set((column - half) * spacing, 0.7, (row - half) * spacing);
        matrix.compose(position, quaternion, scale);
        projectiles.setMatrixAt(index, matrix);
      });
      projectiles.count = list.length;
      projectiles.instanceMatrix.needsUpdate = true;
    }
  }

  _createAgent(index) {
    const T = this.T;
    const color = index === 0 ? 0x00cc33 : 0xff0040;
    const group = new T.Group();
    const body = new T.Mesh(
      new T.CapsuleGeometry(0.32, 1.2, 4, 12),
      new T.MeshLambertMaterial({ color, flatShading: true }),
    );
    body.position.y = 0.92;
    const head = new T.Mesh(
      new T.SphereGeometry(0.2, 14, 10),
      new T.MeshLambertMaterial({ color, flatShading: true }),
    );
    head.position.y = 1.62;
    const gunPivot = new T.Group();
    gunPivot.position.y = 1.24;
    const gun = new T.Mesh(
      new T.BoxGeometry(0.1, 0.14, 0.9),
      new T.MeshLambertMaterial({ color: 0xdfe7e2, flatShading: true }),
    );
    gun.position.set(0.24, 0, 0.34);
    gunPivot.add(gun);
    const healthBar = new T.Mesh(
      new T.PlaneGeometry(1.5, 0.14),
      new T.MeshBasicMaterial({ color, transparent: true, opacity: 0.95, depthTest: false }),
    );
    const label = new T.Sprite(new T.SpriteMaterial({ transparent: true, depthTest: false }));
    label.scale.set(3.4, 0.62, 1);
    group.add(body, head, gunPivot, healthBar, label);
    this.dynamicGroup.add(group);
    return { group, body, head, gunPivot, healthBar, label, target: null, labelText: '', state: null };
  }

  _updateAgentLabel(visual, agent) {
    const text = `${agent.label} · ${agent.weapon}`;
    if (visual.labelText === text) return;
    visual.labelText = text;
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      // Canvas 2D can be unavailable (privacy modes, headless tests); the
      // sprite simply stays empty instead of breaking the render loop.
      visual.labelText = text;
      visual.label.visible = false;
      return;
    }
    visual.label.visible = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgba(7, 11, 8, 0.72)';
    ctx.fillRect(0, 24, canvas.width, 48);
    ctx.strokeStyle = agent.color;
    ctx.lineWidth = 3;
    ctx.strokeRect(1, 25, canvas.width - 2, 46);
    ctx.fillStyle = agent.color;
    ctx.font = 'bold 34px "Cascadia Code", Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, canvas.width / 2, canvas.height / 2 + 2);
    const texture = this.T ? new this.T.CanvasTexture(canvas) : null;
    if (!texture) return;
    if (visual.label.material.map) visual.label.material.map.dispose();
    visual.label.material.map = texture;
    visual.label.material.needsUpdate = true;
  }

  _buildTrails() {
    const T = this.T;
    this.trailCapacity = 40;
    this.trailPositions = new Float32Array(this.trailCapacity * 2 * 3);
    this.trailColors = new Float32Array(this.trailCapacity * 2 * 3);
    const geometry = new T.BufferGeometry();
    geometry.setAttribute('position', new T.BufferAttribute(this.trailPositions, 3).setUsage(T.DynamicDrawUsage));
    geometry.setAttribute('color', new T.BufferAttribute(this.trailColors, 3).setUsage(T.DynamicDrawUsage));
    geometry.setDrawRange(0, 0);
    this.trails = new T.LineSegments(geometry, new T.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.85,
    }));
    this.trails.frustumCulled = false;
    this.dynamicGroup.add(this.trails);
  }

  _updateTrails(trails) {
    const T = this.T;
    const miss = new T.Color(0x8ab6cf);
    const hit = new T.Color(0xffaa00);
    const count = Math.min(this.trailCapacity, trails.length);
    const slice = trails.slice(-count);
    slice.forEach((trail, index) => {
      const [sx, sy, sz] = trail.start;
      const [ex, ey, ez] = trail.end;
      const offset = index * 6;
      this.trailPositions[offset + 0] = sx;
      this.trailPositions[offset + 1] = sz;
      this.trailPositions[offset + 2] = sy;
      this.trailPositions[offset + 3] = ex;
      this.trailPositions[offset + 4] = ez;
      this.trailPositions[offset + 5] = ey;
      const color = trail.hit ? hit : miss;
      for (let vertex = 0; vertex < 2; vertex += 1) {
        this.trailColors[offset + vertex * 3 + 0] = color.r;
        this.trailColors[offset + vertex * 3 + 1] = color.g;
        this.trailColors[offset + vertex * 3 + 2] = color.b;
      }
    });
    const geometry = this.trails.geometry;
    geometry.setDrawRange(0, count * 2);
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
  }

  // ------------------------------------------------------------ camera + loop
  _bindControls() {
    if (!this.interactive) return;
    const element = this.renderer.domElement;
    let dragging = false;
    let panning = false;
    let lastX = 0;
    let lastY = 0;
    element.style.touchAction = 'none';
    element.addEventListener('contextmenu', (event) => event.preventDefault());
    element.addEventListener('pointerdown', (event) => {
      dragging = true;
      panning = event.shiftKey || event.button === 1 || event.button === 2;
      lastX = event.clientX;
      lastY = event.clientY;
      element.setPointerCapture(event.pointerId);
    });
    element.addEventListener('pointerup', (event) => {
      dragging = false;
      panning = false;
      if (element.hasPointerCapture(event.pointerId)) element.releasePointerCapture(event.pointerId);
    });
    element.addEventListener('pointermove', (event) => {
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      if (panning) {
        const scale = this.cameraState.radius * 0.0016;
        const theta = this.cameraState.theta;
        this.cameraState.target[0] -= (Math.cos(theta) * dx - Math.sin(theta) * dy) * scale;
        this.cameraState.target[2] -= (Math.sin(theta) * dx + Math.cos(theta) * dy) * scale;
      } else {
        this.cameraState.theta -= dx * 0.008;
        this.cameraState.phi = clamp(this.cameraState.phi - dy * 0.006, 0.12, 1.48);
      }
      this._applyCamera();
    });
    element.addEventListener('wheel', (event) => {
      event.preventDefault();
      this.cameraState.radius = clamp(
        this.cameraState.radius * Math.exp(event.deltaY * 0.0012), 4, 400,
      );
      this._applyCamera();
    }, { passive: false });
  }

  _applyCamera() {
    if (!this.ready) return;
    const { theta, phi, radius, target } = this.cameraState;
    const sinPhi = Math.sin(phi);
    this.camera.position.set(
      target[0] + radius * sinPhi * Math.sin(theta),
      target[1] + radius * Math.cos(phi),
      target[2] + radius * sinPhi * Math.cos(theta),
    );
    this.camera.lookAt(target[0], target[1], target[2]);
  }

  _resize() {
    if (!this.ready) return;
    const width = Math.max(240, this.viewport.clientWidth);
    const height = Math.max(200, this.viewport.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  _tick() {
    if (!this.ready || this.disposed) return;
    const factor = this.smoothing;
    for (const visual of this.agentVisuals) {
      const target = visual.target;
      if (!target) continue;
      if (!visual.state) {
        visual.state = { ...target };
      }
      const state = visual.state;
      state.x += (target.x - state.x) * factor;
      state.y += (target.y - state.y) * factor;
      state.z += (target.z - state.z) * factor;
      state.height += (target.height - state.height) * factor;
      state.yaw = lerpAngle(state.yaw, target.yaw, factor);
      state.pitch += (target.pitch - state.pitch) * factor;
      state.hp += (target.hp - state.hp) * factor;

      visual.group.position.set(state.x, state.z, state.y);
      visual.group.rotation.y = state.yaw;
      const height = Math.max(0.45, state.height);
      visual.body.scale.y = height / 1.84;
      visual.body.position.y = height / 2;
      visual.head.position.y = Math.max(0.34, height - 0.16);
      visual.gunPivot.position.y = height * 0.72;
      visual.gunPivot.rotation.x = -state.pitch;
      const alive = target.alive !== false;
      visual.body.material.color.set(alive ? (target.color || '#00ff41') : '#3a3a3a');
      visual.head.material.color.copy(visual.body.material.color);
      visual.healthBar.position.set(0, height + 0.65, 0);
      visual.healthBar.quaternion.copy(this.camera.quaternion);
      visual.healthBar.material.color.set(state.hp > 50 ? '#00ff41' : state.hp > 25 ? '#ffaa00' : '#ff0040');
      const fraction = clamp(state.hp / 100, 0.02, 1);
      visual.healthBar.scale.x = fraction;
      visual.healthBar.position.x = -(1 - fraction) * 0.75;
      visual.label.position.set(0, height + 1.15, 0);
      visual.label.quaternion.copy(this.camera.quaternion);
      visual.label.material.opacity = alive ? 1 : 0.35;
    }
    this.renderer.render(this.scene, this.camera);
  }

  _disposeGroup(group) {
    for (const child of [...group.children]) {
      group.remove(child);
      child.traverse?.((node) => {
        if (node.geometry) node.geometry.dispose();
        if (node.material) {
          const materials = Array.isArray(node.material) ? node.material : [node.material];
          for (const material of materials) {
            if (material.map) material.map.dispose();
            material.dispose();
          }
        }
      });
    }
  }

  dispose() {
    this.disposed = true;
    if (this.observer) this.observer.disconnect();
    if (this.renderer) {
      this.renderer.setAnimationLoop(null);
      this._disposeGroup(this.staticGroup);
      this._disposeGroup(this.dynamicGroup);
      this._disposeGroup(this.overlayGroup);
      this.renderer.dispose();
      if (this.renderer.domElement.parentElement) {
        this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
      }
    }
    if (this.viewport.parentElement) this.viewport.parentElement.removeChild(this.viewport);
  }
}

export function createScene(container, options) {
  const scene = new Arena3D(container, options);
  scene.initPromise = scene.init();
  return scene;
}
