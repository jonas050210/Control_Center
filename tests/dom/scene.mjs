// Headless 3D test: builds every map with the real three.js bundle and feeds
// synthetic arena frames through web/js/scene.js. No browser, no WebGL: the
// renderer is stubbed, the scene graph is the real thing.
//
//   cd tests/dom && npm install && node scene.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = process.env.ROOT || path.resolve(HERE, '..', '..');
const VENDOR = path.join(ROOT, 'web', 'vendor', 'three.module.min.js');
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const problems = [];
let failures = 0;
function check(label, condition, detail = '') {
  const ok = Boolean(condition);
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`);
  return ok;
}

if (!fs.existsSync(VENDOR)) {
  console.error(`three.js fehlt: ${VENDOR}\nBitte "python3 install.py" ausführen.`);
  process.exit(1);
}

// --- minimal DOM ------------------------------------------------------------
const dom = new JSDOM('<!doctype html><html><body><div id="host"></div></body></html>',
  { url: 'http://localhost/', pretendToBeVisual: true });
const { window } = dom;
global.window = window;
global.document = window.document;
Object.defineProperty(global, 'navigator', { value: window.navigator, configurable: true });
global.HTMLElement = window.HTMLElement;
global.Node = window.Node;
global.getComputedStyle = window.getComputedStyle.bind(window);
global.requestAnimationFrame = (callback) => setTimeout(() => callback(Date.now()), 16);
global.cancelAnimationFrame = (handle) => clearTimeout(handle);
// jsdom has no canvas 2D context; the labels must degrade instead of throwing.
window.HTMLCanvasElement.prototype.getContext = () => null;

const THREE = await import(VENDOR);
const { Arena3D } = await import(path.join(ROOT, 'web', 'js', 'scene.js'));

function rendererStub() {
  const sizes = [];
  return {
    sizes,
    domElement: window.document.createElement('canvas'),
    setPixelRatio() {},
    setSize(width, height) { sizes.push([width, height]); },
    setAnimationLoop(callback) { this.loop = callback; },
    render(scene, camera) { this.lastRender = { scene, camera }; },
    dispose() { this.disposed = true; },
  };
}

function staticPayload(name, objects, details, spawns) {
  return {
    name,
    width: 40,
    depth: 40,
    theme: '#00ff41',
    floor_color: '#101815',
    grid_step: 2,
    ring_radius: 6,
    objects,
    details,
    spawns,
    camera_radius: 1.1,
  };
}

function coverObjects(count, kinds = ['crate', 'barrel', 'wall']) {
  return Array.from({ length: count }, (_, index) => ({
    kind: kinds[index % kinds.length],
    color: ['#8a5a2b', '#5d6b63', '#46524b'][index % 3],
    x: ((index % 10) - 5) * 3,
    y: (Math.floor(index / 10) - 2) * 4,
    z: 0,
    w: 1.4, h: 1.6 + (index % 3) * 0.4, d: 1.2,
    yaw: (index % 8) * 0.4,
  }));
}

function floorDetails(count) {
  return {
    x: Array.from({ length: count }, (_, index) => (index % 16) - 8),
    y: Array.from({ length: count }, (_, index) => (Math.floor(index / 16) % 16) - 8),
    z: Array.from({ length: count }, () => 0.04),
    size: Array.from({ length: count }, () => 0.4 + (count % 3) * 0.1),
    color: Array.from({ length: count }, () => '#324d3b'),
  };
}

// --- arena scene ------------------------------------------------------------
const host = window.document.getElementById('host');
const renderer = rendererStub();
const arena = new Arena3D(host, { three: THREE, rendererFactory: () => renderer });
await arena.init();
check('Arena3D initialisiert', arena.ready === true);
check('Renderer-Canvas im Viewport', Boolean(host.querySelector('.viewport canvas')));

arena.setStatic(staticPayload('Test', coverObjects(300), floorDetails(320), [[-15, 0], [15, 0]]));
const instanced = [];
arena.staticGroup.traverse((node) => { if (node.isInstancedMesh) instanced.push(node); });
const instanceTotal = instanced.reduce((sum, mesh) => sum + mesh.count, 0);
check('Deckung wird instanziert (kein Objekt pro Mesh)',
  instanced.length <= 6 && instanceTotal === 620,
  `${instanced.length} InstancedMesh(es), ${instanceTotal} Instanzen`);
check('Details als ein InstancedMesh',
  instanced.some((mesh) => mesh.count === 320));
const floor = arena.staticGroup.children[0];
check('Boden entspricht der Kartengröße (40×40)',
  Math.abs(floor.geometry.parameters.width - 40) < 1e-6
  && Math.abs(floor.geometry.parameters.height - 40) < 1e-6);
check('Spawn-Pads gerendert (2)', arena.staticGroup.children.filter(
  (node) => node.geometry && node.geometry.type === 'RingGeometry').length >= 5);

// --- agents, health bar, trails --------------------------------------------
arena.setFrame({
  frame: 12,
  elapsed: 3.5,
  agents: [
    { x: -3, y: 0, z: 0, yaw: 1.57, pitch: 0, hp: 100, alive: true, height: 1.84,
      color: '#00cc33', label: 'AGENT 1', weapon: 'Pistol', ammo: 12, mag_size: 12 },
    { x: 4, y: 2, z: 0, yaw: -3.14, pitch: 0.1, hp: 34, alive: true, height: 1.2,
      color: '#ff0040', label: 'AGENT 2', weapon: 'AK-47', ammo: 3, mag_size: 30 },
  ],
  trails: Array.from({ length: 12 }, (_, index) => ({
    start: [-3, 0, 1.5], end: [-2 + index * 0.1, 4, 1.4], hit: index % 3 === 0,
  })),
});

for (let tick = 0; tick < 12; tick += 1) renderer.loop();
const [first, second] = arena.agentVisuals;
check('zwei Agenten-Visuals vorhanden', arena.agentVisuals.length === 2);
check('Agent 1 folgt der Simulation (x/y/z → x/z/y)',
  Math.abs(first.group.position.x - (-3)) < 0.4 && Math.abs(first.group.position.z - 0) < 0.4,
  `pos=(${first.group.position.x.toFixed(2)}, ${first.group.position.z.toFixed(2)})`);
check('Ducken verkürzt den Körper', second.body.scale.y < first.body.scale.y,
  `${second.body.scale.y.toFixed(2)} < ${first.body.scale.y.toFixed(2)}`);
check('Health-Bar schrumpft mit den HP', second.healthBar.scale.x < first.healthBar.scale.x,
  `${second.healthBar.scale.x.toFixed(2)} < ${first.healthBar.scale.x.toFixed(2)}`);
check('Label ohne Canvas-2D bleibt unsichtbar statt zu crashen',
  first.label.visible === false && first.labelText.includes('AGENT 1'));

const trailGeometry = arena.trails.geometry;
const trailCount = trailGeometry.drawRange.count / 2;
check('Trails: 12 Segmente = 24 Vertices', trailCount === 12, `${trailCount} Segmente`);
const positions = trailGeometry.attributes.position.array;
check('Trail-Koordinaten korrekt gemappt (sim x,y,z → three x,z,y)',
  positions[0] === -3 && positions[1] === 1.5 && positions[2] === 0,
  `[${positions[0]}, ${positions[1]}, ${positions[2]}]`);
check('Treffer-Trails sind anders eingefärbt als Fehlschüsse',
  trailGeometry.attributes.color.array[0] !== trailGeometry.attributes.color.array[6],
  `hit=[${[...trailGeometry.attributes.color.array.slice(0, 3)].map((v) => v.toFixed(2))}] `
  + `miss=[${[...trailGeometry.attributes.color.array.slice(6, 9)].map((v) => v.toFixed(2))}]`);

// --- mini game overlays ----------------------------------------------------
const aimPayload = {
  ...staticPayload('Aim', [], floorDetails(0), [[0, 0]]),
  targets: Array.from({ length: 9 }, (_, index) => ({
    cell: index, x: (index % 3 - 1) * 3, y: (Math.floor(index / 3) - 1) * 3, z: 1.5,
  })),
};
arena.setStatic(aimPayload);
arena.setOverlay({ type: 'aim', states: Array.from({ length: 9 },
  (_, index) => ({ active: index === 4, hit: index < 3 })) });
check('Zielscheiben gerendert (9)', arena.overlayMeshes.targets.length === 9);
const middle = arena.overlayMeshes.targets[4];
check('aktives Ziel wird hervorgehoben',
  middle.disc.scale.x > 1 && middle.ring.material.opacity > 0.9);

const dodgePayload = {
  ...staticPayload('Dodge', [], floorDetails(0), [[0, 0]]),
  cells: Array.from({ length: 25 }, (_, index) => ({
    column: index % 5, row: Math.floor(index / 5),
    x: (index % 5 - 2) * 3.4, y: (Math.floor(index / 5) - 2) * 3.4,
  })),
  spacing: 3.4,
  turrets: Array.from({ length: 5 }, (_, index) => ({ x: 0, y: (index - 2) * 8, angle: 0 })),
};
arena.setStatic(dodgePayload);
arena.setOverlay({
  type: 'dodge',
  game: { active: true, player: [2, 2], projectiles: [[0, 1], [3, 4]], tick: 9 },
});
const dodge = arena.overlayMeshes.dodge;
check('Dodge-Feld: 25 Kacheln', dodge.tiles.length === 25);
check('Spieler-Figur positioniert und sichtbar',
  dodge.player.visible === true && dodge.player.position.y > 0.5);
check('Projektil-Instanzen entsprechen dem Spielzustand', dodge.projectiles.count === 2);

// --- teardown --------------------------------------------------------------
arena.dispose();
check('dispose() räumt Renderer und DOM auf',
  renderer.disposed === true && !host.querySelector('.viewport'));

console.log('\n--- gefundene Probleme ---');
console.log(problems.length ? problems.join('\n') : 'keine');
console.log(failures === 0 && problems.length === 0 ? '\nERGEBNIS: OK' : `\nERGEBNIS: ${failures} Fehler`);
process.exit(failures === 0 && problems.length === 0 ? 0 : 1);
