// 3D field renderer (three.js, bundled locally — works offline).
//
// Mapping RL world → three.js: (x, y, z) → (x, z, y). Swapping two axes
// mirrors space, which turns RL's left-handed coordinates into three.js'
// right-handed ones while keeping the picture identical to the real game.

import * as THREE from "three";
import { OrbitControls } from "three";
import { FIELD, DEFAULT_PADS, TEAM_COLOR, outline, carBasis } from "./field.js";

const v3 = (x, y, z) => new THREE.Vector3(x, z, y); // RL → three
const BALL_RADIUS = 92.75;
// Cars/ball drawn a bit larger than life so they stay readable in wide shots.
const CAR_SCALE = 1.5;
const BALL_SCALE = 1.3;

export const CAMERAS = {
  overview: "Übersicht",
  ball: "Ball-Cam",
  chase: "Verfolger",
  top: "Draufsicht",
  free: "Frei",
};

export function webglAvailable() {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGL2RenderingContext && c.getContext("webgl2")) || !!c.getContext("webgl");
  } catch { return false; }
}

function floorTexture() {
  const W = 1024, H = 1280, c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d");
  // canvas top = orange (+y), bottom = blue (−y); canvas left = +x (RL left when facing orange)
  const P = (x, y) => [((FIELD.halfX - x) / (2 * FIELD.halfX)) * W, ((FIELD.halfY - y) / (2 * FIELD.halfY)) * H];
  const grad = g.createLinearGradient(0, H, 0, 0);
  grad.addColorStop(0, "#14203a"); grad.addColorStop(0.47, "#121722"); grad.addColorStop(0.53, "#121722"); grad.addColorStop(1, "#2a1b10");
  g.fillStyle = grad; g.fillRect(0, 0, W, H);
  // subtle turf stripes
  g.fillStyle = "rgba(255,255,255,.018)";
  for (let i = 0; i < 16; i += 2) g.fillRect(0, (i / 16) * H, W, H / 16);
  g.strokeStyle = "rgba(255,255,255,.22)"; g.lineWidth = 4;
  g.beginPath(); g.moveTo(0, H / 2); g.lineTo(W, H / 2); g.stroke();
  const [cx, cy] = P(0, 0);
  g.beginPath(); g.arc(cx, cy, (1000 / (2 * FIELD.halfX)) * W, 0, Math.PI * 2); g.stroke();
  for (const sign of [-1, 1]) {
    const color = sign < 0 ? "rgba(91,140,255,.55)" : "rgba(255,154,77,.55)";
    g.strokeStyle = color; g.lineWidth = 5;
    const [x0, y0] = P(1800, sign * FIELD.halfY);
    const [x1, y1] = P(-1800, sign * (FIELD.halfY - 1300));
    g.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function makeCar(team) {
  const color = new THREE.Color(TEAM_COLOR[team]);
  const group = new THREE.Group();
  group.scale.setScalar(CAR_SCALE);
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(118, 30, 84),
    new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.25, emissive: color, emissiveIntensity: 0.12 }),
  );
  body.position.set(0, 10, 0);
  const cabin = new THREE.Mesh(
    new THREE.BoxGeometry(52, 22, 66),
    new THREE.MeshStandardMaterial({ color: 0x0b0d12, roughness: 0.2, metalness: 0.6 }),
  );
  cabin.position.set(-8, 34, 0);
  const nose = new THREE.Mesh(
    new THREE.BoxGeometry(18, 14, 76),
    new THREE.MeshStandardMaterial({ color: color.clone().multiplyScalar(1.25), roughness: 0.4 }),
  );
  nose.position.set(62, 4, 0);
  group.add(body, cabin, nose);
  const wheelGeo = new THREE.CylinderGeometry(17, 17, 12, 14);
  wheelGeo.rotateX(Math.PI / 2);
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 });
  for (const [x, z] of [[38, 42], [38, -42], [-38, 42], [-38, -42]]) {
    const wheel = new THREE.Mesh(wheelGeo, wheelMat);
    wheel.position.set(x, -6, z);
    group.add(wheel);
  }
  const flame = new THREE.Mesh(
    new THREE.ConeGeometry(13, 90, 12),
    new THREE.MeshBasicMaterial({ color: 0xffc43d, transparent: true, opacity: 0.85 }),
  );
  flame.rotation.z = Math.PI / 2;
  flame.position.set(-104, 10, 0);
  flame.visible = false;
  group.add(flame);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(170, 186, 40),
    new THREE.MeshBasicMaterial({ color: 0xc6f432, transparent: true, opacity: 0.9, side: THREE.DoubleSide }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.visible = false;
  const shadow = new THREE.Mesh(
    new THREE.PlaneGeometry(130 * CAR_SCALE, 95 * CAR_SCALE),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false }),
  );
  shadow.rotation.x = -Math.PI / 2;
  return { group, flame, ring, shadow };
}

export class Field3D {
  constructor(container, { pads = DEFAULT_PADS, camera = "overview" } = {}) {
    this.container = container;
    this.mode = camera;
    this.focus = 0;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.className = "field-canvas";
    container.appendChild(this.renderer.domElement);

    const scene = (this.scene = new THREE.Scene());
    scene.background = new THREE.Color(0x0b0d12);
    scene.fog = new THREE.Fog(0x0b0d12, 14000, 26000);
    this.camera = new THREE.PerspectiveCamera(50, 16 / 9, 10, 40000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 400;
    this.controls.maxDistance = 18000;
    this.controls.enabled = false;

    scene.add(new THREE.HemisphereLight(0xcfd8ff, 0x1a1208, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(-3000, 8000, 2000);
    scene.add(sun);

    this.buildArena(pads);

    this.ball = new THREE.Mesh(
      new THREE.SphereGeometry(BALL_RADIUS * BALL_SCALE, 32, 20),
      new THREE.MeshStandardMaterial({ color: 0xf2f4f8, roughness: 0.35, metalness: 0.1, emissive: 0x6b7280, emissiveIntensity: 0.25 }),
    );
    this.ballShadow = new THREE.Mesh(
      new THREE.CircleGeometry(BALL_RADIUS * BALL_SCALE, 24),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.45, depthWrite: false }),
    );
    this.ballShadow.rotation.x = -Math.PI / 2;
    this.ballMarker = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: 0xf2f4f8, transparent: true, opacity: 0.25 }),
    );
    scene.add(this.ball, this.ballShadow, this.ballMarker);

    this.trailLength = 40;
    this.trailGeo = new THREE.BufferGeometry();
    this.trailGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.trailLength * 3), 3));
    this.trail = new THREE.Line(this.trailGeo, new THREE.LineBasicMaterial({ color: 0xc6f432, transparent: true, opacity: 0.45 }));
    this.trail.frustumCulled = false;
    scene.add(this.trail);

    this.cars = [];
    this.camPos = new THREE.Vector3();
    this.camLook = new THREE.Vector3();
    this.setCamera(camera);

    this.resize = this.resize.bind(this);
    this.observer = new ResizeObserver(this.resize);
    this.observer.observe(container);
    this.resize();
    this.running = true;
    this.loop = this.loop.bind(this);
    this.raf = requestAnimationFrame(this.loop);
  }

  buildArena(pads) {
    const shape = new THREE.Shape(outline().map(([x, y]) => new THREE.Vector2(x, y)));
    const floorGeo = new THREE.ShapeGeometry(shape);
    floorGeo.rotateX(Math.PI / 2); // shape (x, y) → three (x, 0, y) = RL (x, y)
    // map UVs to the texture: u from +x (left) to −x, v from −y (bottom) to +y
    const pos = floorGeo.attributes.position, uv = floorGeo.attributes.uv;
    for (let i = 0; i < pos.count; i++) {
      uv.setXY(i, (FIELD.halfX - pos.getX(i)) / (2 * FIELD.halfX), (pos.getZ(i) + FIELD.halfY) / (2 * FIELD.halfY));
    }
    const floor = new THREE.Mesh(floorGeo, new THREE.MeshStandardMaterial({ map: floorTexture(), roughness: 0.9, side: THREE.DoubleSide }));
    this.scene.add(floor);

    // walls: translucent panels + bright edges; back walls leave the goal mouth open
    const wallMat = new THREE.MeshBasicMaterial({ color: 0x8b92a5, transparent: true, opacity: 0.045, side: THREE.DoubleSide, depthWrite: false });
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x3a4357, transparent: true, opacity: 0.9 });
    const pts = outline();
    const H = FIELD.height;
    const panel = (a, b, z0, z1) => {
      const geo = new THREE.BufferGeometry().setFromPoints([v3(a[0], a[1], z0), v3(b[0], b[1], z0), v3(b[0], b[1], z1), v3(a[0], a[1], z1)]);
      geo.setIndex([0, 1, 2, 0, 2, 3]);
      this.scene.add(new THREE.Mesh(geo, wallMat));
    };
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      const isBack = Math.abs(a[1]) === FIELD.halfY && Math.abs(b[1]) === FIELD.halfY;
      if (!isBack) { panel(a, b, 0, H); continue; }
      const y = a[1], g = FIELD.goalHalfWidth, dir = Math.sign(b[0] - a[0]);
      panel(a, [-dir * g, y], 0, H);
      panel([dir * g, y], b, 0, H);
      panel([-dir * g, y], [dir * g, y], FIELD.goalHeight, H);
    }
    const ring = (z) => new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts.map(([x, y]) => v3(x, y, z))), edgeMat);
    this.scene.add(ring(1), ring(H));
    for (const [x, y] of pts) this.scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([v3(x, y, 0), v3(x, y, H)]), edgeMat));

    // goals
    for (const [sign, team] of [[-1, 0], [1, 1]]) {
      const color = new THREE.Color(TEAM_COLOR[team]);
      const { goalHalfWidth: g, goalHeight: gh, goalDepth: d, halfY } = FIELD;
      const net = new THREE.Mesh(
        new THREE.BoxGeometry(2 * g, gh, d),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }),
      );
      net.position.copy(v3(0, sign * (halfY + d / 2), gh / 2));
      const frame = new THREE.LineSegments(new THREE.EdgesGeometry(net.geometry), new THREE.LineBasicMaterial({ color }));
      frame.position.copy(net.position);
      const glow = new THREE.Mesh(
        new THREE.PlaneGeometry(2 * g, 26),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, side: THREE.DoubleSide }),
      );
      glow.rotation.x = -Math.PI / 2;
      glow.position.copy(v3(0, sign * halfY, 2));
      this.scene.add(net, frame, glow);
    }

    // boost pads
    this.padMeshes = pads.map(([x, y, big]) => {
      const r = big ? 160 : 90;
      const pad = new THREE.Mesh(
        new THREE.CylinderGeometry(r, r, 8, 24),
        new THREE.MeshBasicMaterial({ color: 0xffc43d, transparent: true, opacity: 0.85 }),
      );
      pad.position.copy(v3(x, y, 4));
      this.scene.add(pad);
      let orb = null;
      if (big) {
        orb = new THREE.Mesh(
          new THREE.SphereGeometry(55, 16, 12),
          new THREE.MeshBasicMaterial({ color: 0xffd36b, transparent: true, opacity: 0.9 }),
        );
        orb.position.copy(v3(x, y, 120));
        this.scene.add(orb);
      }
      return { pad, orb };
    });
  }

  ensureCars(cars) {
    while (this.cars.length > cars.length) {
      const c = this.cars.pop();
      this.scene.remove(c.group, c.shadow, c.ring);
    }
    cars.forEach((car, i) => {
      if (!this.cars[i] || this.cars[i].team !== car.team) {
        if (this.cars[i]) this.scene.remove(this.cars[i].group, this.cars[i].shadow, this.cars[i].ring);
        const made = makeCar(car.team);
        made.team = car.team;
        this.scene.add(made.group, made.shadow, made.ring);
        this.cars[i] = made;
      }
    });
  }

  setCamera(mode) {
    this.mode = mode;
    this.controls.enabled = mode === "free";
    if (mode === "free") {
      this.camera.position.set(-7000, 5200, -6500);
      this.controls.target.set(0, 0, 0);
      this.controls.update();
    }
    if (mode === "top") this.camera.up.set(1, 0, 0);
    else this.camera.up.set(0, 1, 0);
    this.snap = true;
  }

  setFocus(index) { this.focus = index; }

  render(scene) {
    this.state = scene;
  }

  apply(scene, dt) {
    const [bx, by, bz] = scene.ball;
    this.ball.position.copy(v3(bx, by, bz));
    this.ball.rotation.x += dt * 2; this.ball.rotation.z += dt * 1.3;
    this.ballShadow.position.copy(v3(bx, by, 3));
    const s = Math.max(0.35, 1 - bz / 2500);
    this.ballShadow.scale.set(s, s, s);
    this.ballShadow.material.opacity = 0.45 * s;
    const marker = this.ballMarker.geometry.attributes.position;
    marker.setXYZ(0, bx, 3, by); marker.setXYZ(1, bx, Math.max(3, bz - BALL_RADIUS), by); marker.needsUpdate = true;
    this.ballMarker.visible = bz > 300;

    const trail = scene.trail || [];
    const attr = this.trailGeo.attributes.position;
    const n = Math.min(this.trailLength, trail.length);
    for (let i = 0; i < this.trailLength; i++) {
      const p = trail[Math.max(0, trail.length - n + Math.min(i, n - 1))] || scene.ball;
      attr.setXYZ(i, p[0], p[2], p[1]);
    }
    attr.needsUpdate = true;
    this.trailGeo.setDrawRange(0, n);

    if (scene.pads != null) {
      const bits = BigInt(scene.pads);
      this.padMeshes.forEach(({ pad, orb }, i) => {
        const on = ((bits >> BigInt(i)) & 1n) === 1n;
        pad.material.opacity = on ? 0.85 : 0.12;
        if (orb) orb.visible = on;
      });
    }
    this.padMeshes.forEach(({ orb }, i) => { if (orb) orb.position.y = 120 + Math.sin(performance.now() / 400 + i) * 12; });

    this.ensureCars(scene.cars);
    const basis = new THREE.Matrix4();
    scene.cars.forEach((car, i) => {
      const c = this.cars[i];
      const visible = !car.demo;
      c.group.visible = c.shadow.visible = visible;
      c.ring.visible = visible && scene.focus === i;
      if (!visible) return;
      const { forward, up } = carBasis(car.pitch || 0, car.yaw, car.roll || 0);
      const f = v3(...forward), u = v3(...up), r = new THREE.Vector3().crossVectors(f, u);
      basis.makeBasis(f, u, r);
      c.group.position.copy(v3(car.x, car.y, car.z));
      c.group.quaternion.setFromRotationMatrix(basis);
      c.flame.visible = !!car.boosting;
      c.flame.scale.y = 0.8 + Math.random() * 0.4;
      c.shadow.position.copy(v3(car.x, car.y, 2));
      c.shadow.rotation.z = -car.yaw;
      c.shadow.material.opacity = 0.35 * Math.max(0.2, 1 - car.z / 1500);
      c.ring.position.copy(v3(car.x, car.y, 3));
    });
  }

  updateCamera(scene, dt) {
    if (this.mode === "free") { this.controls.update(); return; }
    const target = new THREE.Vector3(), look = new THREE.Vector3();
    const ball = v3(...scene.ball);
    const car = scene.cars[scene.focus ?? this.focus] || scene.cars[0];
    const carPos = car && !car.demo ? v3(car.x, car.y, car.z) : null;
    switch (this.mode) {
      case "top":
        target.set(0, 12500, 0); look.set(0, 0, 0); break;
      case "ball":
        if (carPos) {
          const dir = ball.clone().sub(carPos); dir.y = 0;
          if (dir.lengthSq() < 1) dir.set(0, 0, 1);
          dir.normalize();
          target.copy(carPos).addScaledVector(dir, -1000); target.y = Math.max(carPos.y + 420, 320);
          look.copy(ball);
          break;
        }
      // fallthrough
      case "chase":
        if (carPos) {
          const f = v3(...carBasis(0, car.yaw, 0).forward);
          target.copy(carPos).addScaledVector(f, -900); target.y = carPos.y + 380;
          look.copy(carPos).addScaledVector(f, 500); look.y = carPos.y + 80;
          break;
        }
      // fallthrough
      default: { // overview: broadcast camera on the side, following the ball along the long axis
        const follow = THREE.MathUtils.clamp(ball.z * 0.35, -1800, 1800);
        target.set(-9400, 5000, follow * 0.7); look.set(0, -400, follow);
      }
    }
    const k = this.snap ? 1 : 1 - Math.exp(-dt * (this.mode === "overview" ? 2.5 : 6));
    this.camPos.lerp(target, k);
    this.camLook.lerp(look, k);
    if (this.snap) { this.camPos.copy(target); this.camLook.copy(look); this.snap = false; }
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.camLook);
  }

  loop(ts) {
    if (!this.running) return;
    const dt = Math.min(0.1, (ts - (this.lastTs ?? ts)) / 1000);
    this.lastTs = ts;
    if (this.state) {
      this.apply(this.state, dt);
      this.updateCamera(this.state, dt);
    }
    this.renderer.render(this.scene, this.camera);
    this.raf = requestAnimationFrame(this.loop);
  }

  resize() {
    const w = this.container.clientWidth, hh = this.container.clientHeight;
    if (!w || !hh) return;
    this.renderer.setSize(w, hh, false);
    this.camera.aspect = w / hh;
    this.camera.updateProjectionMatrix();
  }

  dispose() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.observer.disconnect();
    this.controls.dispose();
    this.scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); }
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
