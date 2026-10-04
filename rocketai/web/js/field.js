// Field geometry, frame interpolation and the 2D (top-down) renderer.
//
// Frames come from the server as [t, [bx, by, bz], cars, pads?] with
// cars = [team, x, y, z, yaw, boost, demoed, pitch?, roll?].
// Rocket League uses a left-handed system: standing in the blue goal looking
// at the orange goal (+y), +x is on your LEFT. Both renderers respect that.

export const FIELD = { halfX: 4096, halfY: 5120, height: 2044, goalHalfWidth: 893, goalHeight: 642.5, goalDepth: 880, corner: 1152 };
export const TEAM_COLOR = ["#5b8cff", "#ff9a4d"];

// Fallback if /api/field is unavailable (RLGym BOOST_LOCATIONS order).
export const DEFAULT_PADS = [
  [0, -4240, 0], [-1792, -4184, 0], [1792, -4184, 0], [-3072, -4096, 1], [3072, -4096, 1], [-940, -3308, 0], [940, -3308, 0], [0, -2816, 0],
  [-3584, -2484, 0], [3584, -2484, 0], [-1788, -2300, 0], [1788, -2300, 0], [-2048, -1036, 0], [0, -1024, 0], [2048, -1036, 0], [-3584, 0, 1],
  [-1024, 0, 0], [1024, 0, 0], [3584, 0, 1], [-2048, 1036, 0], [0, 1024, 0], [2048, 1036, 0], [-1788, 2300, 0], [1788, 2300, 0],
  [-3584, 2484, 0], [3584, 2484, 0], [0, 2816, 0], [-940, 3310, 0], [940, 3308, 0], [-3072, 4096, 1], [3072, 4096, 1], [-1792, 4184, 0],
  [1792, 4184, 0], [0, 4240, 0],
];

/** Octagon outline of the pitch as [x, y] world points (counter-clockwise from above). */
export function outline() {
  const { halfX: X, halfY: Y, corner: C } = FIELD;
  return [[X - C, -Y], [X, -Y + C], [X, Y - C], [X - C, Y], [-X + C, Y], [-X, Y - C], [-X, -Y + C], [-X + C, -Y]];
}

const wrapAngle = (d) => { while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return d; };

/** Interpolated scene between two raw frames (k in [0, 1]); never across kickoff resets. */
export function blend(A, B, k) {
  if (!B) B = A;
  const jump = Math.hypot(B[1][0] - A[1][0], B[1][1] - A[1][1]) > 1500 || B[2].length !== A[2].length;
  if (jump) k = 0;
  const lerp = (x, y) => x + (y - x) * k;
  return {
    ball: A[1].map((v, n) => lerp(v, B[1][n])),
    pads: A[3] ?? null,
    cars: A[2].map((car, n) => {
      const o = B[2][n] || car;
      return {
        team: car[0], x: lerp(car[1], o[1]), y: lerp(car[2], o[2]), z: lerp(car[3], o[3]),
        yaw: car[4] + wrapAngle(o[4] - car[4]) * k,
        pitch: (car[7] ?? 0) + wrapAngle((o[7] ?? 0) - (car[7] ?? 0)) * k,
        roll: (car[8] ?? 0) + wrapAngle((o[8] ?? 0) - (car[8] ?? 0)) * k,
        boost: car[5], demo: car[6], boosting: o[5] < car[5],
      };
    }),
  };
}

/** Unit forward/up vectors (world space) from RL Euler angles. */
export function carBasis(pitch, yaw, roll) {
  const cp = Math.cos(pitch), sp = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw), cr = Math.cos(roll), sr = Math.sin(roll);
  return {
    forward: [cp * cy, cp * sy, sp],
    up: [-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr],
  };
}

function roundRect(ctx, x, y, w, hh, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + hh, r); ctx.arcTo(x + w, y + hh, x, y + hh, r);
  ctx.arcTo(x, y + hh, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

/** Top-down canvas renderer. World +y → screen right, world +x → screen up. */
export class Field2D {
  constructor(container, { pads = DEFAULT_PADS } = {}) {
    this.pads = pads;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "field-canvas";
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.resize = this.resize.bind(this);
    this.observer = new ResizeObserver(this.resize);
    this.observer.observe(container);
    this.container = container;
    this.resize();
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    const w = this.container.clientWidth, hh = this.container.clientHeight || (w * 8800) / 13000;
    this.canvas.width = Math.max(1, Math.round(w * ratio));
    this.canvas.height = Math.max(1, Math.round(hh * ratio));
    if (this.last) this.render(this.last);
  }

  render(scene) {
    this.last = scene;
    const { ctx, canvas } = this;
    const W = canvas.width, H = canvas.height, s = Math.min(W / 12400, H / 8800);
    const X = (wy) => W / 2 + wy * s; // world y → screen x
    const Y = (wx) => H / 2 - wx * s; // world x → screen y (up)
    const { halfX, halfY, goalHalfWidth, goalDepth } = FIELD;
    ctx.clearRect(0, 0, W, H);

    // pitch
    ctx.beginPath();
    outline().forEach(([wx, wy], i) => (i ? ctx.lineTo(X(wy), Y(wx)) : ctx.moveTo(X(wy), Y(wx))));
    ctx.closePath();
    const grad = ctx.createLinearGradient(X(-halfY), 0, X(halfY), 0);
    grad.addColorStop(0, "#121a2b"); grad.addColorStop(0.5, "#11151d"); grad.addColorStop(1, "#21170f");
    ctx.fillStyle = grad; ctx.fill();
    ctx.lineWidth = Math.max(1, 20 * s); ctx.strokeStyle = "#2a3140"; ctx.stroke();
    for (const [sign, color] of [[-1, "rgba(91,140,255,.22)"], [1, "rgba(255,154,77,.22)"]]) {
      ctx.fillStyle = color;
      ctx.fillRect(Math.min(X(sign * halfY), X(sign * (halfY + goalDepth))), Y(goalHalfWidth), goalDepth * s, 2 * goalHalfWidth * s);
    }
    ctx.strokeStyle = "rgba(255,255,255,.07)"; ctx.lineWidth = Math.max(1, 14 * s);
    ctx.beginPath(); ctx.moveTo(X(0), Y(-halfX)); ctx.lineTo(X(0), Y(halfX)); ctx.stroke();
    ctx.beginPath(); ctx.arc(X(0), Y(0), 1000 * s, 0, Math.PI * 2); ctx.stroke();

    // boost pads
    this.pads.forEach(([px, py, big], i) => {
      const on = scene.pads == null || (Number(BigInt(scene.pads) >> BigInt(i)) & 1) === 1;
      ctx.fillStyle = on ? (big ? "rgba(255,196,61,.9)" : "rgba(255,196,61,.55)") : "rgba(255,255,255,.06)";
      ctx.beginPath(); ctx.arc(X(py), Y(px), (big ? 110 : 60) * s, 0, Math.PI * 2); ctx.fill();
    });

    // ball trail
    if (scene.trail && scene.trail.length > 1) {
      ctx.strokeStyle = "rgba(242,244,248,.18)"; ctx.lineWidth = Math.max(1, 30 * s); ctx.lineCap = "round";
      ctx.beginPath(); scene.trail.forEach(([bx, by], i) => (i ? ctx.lineTo(X(by), Y(bx)) : ctx.moveTo(X(by), Y(bx)))); ctx.stroke();
    }

    // cars
    scene.cars.forEach((car, index) => {
      if (car.demo) return;
      const color = TEAM_COLOR[car.team];
      const angle = Math.atan2(-Math.cos(car.yaw), Math.sin(car.yaw));
      ctx.save(); ctx.translate(X(car.y), Y(car.x)); ctx.rotate(angle);
      const air = Math.min(1, car.z / 1200);
      if (scene.focus === index) { ctx.strokeStyle = "#c6f432"; ctx.lineWidth = 3 * s * 10; ctx.beginPath(); ctx.arc(0, 0, 190 * s, 0, Math.PI * 2); ctx.stroke(); }
      ctx.shadowColor = color; ctx.shadowBlur = 18 * air * s * 40;
      ctx.fillStyle = color; roundRect(ctx, -118 * s, -42 * s, 236 * s, 84 * s, 22 * s); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(11,13,18,.75)"; roundRect(ctx, 30 * s, -30 * s, 50 * s, 60 * s, 10 * s); ctx.fill();
      if (car.boosting) { ctx.fillStyle = "rgba(255,196,61,.85)"; ctx.beginPath(); ctx.moveTo(-118 * s, -18 * s); ctx.lineTo(-210 * s, 0); ctx.lineTo(-118 * s, 18 * s); ctx.fill(); }
      ctx.restore();
      ctx.fillStyle = "#1f2430"; ctx.fillRect(X(car.y) - 120 * s, Y(car.x) + 90 * s, 240 * s, 22 * s);
      ctx.fillStyle = "#ffc43d"; ctx.fillRect(X(car.y) - 120 * s, Y(car.x) + 90 * s, 240 * s * (car.boost / 100), 22 * s);
    });

    // ball (shadow on the floor, lifted body shows height)
    const [bx, by, bz] = scene.ball;
    ctx.fillStyle = "rgba(0,0,0,.45)";
    ctx.beginPath(); ctx.ellipse(X(by), Y(bx), 92 * s, 92 * s, 0, 0, Math.PI * 2); ctx.fill();
    const r = (92 + Math.min(bz, 2000) * 0.06) * s, lift = Math.min(bz, 2000) * 0.08 * s;
    ctx.fillStyle = "#f2f4f8"; ctx.beginPath(); ctx.arc(X(by), Y(bx) - lift, r, 0, Math.PI * 2); ctx.fill();

    if (this.flash) {
      const age = (performance.now() - this.flash.t) / 1400;
      if (age < 1) {
        ctx.fillStyle = this.flash.team === 0 ? `rgba(91,140,255,${0.35 * (1 - age)})` : `rgba(255,154,77,${0.35 * (1 - age)})`;
        ctx.fillRect(0, 0, W, H);
      } else this.flash = null;
    }
  }

  setCamera() {}

  goal(team) { this.flash = { team, t: performance.now() }; }

  dispose() {
    this.observer.disconnect();
    this.canvas.remove();
  }
}
