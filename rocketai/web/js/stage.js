// A field view that can switch between 3D (three.js) and 2D (canvas).

import { Field2D, DEFAULT_PADS } from "./field.js";
import { h, icon } from "./core.js";

let field3dModule = null;
async function load3d() {
  if (!field3dModule) field3dModule = await import("./field3d.js");
  return field3dModule;
}

const pref = {
  get mode() { return localStorage.getItem("rocketai.view") || "3d"; },
  set mode(v) { localStorage.setItem("rocketai.view", v); },
  get camera() { return localStorage.getItem("rocketai.camera") || "overview"; },
  set camera(v) { localStorage.setItem("rocketai.camera", v); },
};

const CAMERA_LABELS = { overview: "Übersicht", ball: "Ball-Cam", chase: "Verfolger", top: "Draufsicht", free: "Frei" };
const CAMERA_KEYS = { 1: "overview", 2: "ball", 3: "chase", 4: "top", 5: "free" };

export function stageToolbar() {
  return `<div class="stage-tools" data-static>
    <div class="segmented sm" data-stage="mode"><button data-v="3d">${icon("cube")}3D</button><button data-v="2d">${icon("grid")}2D</button></div>
    <div class="segmented sm" data-stage="camera">${Object.entries(CAMERA_LABELS).map(([k, l], i) => `<button data-v="${k}" title="Taste ${i + 1}">${h(l)}</button>`).join("")}</div>
    <button class="btn ghost sm icon-only" data-stage="fullscreen" title="Vollbild (F)">${icon("expand")}</button>
  </div>`;
}

/** Mounts a renderer into ``host`` and wires the toolbar inside ``root``. */
export async function createStage(root, host, { pads = DEFAULT_PADS } = {}) {
  let renderer = null, mode = null, last = null, disposed = false;
  const has3d = (await load3d()).webglAvailable();
  const tools = root.querySelector(".stage-tools");

  const sync = () => {
    if (!tools) return;
    tools.querySelectorAll("[data-stage=mode] button").forEach((b) => {
      b.classList.toggle("on", b.dataset.v === mode);
      if (b.dataset.v === "3d") b.disabled = !has3d;
    });
    const cams = tools.querySelector("[data-stage=camera]");
    cams.style.display = mode === "3d" ? "" : "none";
    cams.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === pref.camera));
  };

  const setMode = async (next) => {
    if (next === "3d" && !has3d) next = "2d";
    if (next === mode) return;
    if (renderer) renderer.dispose();
    renderer = null;
    mode = next;
    pref.mode = next;
    if (next === "3d") {
      const { Field3D } = await load3d();
      if (disposed) return;
      try {
        renderer = new Field3D(host, { pads, camera: pref.camera });
      } catch (error) {
        console.warn("3D nicht verfügbar", error);
        mode = null;
        return setMode("2d");
      }
    } else {
      renderer = new Field2D(host, { pads });
    }
    host.classList.toggle("is-3d", next === "3d");
    if (last) renderer.render(last);
    sync();
  };

  const setCamera = (cam) => {
    pref.camera = cam;
    if (mode === "3d" && renderer) renderer.setCamera(cam);
    sync();
  };

  const onClick = (e) => {
    const b = e.target.closest("button");
    if (!b || !tools || !tools.contains(b)) return;
    const group = b.closest("[data-stage]")?.dataset.stage || b.dataset.stage;
    if (group === "mode") setMode(b.dataset.v);
    else if (group === "camera") setCamera(b.dataset.v);
    else if (group === "fullscreen") toggleFullscreen();
  };
  const toggleFullscreen = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else root.requestFullscreen?.().catch(() => root.classList.toggle("pseudo-full"));
  };
  const onKey = (e) => {
    if (e.target.closest("input, select, textarea") || e.ctrlKey || e.metaKey || e.altKey) return;
    if (CAMERA_KEYS[e.key] && mode === "3d") setCamera(CAMERA_KEYS[e.key]);
    else if (e.key === "f" || e.key === "F") toggleFullscreen();
    else if (e.key === "v" || e.key === "V") setMode(mode === "3d" ? "2d" : "3d");
  };
  tools?.addEventListener("click", onClick);
  document.addEventListener("keydown", onKey);
  await setMode(pref.mode);

  return {
    get mode() { return mode; },
    render(scene) { last = scene; if (renderer) renderer.render(scene); },
    dispose() {
      disposed = true;
      tools?.removeEventListener("click", onClick);
      document.removeEventListener("keydown", onKey);
      if (renderer) renderer.dispose();
    },
  };
}

let fieldCache = null;
export async function fieldPads() {
  if (fieldCache) return fieldCache;
  try {
    const res = await fetch("/api/field");
    fieldCache = (await res.json()).pads;
  } catch { fieldCache = DEFAULT_PADS; }
  return fieldCache;
}
