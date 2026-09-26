"""SandboxAI feasibility: raw environment throughput benchmark.

Connects to an exported godot_rl_agents environment, sends random actions and
measures environment steps/sec, RAM and (if nvidia-smi is present) VRAM.

Usage (from the repo root, venv active):
  python feasibility/benchmark_env.py --env_path build\\fps_windows.exe
  python feasibility/benchmark_env.py --env_path build\\fps_windows.exe --n_parallel 4
  python feasibility/benchmark_env.py --env_path build\\virtualcamera_windows.exe --viz

  --env_path must be the EXPORTED game executable produced by
  feasibility/export_envs.py (godot-rl requires a real .exe on Windows /
  .x86_64 on Linux; a renamed .bat does not work, and the Godot editor
  binary itself is NOT an RL environment).

  Without --env_path it waits for a Godot editor "play" on port 11008.
  --viz keeps the game window visible - REQUIRED for pixel observations
  (RGBCameraSensor3D) because godot-rl otherwise passes --headless
  --disable-render-loop, which disables rendering entirely.
"""

import argparse
import shutil
import subprocess
import time

import numpy as np
import psutil

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


def gpu_mem_mb() -> tuple[float | None, str | None]:
    if shutil.which("nvidia-smi") is None:
        return None, None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,name", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        first = out.strip().splitlines()[0]
        mem, _, name = first.partition(",")
        return float(mem), name.strip() or None
    except Exception:
        return None, None


def godot_ram_mb(env) -> float | None:
    """Total RSS of the Godot game processes launched by godot-rl."""
    try:
        total = 0.0
        found = False
        for e in env.envs:
            if e.proc is not None and e.proc.poll() is None:
                total += psutil.Process(e.proc.pid).memory_info().rss / 1e6
                found = True
        return total if found else None
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env_path", default=None, help="exported env executable (see feasibility/export_envs.py)")
    p.add_argument("--n_parallel", type=int, default=1)
    p.add_argument("--speedup", type=int, default=30)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--port", type=int, default=11008,
                   help="base TCP port (use e.g. 51008 if Windows blocks low ports)")
    p.add_argument("--viz", action="store_true", help="show the game window (required for pixel obs)")
    args = p.parse_args()

    env = StableBaselinesGodotEnv(
        env_path=args.env_path,
        n_parallel=args.n_parallel,
        show_window=args.viz,
        speedup=args.speedup,
        port=args.port,
    )
    n = env.num_envs
    print(f"connected: num_envs={n} (agents x instances), obs_space={env.observation_space}")

    action_space = env.envs[0].action_space

    def random_actions():
        return np.stack([action_space.sample() for _ in range(n)])

    # warmup
    for _ in range(10):
        env.step(random_actions())

    vram, gpu_name = gpu_mem_mb()
    ram0 = psutil.virtual_memory().used / 1e6
    gram0 = godot_ram_mb(env)
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        env.step(random_actions())
        steps += n
    dt = time.perf_counter() - t0

    ram1 = psutil.virtual_memory().used / 1e6
    gram1 = godot_ram_mb(env)
    vram1, _ = gpu_mem_mb()
    print("-" * 60)
    print(f"instances (godot processes):        {args.n_parallel}")
    print(f"agents per instance:                {n // args.n_parallel}")
    print(f"env steps/sec (total, all agents):  {steps / dt:,.0f}")
    print(f"python-side step calls/sec:         {steps / n / dt:,.0f}")
    print(f"env steps/sec per instance:         {steps / args.n_parallel / dt:,.0f}")
    print(f"system RAM used during run:         {ram1:,.0f} MB (delta {ram1 - ram0:+,.0f} MB)")
    if gram1 is not None:
        base = f" (delta {gram1 - (gram0 or gram1):+,.0f} MB)" if gram0 is not None else ""
        print(f"godot process(es) RSS:              {gram1:,.0f} MB{base}")
    if gpu_name is not None:
        print(f"gpu:                                {gpu_name}")
    if vram1 is not None:
        print(f"GPU VRAM used:                      {vram1:,.0f} MB (start of run {vram:,.0f} MB)")
    print("-" * 60)
    env.close()


if __name__ == "__main__":
    main()
