"""SandboxAI feasibility: raw environment throughput benchmark.

Connects to a godot_rl_agents environment, sends random actions and
measures environment steps/sec, RAM and (if nvidia-smi is present) VRAM.

Usage:
  python benchmark_env.py --env_path <exported exe or wrapper script> \
      [--n_parallel 1] [--speedup 30] [--seconds 30] [--viz]

  Without --env_path it waits for a Godot editor "play" on port 11008.
"""

import argparse
import shutil
import subprocess
import time

import numpy as np
import psutil

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


def gpu_mem_mb() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env_path", default=None, help="exported env executable (or wrapper script)")
    p.add_argument("--n_parallel", type=int, default=1)
    p.add_argument("--speedup", type=int, default=30)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--viz", action="store_true", help="show the game window")
    args = p.parse_args()

    env = StableBaselinesGodotEnv(
        env_path=args.env_path,
        n_parallel=args.n_parallel,
        show_window=args.viz,
        speedup=args.speedup,
    )
    n = env.num_envs
    print(f"connected: num_envs={n} (agents x instances), obs_space={env.observation_space}")

    action_space = env.envs[0].action_space

    def random_actions():
        return np.stack([action_space.sample() for _ in range(n)])

    # warmup
    for _ in range(10):
        env.step(random_actions())

    ram0 = psutil.virtual_memory().used / 1e6
    vram = gpu_mem_mb()
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        env.step(random_actions())
        steps += n
    dt = time.perf_counter() - t0

    ram1 = psutil.virtual_memory().used / 1e6
    vram1 = gpu_mem_mb()
    print("-" * 50)
    print(f"env steps/sec (total, all agents):  {steps / dt:,.0f}")
    print(f"python-side step calls/sec:         {steps / n / dt:,.0f}")
    print(f"system RAM used during run:         {ram1:,.0f} MB (delta {ram1 - ram0:+,.0f} MB)")
    if vram1 is not None:
        print(f"GPU VRAM used:                      {vram1:,.0f} MB")
    print("-" * 50)
    env.close()


if __name__ == "__main__":
    main()
