"""SandboxAI feasibility: short PPO training on a godot_rl_agents env.

Usage:
  python train_ppo.py --env_path <exported exe or wrapper script> \
      [--timesteps 50000] [--n_parallel 1] [--speedup 30] [--viz]
"""

import argparse
import os
import time

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env_path", default=None)
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--n_parallel", type=int, default=1)
    p.add_argument("--speedup", type=int, default=30)
    p.add_argument("--port", type=int, default=11008,
                   help="base TCP port (use e.g. 51008 if Windows blocks low ports)")
    p.add_argument("--viz", action="store_true")
    args = p.parse_args()

    env = VecMonitor(
        StableBaselinesGodotEnv(
            env_path=args.env_path,
            n_parallel=args.n_parallel,
            show_window=args.viz,
            speedup=args.speedup,
            port=args.port,
        )
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        ent_coef=0.001,
        learning_rate=3e-4,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps)
    dt = time.perf_counter() - t0
    print(f"\nPPO finished: {args.timesteps} steps in {dt:,.0f}s "
          f"=> {args.timesteps / dt:,.0f} steps/sec including learner")
    model.save("logs/ppo_feasibility")
    env.close()


if __name__ == "__main__":
    main()
