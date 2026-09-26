"""SandboxAI feasibility: short PPO training on a godot_rl_agents env.

Usage (repo root, venv active; also works from other directories):
  python feasibility/train_ppo.py --env_path build\\fps_windows.exe \
      [--timesteps 50000] [--n_parallel 2] [--speedup 30] [--viz]

  --env_path must be the EXPORTED game executable produced by
  feasibility/export_envs.py. The trained model is saved to
  <repo>/logs/ppo_feasibility.zip. Ports used: --port .. --port+n_parallel-1.
"""

import argparse
import time

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from gdrl_common import ROOT, check_ports_free, kill_env_processes, resolve_env_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env_path", default=None, help="exported env executable (see feasibility/export_envs.py)")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--n_parallel", type=int, default=1)
    p.add_argument("--speedup", type=int, default=30)
    p.add_argument("--port", type=int, default=11008,
                   help="base TCP port (use e.g. 51008 if Windows blocks low ports)")
    p.add_argument("--viz", action="store_true")
    args = p.parse_args()

    env_path = resolve_env_path(args.env_path)
    check_ports_free(args.port, args.n_parallel)

    try:
        env = VecMonitor(
            StableBaselinesGodotEnv(
                env_path=env_path,
                n_parallel=args.n_parallel,
                show_window=args.viz,
                speedup=args.speedup,
                port=args.port,
            )
        )
    except Exception:
        kill_env_processes(env_path)
        raise

    try:
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

        # save INSIDE the repo, regardless of the current working directory
        logs_dir = ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        model_path = logs_dir / "ppo_feasibility"
        model.save(str(model_path))
        print(f"model saved to {model_path}.zip")
    finally:
        # close the connection, then make sure no game process is left over
        # (covers normal exit, Ctrl+C and mid-training exceptions)
        try:
            env.close()
        except Exception:
            pass
        kill_env_processes(env_path)


if __name__ == "__main__":
    main()
