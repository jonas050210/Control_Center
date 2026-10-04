"""Experience collection: worker processes that play RocketSim matches with the current policy.

Every car in every match is controlled by the same policy (self-play), and
each (match, car) pair is its own trajectory. Workers compute GAE advantages
themselves, so the learner only concatenates batches and updates.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .env import make_env
from .model import ActorCritic


@dataclass
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    values: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)

    @staticmethod
    def concat(batches: list[Batch]) -> Batch:
        stats = merge_stats([b.stats for b in batches])
        return Batch(
            obs=np.concatenate([b.obs for b in batches]),
            actions=np.concatenate([b.actions for b in batches]),
            log_probs=np.concatenate([b.log_probs for b in batches]),
            advantages=np.concatenate([b.advantages for b in batches]),
            returns=np.concatenate([b.returns for b in batches]),
            values=np.concatenate([b.values for b in batches]),
            stats=stats,
        )


def merge_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "episodes": [],
        "reward_parts": {},
        "agent_steps": 0,
        "collect_seconds": 0.0,
    }
    for stats in items:
        merged["episodes"].extend(stats.get("episodes", []))
        merged["agent_steps"] += stats.get("agent_steps", 0)
        merged["collect_seconds"] = max(
            merged["collect_seconds"], stats.get("collect_seconds", 0.0)
        )
        for name, value in stats.get("reward_parts", {}).items():
            merged["reward_parts"][name] = merged["reward_parts"].get(name, 0.0) + value
    return merged


@dataclass
class _Stream:
    """One car's trajectory segment inside one match."""

    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    next_values: list[float] = field(default_factory=list)  # only meaningful where done


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    bootstrap: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE for one stream. ``next_values[t]`` is used where ``dones[t]`` (0 for goals,
    V(final obs) for timeouts); ``bootstrap`` continues an unfinished last step."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        if dones[t]:
            next_value = next_values[t]
            gae = 0.0
        elif t == n - 1:
            next_value = bootstrap
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages, advantages + values


class Collector:
    """Plays ``n_envs`` matches with one policy; usable in-process (tests) or in a worker."""

    def __init__(self, config: dict[str, Any], seed: int):
        self.config = config
        self.envs = [
            make_env(
                team_size=config["team_size"],
                reward_stage=config["reward_stage"],
                episode_seconds=config["episode_seconds"],
                no_touch_seconds=config["no_touch_seconds"],
                seed=seed + index,
            )
            for index in range(config["envs_per_worker"])
        ]
        self.model = ActorCritic(hidden_sizes=config["hidden_sizes"])
        self.model.eval()
        self.obs = [env.reset() for env in self.envs]
        self.episode = [self._new_episode(env) for env in self.envs]
        self.streams: list[dict[str, _Stream]] = [{} for _ in self.envs]

    @staticmethod
    def _new_episode(env: Any) -> dict[str, Any]:
        return {"reward": dict.fromkeys(env.agents, 0.0), "ticks": 0, "touches": 0}

    def load_weights(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict)

    def collect(self, n_agent_steps: int) -> Batch:
        started = time.perf_counter()
        gamma, lam = self.config["gamma"], self.config["gae_lambda"]
        tick_skip = 8
        finished: list[_Stream] = []
        episodes: list[dict[str, Any]] = []
        parts: dict[str, float] = {}
        collected = 0
        while collected < n_agent_steps:
            keys = [(i, agent) for i, env in enumerate(self.envs) for agent in env.agents]
            obs_batch = np.stack([self.obs[i][agent] for i, agent in keys]).astype(np.float32)
            actions, log_probs, values = self.model.act(obs_batch)
            per_env: list[dict[str, np.ndarray]] = [{} for _ in self.envs]
            for k, (i, agent) in enumerate(keys):
                per_env[i][agent] = np.array([actions[k]])
                stream = self.streams[i].setdefault(agent, _Stream())
                stream.obs.append(obs_batch[k])
                stream.actions.append(int(actions[k]))
                stream.log_probs.append(float(log_probs[k]))
                stream.values.append(float(values[k]))
            collected += len(keys)
            for i, env in enumerate(self.envs):
                next_obs, rewards, terminated, truncated = env.step(per_env[i])
                episode = self.episode[i]
                episode["ticks"] += tick_skip
                state = env.state
                episode["touches"] += sum(car.ball_touches for car in state.cars.values())
                for name, value in env.shared_info.get("reward_parts", {}).items():
                    parts[name] = parts.get(name, 0.0) + value
                env.shared_info["reward_parts"] = dict.fromkeys(
                    env.shared_info.get("reward_parts", {}), 0.0
                )
                is_terminal = any(terminated.values())
                is_done = is_terminal or any(truncated.values())
                final_values: dict[str, float] = {}
                if is_done and not is_terminal:
                    agents = list(next_obs)
                    _, _, vals = self.model.act(np.stack([next_obs[a] for a in agents]))
                    final_values = dict(zip(agents, (float(v) for v in vals), strict=True))
                for agent, reward in rewards.items():
                    stream = self.streams[i][agent]
                    stream.rewards.append(float(reward))
                    stream.dones.append(is_done)
                    stream.next_values.append(final_values.get(agent, 0.0))
                    episode["reward"][agent] += float(reward)
                if is_done:
                    scoring = state.scoring_team if is_terminal else None
                    episodes.append(
                        {
                            "reward": float(np.mean(list(episode["reward"].values()))),
                            "seconds": episode["ticks"] / 120.0,
                            "goal": scoring is not None,
                            "touches": episode["touches"],
                        }
                    )
                    finished.extend(self.streams[i].values())
                    self.streams[i] = {}
                    self.obs[i] = env.reset()
                    self.episode[i] = self._new_episode(env)
                else:
                    self.obs[i] = next_obs

        # Close the open segments: bootstrap from the current observation.
        open_streams: list[tuple[_Stream, float]] = []
        for i, streams in enumerate(self.streams):
            if not streams:
                continue
            agents = list(streams)
            _, _, vals = self.model.act(
                np.stack([self.obs[i][a] for a in agents]).astype(np.float32)
            )
            for agent, value in zip(agents, vals, strict=True):
                open_streams.append((streams[agent], float(value)))
            self.streams[i] = {}

        pieces = [(stream, 0.0) for stream in finished] + open_streams
        obs, acts, logps, advs, rets, vals_all = [], [], [], [], [], []
        for stream, bootstrap in pieces:
            if not stream.rewards:
                continue
            values = np.asarray(stream.values, dtype=np.float32)
            adv, ret = compute_gae(
                np.asarray(stream.rewards, dtype=np.float32),
                values,
                np.asarray(stream.dones, dtype=bool),
                np.asarray(stream.next_values, dtype=np.float32),
                bootstrap,
                gamma,
                lam,
            )
            obs.append(np.asarray(stream.obs, dtype=np.float32))
            acts.append(np.asarray(stream.actions, dtype=np.int64))
            logps.append(np.asarray(stream.log_probs, dtype=np.float32))
            advs.append(adv)
            rets.append(ret)
            vals_all.append(values)
        return Batch(
            obs=np.concatenate(obs),
            actions=np.concatenate(acts),
            log_probs=np.concatenate(logps),
            advantages=np.concatenate(advs),
            returns=np.concatenate(rets),
            values=np.concatenate(vals_all),
            stats={
                "episodes": episodes,
                "reward_parts": parts,
                "agent_steps": collected,
                "collect_seconds": time.perf_counter() - started,
            },
        )


def _worker_main(conn: Any, config: dict[str, Any], seed: int) -> None:
    torch.set_num_threads(1)
    try:
        collector = Collector(config, seed)
        conn.send(("ready", None))
        while True:
            command, payload = conn.recv()
            if command == "collect":
                weights, n_steps = payload
                collector.load_weights(weights)
                conn.send(("batch", collector.collect(n_steps)))
            elif command == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:  # report to the learner instead of dying silently
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class WorkerPool:
    """``n`` collector processes (spawn context: identical on Windows and Linux)."""

    def __init__(self, n_workers: int, config: dict[str, Any], seed: int):
        context = mp.get_context("spawn")
        self.connections = []
        self.processes = []
        for index in range(n_workers):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(child, config, seed + 1000 * index),
                daemon=True,
                name=f"rocketai-worker-{index}",
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        for conn in self.connections:
            kind, payload = self._receive(conn)
            if kind == "error":
                self.close()
                raise RuntimeError(f"worker failed to start:\n{payload}")

    @staticmethod
    def _receive(conn: Any) -> tuple[str, Any]:
        try:
            return conn.recv()
        except EOFError:
            return "error", "simulation process exited unexpectedly (see the output above)"

    def collect(self, weights: dict[str, Any], total_steps: int) -> Batch:
        per_worker = max(1, -(-total_steps // len(self.connections)))
        for conn in self.connections:
            conn.send(("collect", (weights, per_worker)))
        batches = []
        for conn in self.connections:
            kind, payload = self._receive(conn)
            if kind == "error":
                raise RuntimeError(f"worker crashed:\n{payload}")
            batches.append(payload)
        return Batch.concat(batches)

    def close(self) -> None:
        for conn in self.connections:
            with contextlib.suppress(OSError):
                conn.send(("close", None))
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
