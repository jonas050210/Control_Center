"""Experience collection: worker processes that play RocketSim matches with the current policy.

By default every car in a match is controlled by the current policy
(self-play). With ``past_opponent_prob`` > 0, that share of matches puts a
frozen *older* version of the policy on the orange team ("opponent pool");
only the blue cars' experience is trained on there. This keeps the bot from
over-fitting to its own current habits and gives an honest progress signal:
the win rate against its predecessors.

Each (match, car) pair is its own trajectory. Workers compute GAE advantages
themselves, so the learner only concatenates batches and updates.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config import PYTHON_CMD, TICK_SKIP
from .env import lookup_table, make_env
from .model import ActorCritic
from .obs import obs_size
from .teacher import NextoTeacher, TeacherPlayer, canonical_players, encode_compact


@dataclass
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    values: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)
    #: Teacher data (only with ``teacher_weight`` > 0):
    #: ``teacher_states`` is the compact world state, ``teacher_rows`` the row
    #: in the batch it belongs to, ``teacher_slots`` which car was asked,
    #: ``teacher_previous`` the last input of that car, and after the trainer
    #: has filled them in, ``teacher_probs``/``teacher_mask`` hold the target
    #: distribution per row.
    teacher_states: np.ndarray | None = None
    teacher_rows: np.ndarray | None = None
    teacher_slots: np.ndarray | None = None
    teacher_previous: np.ndarray | None = None
    teacher_probs: np.ndarray | None = None
    teacher_mask: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.actions)

    @staticmethod
    def concat(batches: list[Batch]) -> Batch:
        """Batches mehrerer Worker zu einem zusammenlegen.

        Wichtig: ``teacher_rows`` sind Zeilennummern **innerhalb** des Batches
        des jeweiligen Workers. Beim Zusammenlegen müssen sie um die Länge
        **aller** vorherigen Batches verschoben werden, sonst lernt die KI vom
        Lehrer für die falschen Spielsituationen — und zwar nur, wenn mehr als
        ein Worker läuft (bei einem Worker fällt der Fehler nicht auf).

        Der Versatz zählt deshalb jeden Stapel mit, auch die ohne Lehrer-Daten:
        Ein Worker, der in dieser Runde nichts aufgezeichnet hat, verschiebt
        sonst die Zeilen *aller* folgenden Worker um seine eigene Länge.
        """
        stats = merge_stats([b.stats for b in batches])
        states: list[np.ndarray] = []
        rows: list[np.ndarray] = []
        slots: list[np.ndarray] = []
        previous: list[np.ndarray] = []
        offset = 0
        for batch in batches:
            if batch.teacher_states is not None:
                states.append(batch.teacher_states)
                rows.append(np.asarray(batch.teacher_rows, dtype=np.int64) + offset)
                slots.append(np.asarray(batch.teacher_slots, dtype=np.int64))
                previous.append(batch.teacher_previous)
            offset += len(batch)
        return Batch(
            obs=np.concatenate([b.obs for b in batches]),
            actions=np.concatenate([b.actions for b in batches]),
            log_probs=np.concatenate([b.log_probs for b in batches]),
            advantages=np.concatenate([b.advantages for b in batches]),
            returns=np.concatenate([b.returns for b in batches]),
            values=np.concatenate([b.values for b in batches]),
            stats=stats,
            teacher_states=(np.concatenate(states) if states else None),
            teacher_rows=(np.concatenate(rows) if rows else None),
            teacher_slots=(np.concatenate(slots) if slots else None),
            teacher_previous=(np.concatenate(previous) if previous else None),
        )


def merge_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "episodes": [],
        "reward_parts": {},
        "agent_steps": 0,
        "collect_seconds": 0.0,
        "game_seconds": 0.0,
    }
    for stats in items:
        merged["episodes"].extend(stats.get("episodes", []))
        merged["agent_steps"] += stats.get("agent_steps", 0)
        merged["game_seconds"] += stats.get("game_seconds", 0.0)
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
    #: teacher questions for a subset of the steps above (local obs indices)
    teacher: list[np.ndarray] = field(default_factory=list)
    teacher_index: list[int] = field(default_factory=list)
    teacher_slot: list[int] = field(default_factory=list)


def teacher_stride(steps_per_iteration: int, teacher_samples: int) -> int:
    """Wie viele Schritte zwischen zwei aufgezeichneten Lehrer-Fragen liegen.

    Der Lernprozess beantwortet je Runde nur ``teacher_samples`` Fragen
    (``complete_teacher_batch`` wählt sie gleichmäßig aus dem Vorrat). Wird
    jeder Schritt aufgezeichnet, wandern je Runde zig MB durch die Pipe, die
    nie gefragt werden. ``stride`` stichprobt deshalb schon in den
    Simulationsprozessen — gleichmäßig über die Schritte, nicht als Block.

    Gerechnet wird mit dem **Gesamt**budget der Runde, nicht mit dem Anteil
    eines einzelnen Prozesses: Sonst liefert jeder Prozess fast die volle Dosis
    und der Vorrat ist ein Vielfaches dessen, was gefragt wird. Abgerundet,
    damit der Vorrat die gefragte Dosis noch hergibt.
    """
    if teacher_samples <= 0 or steps_per_iteration <= 0:
        return 1
    return max(1, steps_per_iteration // teacher_samples)


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
        self.extras = bool(config.get("obs_extras", True))
        self.tick_skip = int(config.get("tick_skip") or TICK_SKIP)
        self.envs = [
            make_env(
                team_size=config["team_size"],
                reward_stage=config["reward_stage"],
                episode_seconds=config["episode_seconds"],
                no_touch_seconds=config["no_touch_seconds"],
                seed=seed + index,
                obs_extras=self.extras,
            )
            for index in range(config["envs_per_worker"])
        ]
        self.model = ActorCritic(
            obs_size=int(config.get("obs_size") or obs_size(self.extras)),
            hidden_sizes=config["hidden_sizes"],
        )
        self.model.eval()
        self.rng = random.Random(seed)
        self.past_models: list[ActorCritic] = []
        #: per match: index into ``past_models`` controlling orange, "teacher",
        #: or None (self-play). The learning policy always plays blue.
        self.opponent: list[int | str | None] = [None for _ in self.envs]
        #: One teacher player per match so each keeps its own "last input"
        #: memory; they share a single loaded network.
        self.teachers: list[TeacherPlayer] = []
        self.teacher_used = False
        if float(config.get("teacher_opponent_prob") or 0.0) > 0:
            try:
                network = NextoTeacher()
            except Exception as error:  # missing download or broken files
                raise RuntimeError(
                    f"Der Lehrer konnte nicht geladen werden ({error}). "
                    f"Starte '{PYTHON_CMD} -m rocketai teacher' oder schalte den Lehrer aus."
                ) from error
            self.teachers = [TeacherPlayer(teacher=network) for _ in self.envs]
        self.obs = [env.reset() for env in self.envs]
        self.episode = [self._new_episode(env) for env in self.envs]
        self.streams: list[dict[str, _Stream]] = [{} for _ in self.envs]
        # Teacher (see rocketai/teacher.py): the workers only *record* the
        # situations; the answers are computed in the learner process.
        self.teacher_enabled = bool(
            config.get("teacher_weight") or config.get("teacher_final_weight")
        )
        self.order = [canonical_players(env.state) for env in self.envs]
        self.slot = [{agent: index for index, agent in enumerate(order)} for order in self.order]
        # Wird in ``collect`` gesetzt: hängt am Stapel dieser Runde.
        self.stride = 1
        self.teacher_tick = 0
        self.table = lookup_table()

    @staticmethod
    def _new_episode(env: Any) -> dict[str, Any]:
        """Zähler für ein Spiel.

        Ballkontakte und Tore werden **getrennt** nach Team gezählt: Die
        lernende KI spielt Blau, alles andere (Lehrer, ältere Version, anderes
        blaues Auto) ist „die anderen". Nur so ist „die KI trifft den Ball
        häufiger" eine ehrliche Aussage — sonst zählen die Kontakte des Gegners
        mit und die Zahl steigt, obwohl die KI nichts besser macht.
        """
        return {
            "reward": dict.fromkeys(env.agents, 0.0),
            "ticks": 0,
            "touches_own": 0,
            "touches_other": 0,
            # None = in diesem Spiel nie berührt.
            "first_touch": None,
            "goals_own": 0,
            "goals_other": 0,
            "opponent": "self",
        }

    def load_weights(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict)

    def set_past(self, state_dicts: list[dict[str, Any]]) -> None:
        """Replace the pool of frozen older policies."""
        models = []
        for state in state_dicts:
            model = ActorCritic(
                obs_size=self.model.obs_size, hidden_sizes=self.config["hidden_sizes"]
            )
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        self.past_models = models
        self.opponent = [
            o if (o == "teacher" or (isinstance(o, int) and o < len(models))) else None
            for o in self.opponent
        ]

    def _pick_opponent(self) -> int | str | None:
        teacher_prob = float(self.config.get("teacher_opponent_prob", 0.0) or 0.0)
        if self.teachers and self.rng.random() < teacher_prob:
            return "teacher"
        prob = float(self.config.get("past_opponent_prob", 0.0))
        if self.past_models and self.rng.random() < prob:
            return self.rng.randrange(len(self.past_models))
        return None

    def _controller(self, i: int, agent: str) -> int | str | None:
        """None = the learning policy, else a past-policy index or "teacher"."""
        opponent = self.opponent[i]
        if opponent is None or self.envs[i].state.cars[agent].team_num == 0:
            return None
        return opponent

    def collect(self, n_agent_steps: int, teacher_weight: float | None = None) -> Batch:
        """Play until ``n_agent_steps`` of *learning* steps are collected.

        ``teacher_weight`` ist das Nachahmungs-Gewicht dieses Schritts; bei 0
        werden keine Lehrer-Daten aufgezeichnet (der Lehrer ist dann aus).
        """
        started = time.perf_counter()
        gamma, lam = self.config["gamma"], self.config["gae_lambda"]
        tick_skip = self.tick_skip
        recording = self.teacher_enabled and (teacher_weight is None or teacher_weight > 0)
        wanted = int(self.config.get("teacher_samples") or 0)
        # Gesamtbudget der Runde (n_agent_steps ist nur der Anteil dieses
        # Prozesses); ohne Eintrag bleibt der eigene Anteil die Grundlage.
        budget = int(self.config.get("steps_per_iteration") or n_agent_steps)
        self.stride = 1 if (wanted <= 0 or not recording) else teacher_stride(budget, wanted)
        finished: list[_Stream] = []
        episodes: list[dict[str, Any]] = []
        parts: dict[str, float] = {}
        collected = 0
        while collected < n_agent_steps:
            all_keys = [(i, agent) for i, env in enumerate(self.envs) for agent in env.agents]
            groups: dict[Any, list[tuple[int, str]]] = {}
            for key in all_keys:
                groups.setdefault(self._controller(*key), []).append(key)
            per_env: list[dict[str, np.ndarray]] = [{} for _ in self.envs]
            keys = groups.pop(None, [])
            if keys:
                obs_batch = np.stack([self.obs[i][agent] for i, agent in keys]).astype(np.float32)
                actions, log_probs, values = self.model.act(obs_batch)
                for k, (i, agent) in enumerate(keys):
                    per_env[i][agent] = np.array([actions[k]])
                    stream = self.streams[i].setdefault(agent, _Stream())
                    stream.obs.append(obs_batch[k])
                    stream.actions.append(int(actions[k]))
                    stream.log_probs.append(float(log_probs[k]))
                    stream.values.append(float(values[k]))
                    if recording:
                        # Ask the teacher about a regular subset of the steps.
                        if self.teacher_tick % self.stride == 0:
                            stream.teacher.append(encode_compact(self.envs[i].state, self.order[i]))
                            stream.teacher_index.append(len(stream.obs) - 1)
                            stream.teacher_slot.append(self.slot[i][agent])
                        self.teacher_tick += 1
            for index, past_keys in groups.items():
                if index == "teacher":
                    continue  # handled per match below: he needs the full state
                past_obs = np.stack([self.obs[i][a] for i, a in past_keys]).astype(np.float32)
                past_actions, _, _ = self.past_models[index].act(past_obs, deterministic=False)
                for k, (i, agent) in enumerate(past_keys):
                    per_env[i][agent] = np.array([past_actions[k]])
            teacher_keys = groups.pop("teacher", [])
            if teacher_keys and self.teachers:
                self.teacher_used = True
                by_env: dict[int, list[str]] = {}
                for i, agent in teacher_keys:
                    by_env.setdefault(i, []).append(agent)
                for i, agents in by_env.items():
                    answer = self.teachers[i].act(agents, self.obs[i], self.envs[i].state)
                    for agent, action in answer.items():
                        per_env[i][agent] = np.array([action])
            collected += len(keys)
            for i, env in enumerate(self.envs):
                next_obs, rewards, terminated, truncated = env.step(per_env[i])
                episode = self.episode[i]
                episode["ticks"] += tick_skip
                state = env.state
                for car in state.cars.values():
                    # Team 0 = Blau = die lernende KI (siehe _new_episode).
                    if int(car.team_num) == 0:
                        episode["touches_own"] += car.ball_touches
                        if car.ball_touches and episode["first_touch"] is None:
                            # Stabileres Fortschrittssignal als "Kontakte pro
                            # Minute": Wann erreicht die KI den Ball zum ersten
                            # Mal? Der Balljäger-Bot braucht dafür ~3 s.
                            episode["first_touch"] = episode["ticks"] / 120.0
                    else:
                        episode["touches_other"] += car.ball_touches
                for name, value in env.shared_info.get("reward_parts", {}).items():
                    parts[name] = parts.get(name, 0.0) + value
                env.shared_info["reward_parts"] = dict.fromkeys(
                    env.shared_info.get("reward_parts", {}), 0.0
                )
                is_terminal = any(terminated.values())
                is_done = is_terminal or any(truncated.values())
                final_values: dict[str, float] = {}
                learners = list(self.streams[i])
                if is_done and not is_terminal and learners:
                    _, _, vals = self.model.act(np.stack([next_obs[a] for a in learners]))
                    final_values = dict(zip(learners, (float(v) for v in vals), strict=True))
                for agent, reward in rewards.items():
                    stream = self.streams[i].get(agent)
                    if stream is None:  # controlled by a past policy: not trained on
                        continue
                    stream.rewards.append(float(reward))
                    stream.dones.append(is_done)
                    stream.next_values.append(final_values.get(agent, 0.0))
                    episode["reward"][agent] += float(reward)
                if is_done:
                    scoring = state.scoring_team if is_terminal else None
                    if scoring is not None:
                        if int(scoring) == 0:
                            episode["goals_own"] += 1
                        else:
                            episode["goals_other"] += 1
                    learner_rewards = [episode["reward"][a] for a in learners] or [0.0]
                    opponent = self.opponent[i]
                    episode["opponent"] = (
                        "self"
                        if opponent is None
                        else ("teacher" if opponent == "teacher" else "past")
                    )
                    record = {
                        "reward": float(np.mean(learner_rewards)),
                        "seconds": episode["ticks"] / 120.0,
                        "goal": scoring is not None,
                        # "touches" = Ballkontakte der lernenden KI (Blau).
                        "touches": episode["touches_own"],
                        "touches_other": episode["touches_other"],
                        "first_touch": episode["first_touch"],
                        "goals_own": episode["goals_own"],
                        "goals_other": episode["goals_other"],
                        "opponent": episode["opponent"],
                    }
                    if opponent == "teacher":
                        # +1 the current policy (blue) scored, -1 the teacher did, 0 timeout
                        record["vs_teacher"] = 0 if scoring is None else (1 if scoring == 0 else -1)
                        record["teacher_goals_for"] = int(episode["goals_own"])
                        record["teacher_goals_against"] = int(episode["goals_other"])
                    elif opponent is not None:
                        # +1 the current policy (blue) scored, -1 the old one did, 0 timeout
                        record["vs_past"] = 0 if scoring is None else (1 if scoring == 0 else -1)
                    episodes.append(record)
                    finished.extend(self.streams[i].values())
                    self.streams[i] = {}
                    self.obs[i] = env.reset()
                    self.episode[i] = self._new_episode(env)
                    self.opponent[i] = self._pick_opponent()
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
        teacher_states: list[np.ndarray] = []
        teacher_rows: list[int] = []
        teacher_slots: list[int] = []
        teacher_previous: list[np.ndarray] = []
        row_offset = 0
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
            if stream.teacher:
                # The teacher also sees the *previous* input of that car.
                inputs = self.table[np.asarray(stream.actions, dtype=np.int64)]
                previous = np.concatenate(
                    [np.zeros((1, 8), dtype=np.float32), inputs[:-1].astype(np.float32)]
                )
                for local, slot, compact in zip(
                    stream.teacher_index, stream.teacher_slot, stream.teacher, strict=True
                ):
                    teacher_rows.append(row_offset + local)
                    teacher_slots.append(slot)
                    teacher_states.append(compact)
                    teacher_previous.append(previous[local])
            row_offset += len(stream.rewards)
        return Batch(
            obs=np.concatenate(obs),
            actions=np.concatenate(acts),
            log_probs=np.concatenate(logps),
            advantages=np.concatenate(advs),
            returns=np.concatenate(rets),
            values=np.concatenate(vals_all),
            teacher_states=(
                np.stack(teacher_states).astype(np.float32) if teacher_states else None
            ),
            teacher_rows=(np.asarray(teacher_rows, dtype=np.int64) if teacher_rows else None),
            teacher_slots=(np.asarray(teacher_slots, dtype=np.int64) if teacher_slots else None),
            teacher_previous=(
                np.stack(teacher_previous).astype(np.float32) if teacher_previous else None
            ),
            stats={
                "episodes": episodes,
                "reward_parts": parts,
                "agent_steps": collected,
                "collect_seconds": time.perf_counter() - started,
                # Simulierte Spielzeit — daraus wird der „Echtzeit-Faktor".
                "game_seconds": float(sum(e["seconds"] for e in episodes)),
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
                weights, n_steps, past, teacher_weight = payload
                collector.load_weights(weights)
                if past is not None:
                    collector.set_past(past)
                conn.send(("batch", collector.collect(n_steps, teacher_weight)))
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

    def collect(
        self,
        weights: dict[str, Any],
        total_steps: int,
        past: list[dict[str, Any]] | None = None,
        teacher_weight: float = 0.0,
    ) -> Batch:
        """``past``: opponent pool to install first (None = keep the current one).

        ``teacher_weight``: Nachahmungs-Gewicht dieses Schritts; bei 0 zeichnen
        die Worker keine Lehrer-Daten auf.
        """
        per_worker = max(1, -(-total_steps // len(self.connections)))
        for conn in self.connections:
            conn.send(("collect", (weights, per_worker, past, teacher_weight)))
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
