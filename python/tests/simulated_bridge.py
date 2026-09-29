"""A seed-deterministic stand-in for ``scripts/rl/rl_server.gd``.

The scripted fake in ``test_godot_env.py`` answers with constants, which is
right for protocol tests but useless for the question this helper exists to
answer: *does splitting environments across several bridge processes change
results?* Here every environment derives its observations, rewards and
episode boundaries from ``seed + environment_index`` and a per-environment
step counter, exactly like the real engine derives its world from the seed
it was given. Two different process layouts holding the same environment
seeds must therefore produce byte-identical batches.

It also accepts a synthetic per-environment step cost
(``SANDBOXAI_FAKE_STEP_USEC``) so architecture-level scaling
(serial-in-process vs concurrent processes) can be measured without a Godot
binary. The cost is a busy-wait, i.e. it consumes a core the way the real
serial GDScript simulation does.

This is test/measurement infrastructure. It never ships in a training path.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import tempfile

from sandboxai.contract import ACTION_NVEC, OBSERVATION_FIELD_COUNT

BRIDGE_SOURCE = r'''
import json, os, sys, time

OBS_DIM = __OBS_DIM__
STEP_USEC = float(os.environ.get("SANDBOXAI_FAKE_STEP_USEC", "0"))
EPISODE_LENGTH = int(os.environ.get("SANDBOXAI_FAKE_EPISODE_LENGTH", "17"))
CRASH_AFTER = int(os.environ.get("SANDBOXAI_FAKE_CRASH_AFTER_STEPS", "0"))
CRASH_SEED = int(os.environ.get("SANDBOXAI_FAKE_CRASH_SEED", "-1"))


def argv_value(name, default):
    for index, arg in enumerate(sys.argv):
        if arg == name and index + 1 < len(sys.argv):
            try:
                return int(sys.argv[index + 1])
            except ValueError:
                return default
    return default


if "--version" in sys.argv:
    sys.stdout.write("4.7.2.simulated\n")
    sys.stdout.flush()
    sys.exit(0)

ENV_COUNT = max(1, argv_value("--env-count", 1))
BASE_SEED = argv_value("--seed", 1234)

seeds = [BASE_SEED + index for index in range(ENV_COUNT)]
steps = [0 for _ in range(ENV_COUNT)]
episodes = [0 for _ in range(ENV_COUNT)]
plans = {}
total_steps = 0


def mix(a, b):
    """Deterministic 32-bit integer hash (xorshift-ish, pure Python)."""
    value = (a * 2654435761 + b * 40503 + 0x9E3779B9) & 0xFFFFFFFF
    value ^= (value >> 15)
    value = (value * 2246822519) & 0xFFFFFFFF
    value ^= (value >> 13)
    value = (value * 3266489917) & 0xFFFFFFFF
    value ^= (value >> 16)
    return value


def unit(a, b):
    return mix(a, b) / 4294967295.0


def observation(index):
    seed = seeds[index]
    step = steps[index]
    return [round(unit(seed, step * 131 + field) * 2.0 - 1.0, 6) for field in range(OBS_DIM)]


def burn(microseconds):
    if microseconds <= 0.0:
        return
    deadline = time.perf_counter() + microseconds / 1_000_000.0
    while time.perf_counter() < deadline:
        pass


def out(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    command = request.get("cmd")
    if command == "spaces":
        out({
            "ok": True,
            "action_space": {"type": "multi_discrete", "nvec": __NVEC__, "dimension": len(__NVEC__)},
            "observation_space": {"type": "structured_float_vector", "size": OBS_DIM,
                                  "shape": [OBS_DIM], "low": -1.0, "high": 1.0},
        })
    elif command == "ping":
        out({"ok": True, "pong": True})
    elif command == "reset":
        seed = int(request.get("seed", -1))
        for index in range(ENV_COUNT):
            if seed >= 0:
                seeds[index] = seed + index
            if index in plans:
                seeds[index] = int(plans.pop(index)["seed"])
            steps[index] = 0
            episodes[index] += 1
        out({"ok": True,
             "observations": [observation(index) for index in range(ENV_COUNT)],
             "infos": [{"seed": seeds[index]} for index in range(ENV_COUNT)]})
    elif command == "set_episode_plans":
        staged = []
        for plan in request.get("plans", []):
            index = int(plan.get("index", -1))
            if index < 0 or index >= ENV_COUNT:
                out({"ok": False, "error": "plan index %d out of range" % index})
                break
            plans[index] = plan
            staged.append(index)
        else:
            out({"ok": True, "staged": staged})
    elif command == "episode_conditions":
        out({"ok": True, "conditions": [
            {"environment_index": index, "seed": seeds[index], "episode": episodes[index]}
            for index in range(ENV_COUNT)
        ]})
    elif command == "metrics":
        out({"ok": True, "metrics": [
            {"environment_index": index, "episode_length": steps[index], "seed": seeds[index]}
            for index in range(ENV_COUNT)
        ]})
    elif command == "health_check":
        out({"ok": True, "health": [
            {"environment_index": index, "healthy": True} for index in range(ENV_COUNT)
        ]})
    elif command == "reward_breakdown":
        out({"ok": True, "breakdowns": [
            {"environment_index": index, "reward_hits": 0.0} for index in range(ENV_COUNT)
        ]})
    elif command == "set_curriculum":
        out({"ok": True, "curriculum_level": int(request.get("level", 1))})
    elif command == "step":
        actions = request.get("actions", [])
        observations = []
        rewards = []
        dones = []
        infos = []
        for index in range(ENV_COUNT):
            action = actions[index] if index < len(actions) else [1, 1, 1, 1, 0, 0]
            burn(STEP_USEC)
            steps[index] += 1
            total_steps += 1
            if CRASH_AFTER > 0 and total_steps >= CRASH_AFTER and (
                CRASH_SEED < 0 or BASE_SEED == CRASH_SEED
            ):
                sys.stderr.write("simulated worker crash\n")
                sys.stderr.flush()
                os._exit(3)
            reward = round(
                unit(seeds[index], steps[index] * 7919) - 0.5 + 0.01 * sum(int(v) for v in action),
                6,
            )
            done = steps[index] >= EPISODE_LENGTH
            info = {"done_reason": "timeout" if done else "",
                    "metrics": {"episode_length": steps[index], "seed": seeds[index]}}
            if done:
                info["terminal_observation"] = observation(index)
                info["TimeLimit.truncated"] = True
                # Auto-reset exactly like the real vector bridge: the
                # environment's own seeded stream continues unless a plan
                # was staged for it.
                episodes[index] += 1
                steps[index] = 0
                if index in plans:
                    seeds[index] = int(plans.pop(index)["seed"])
                else:
                    seeds[index] = mix(seeds[index], episodes[index]) % 1000000007
            observations.append(observation(index))
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        out({"ok": True, "observations": observations, "rewards": rewards,
             "dones": dones, "infos": infos})
    elif command == "profile_snapshot":
        out({"ok": True, "profile": {"available": True, "timings": {}, "counters": {}}})
    elif command == "close":
        out({"ok": True, "close": True})
        break
    else:
        out({"ok": False, "error": "unknown command: %s" % command})
'''.replace("__OBS_DIM__", str(OBSERVATION_FIELD_COUNT)).replace(
    "__NVEC__", str(list(ACTION_NVEC))
)


class SimulatedBridgeExecutable:
    """Creates an executable that speaks the bridge protocol.

    Usable as a context manager; ``path`` is what callers pass as
    ``godot_executable``.
    """

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="sandboxai-simulated-bridge-")
        root = Path(self._tmp.name)
        source = root / "simulated_bridge.py"
        source.write_text(BRIDGE_SOURCE, encoding="utf-8")
        wrapper = root / "simulated_godot"
        wrapper.write_text(
            f"#!/bin/sh\nexec '{sys.executable}' '{source}' \"$@\"\n", encoding="utf-8"
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.path = str(wrapper)

    def cleanup(self) -> None:
        self._tmp.cleanup()

    def __enter__(self) -> "SimulatedBridgeExecutable":
        return self

    def __exit__(self, *_args: object) -> None:
        self.cleanup()


def supported() -> bool:
    """POSIX only: the wrapper relies on shebang execution."""
    return os.name == "posix"
