"""Shared helpers for the SandboxAI feasibility scripts.

Kept dependency-light (stdlib + psutil) and importable when the scripts are
run directly (`python feasibility/benchmark_env.py ...` from the repo root).
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]

_INTERPRETERS = {"sh", "bash", "dash", "zsh"}


def resolve_env_path(env_path: str | None) -> str | None:
    """Accept --env_path relative to the repo root as well as to the cwd.

    The documented commands run from the repo root
    (e.g. --env_path build\\fps_windows.exe); this keeps them working even
    when invoked from inside feasibility/.
    """
    if env_path is None:
        return None
    p = Path(env_path)
    if p.exists():
        return str(p)
    root_rel = ROOT / env_path
    if root_rel.exists():
        return str(root_rel)
    return env_path  # let godot-rl raise its own (informative) error


def check_ports_free(base_port: int, n_parallel: int) -> None:
    """Fail fast if any required port is unavailable, BEFORE launching games.

    godot-rl binds its server socket only AFTER the game processes are
    already running, so a blocked port would otherwise leave orphaned game
    processes behind. Windows note (godot_rl_agents issue #225): some port
    ranges can be blocked by policy/permissions - retry with --port 51008.
    """
    for port in range(base_port, base_port + n_parallel):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as e:
            sys.exit(
                f"ERROR: port {port} is not available ({e}).\n"
                f"  Either another feasibility run is still active, or the port range is\n"
                f"  blocked on this machine (known Windows issue, godot_rl_agents #225).\n"
                f"  Retry with: --port 51008"
            )
        finally:
            sock.close()


def _is_env_process(info, name: str, env_file: Path) -> bool:
    """True if this process IS the environment executable.

    Matches a process whose name is the env executable (Windows exported
    .exe, or a directly executed Linux binary), or an interpreter whose
    script argument is a real file resolving to exactly the env executable
    (Linux wrapper scripts, where the process name is 'sh'). Deliberately
    does NOT match processes that merely mention the path in an argument
    (e.g. our own --env_path, or a shell whose -c script text ends with it).
    """
    if (info["name"] or "").lower() == name:
        return True
    cl = info["cmdline"] or []
    if len(cl) >= 2 and Path(cl[0]).name.lower() in _INTERPRETERS:
        script = Path(cl[1])
        if script.name.lower() != name:
            return False
        try:
            return script.is_file() and script.resolve() == env_file.resolve()
        except OSError:
            return False
    return False


def kill_env_processes(env_path: str | None) -> None:
    """Safety net: terminate leftover game processes of this environment.

    godot-rl cannot clean up the game processes it launched when the
    connection fails mid-way; this finds them (see _is_env_process) and
    terminates them gracefully first, then kills if necessary. No-op when
    nothing is left over.
    """
    if not env_path:
        return
    env_file = Path(env_path)
    name = env_file.name.lower()
    me = os.getpid()
    killed = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        if proc.pid == me:
            continue
        try:
            if _is_env_process(proc.info, name, env_file):
                # Terminate children first
                for child in proc.children(recursive=True):
                    try:
                        child.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                proc.terminate()
                # Wait briefly for clean engine shutdown before force killing
                try:
                    proc.wait(timeout=0.5)
                except psutil.TimeoutExpired:
                    proc.kill()
                killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    if killed:
        print(f"cleaned up {len(killed)} leftover game process(es): {killed}")
