"""WSL -> Windows interop for launching Windows Godot binaries from WSL.

SandboxAI's training stack can run inside Windows Subsystem for Linux while
driving the *Windows* Godot binary (e.g. the project lives on ``/mnt/c`` and
Godot was installed for Windows). Two things break that flow without help:

1.  **Path form.** The bridge passes ``--path <project>`` to Godot, and all
    Python-side paths are absolute POSIX paths such as
    ``/mnt/c/Users/<you>/SandboxAI``. A Windows process cannot resolve that
    form; the project path must be handed over as ``C:\\...``.
2.  **Launch failures.** Direct ``exec`` of a ``.exe`` goes through the
    kernel's binfmt_misc WSLInterop registration. When that registration is
    broken, when the mount lacks the exec bit, or when the Windows side
    refuses ``CreateProcess`` (e.g. elevation policy), the spawn fails with
    ``PermissionError`` / ``OSError``. The reliable alternative is
    ``cmd.exe /C call <exe> ...``.

How the pieces fit together (all verifiable in the open-sourced WSL
implementation, ``src/windows/common/interop.cpp`` ``FormatCommandLine``):

*   WSL interop translates the Linux argv list into the single Windows
    command-line string using the canonical MSVC/CommandLineToArgvW quoting
    rules. Passing an argv *list* is therefore safe for arguments containing
    spaces (``C:\\Program Files\\...``, OneDrive folders, ...) — the quoting
    is applied by WSL, never by us. This module never embeds quote characters
    in arguments; every token stays a single argv element.
*   ``cmd.exe /C`` is NOT MSVCRT-parsed: when the text after ``/C`` starts
    with a quote, cmd strips the first and the last quote of the whole line,
    which mangles any command whose executable path needed quoting. Prefixing
    the command with ``call`` guarantees the first character is a letter, so
    the line is executed verbatim.

Caveats of the ``cmd.exe`` fallback (best-effort by design):

*   Godot becomes a child of ``cmd.exe``. Graceful shutdown (the bridge's
    ``close`` command) still works, but force-killing the transport only
    kills ``cmd.exe``; Godot then runs until it notices the broken pipes.
*   cmd interprets a handful of metacharacters (``&`` ``^`` ``%``) even in
    arguments. Realistic Godot/project paths do not contain them.

Native Windows and native Linux behavior is untouched: :class:`WindowsInterop`
is a pass-through unless it detects WSL *and* a Windows executable.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# /mnt/<drive>/... — the default DrvFs automount layout ("wsl.conf [automount]
# root" may relocate it, which is why wslpath is probed first).
_DRIVE_MOUNT_PATTERN = re.compile(r"^/mnt/([A-Za-z])($|/)")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^([A-Za-z]):[\\/]")
# \\wsl$\<distro>\... / \\wsl.localhost\<distro>\... — how Windows addresses
# files that live inside a WSL distribution.
_WSL_UNC_PATTERN = re.compile(
    r"^(?:\\\\|//)(?:wsl\$|wsl\.localhost)[\\/](?P<distro>[^\\/]+)(?P<rest>[/\\].*)?$",
    re.IGNORECASE,
)

# Program names that are Windows *shells*, not Godot binaries. Passing one of
# these as --godot-executable used to be a (broken) workaround for the launch
# failures this module now handles properly.
_WINDOWS_SHELL_NAMES = frozenset(
    {
        "cmd",
        "cmd.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
    }
)


def _basename(text: str) -> str:
    return text.replace("/", "\\").rsplit("\\", 1)[-1]


def is_windows_executable(executable: str | Path) -> bool:
    """Whether `executable` names a Windows binary from WSL's point of view."""
    return _basename(str(executable)).lower().endswith(".exe")


def is_windows_shell(executable: str | Path) -> bool:
    """Whether `executable` names a Windows shell (cmd/powershell), not Godot."""
    return _basename(str(executable)).lower() in _WINDOWS_SHELL_NAMES


def looks_like_windows_path(value: str | Path) -> bool:
    """Drive-letter or UNC form, e.g. ``C:\\x``, ``c:/x``, ``\\\\wsl$\\\\...``."""
    text = str(value)
    return bool(_WINDOWS_DRIVE_PATTERN.match(text)) or text.startswith("\\\\")


def _proc_version_text() -> str:
    try:
        return Path("/proc/version").read_text(encoding="ascii", errors="ignore")
    except OSError:
        return ""


def is_wsl() -> bool:
    """True when this Python process runs inside Windows Subsystem for Linux."""
    if os.name != "posix":
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    text = _proc_version_text().lower()
    return "microsoft" in text or "wsl" in text


def _wslpath(flag: str, path: str) -> str | None:
    """Translate `path` with the ``wslpath`` helper; None when it is unavailable.

    Every failure (missing tool, non-zero exit, timeout) is swallowed: the
    callers have their own ``/mnt/<drive>`` fallback and a translation hiccup
    must never break an otherwise working launch.
    """
    tool = shutil.which("wslpath")
    if tool is None:
        return None
    try:
        completed = subprocess.run(
            [tool, flag, path],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    translated = completed.stdout.strip()
    return translated or None


def wsl_to_windows_path(path: str | Path) -> str:
    """Windows form of a POSIX path: ``/mnt/c/x`` -> ``C:\\x``.

    Uses ``wslpath -w`` when available (it handles custom automount roots and
    the ``\\\\wsl$`` UNC form for files inside the Linux filesystem) and falls
    back to the standard ``/mnt/<drive>`` mapping otherwise. Values that are
    already in Windows form pass through unchanged.
    """
    text = str(path)
    if not text or looks_like_windows_path(text):
        return text
    translated = _wslpath("-w", text)
    if translated:
        return translated
    match = _DRIVE_MOUNT_PATTERN.match(text)
    if match:
        drive = match.group(1).upper()
        rest = text[match.end() :].replace("/", "\\")
        return f"{drive}:\\{rest}" if rest else f"{drive}:\\"
    return text


def windows_to_wsl_path(path: str | Path) -> str:
    """POSIX form of a Windows path on this WSL machine: ``C:\\x`` -> ``/mnt/c/x``.

    Falls back to the standard ``/mnt/<drive>`` mapping (and to unwrapping the
    ``\\\\wsl$``/``\\\\wsl.localhost`` UNC form) when ``wslpath`` is
    unavailable. Non-Windows input passes through unchanged.
    """
    text = str(path)
    if not text or not looks_like_windows_path(text):
        return text
    translated = _wslpath("-u", text)
    if translated:
        return translated
    match = _WINDOWS_DRIVE_PATTERN.match(text)
    if match:
        drive = match.group(1).lower()
        rest = text[match.end() :].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    unc = _WSL_UNC_PATTERN.match(text)
    if unc:
        rest = (unc.group("rest") or "").replace("\\", "/")
        return rest or "/"
    return text


def normalize_host_path(path: str | Path) -> str:
    """Host-filesystem form of `path` for APIs that open it from Python.

    Lets WSL users pass a Windows-looking path (``C:\\...`` — for example one
    copied out of a shell history or remembered by a Windows-side run) to any
    SandboxAI option that expects a Python-side filesystem path.
    """
    text = str(path)
    if looks_like_windows_path(text) and is_wsl():
        return windows_to_wsl_path(text)
    return text


class GodotLaunchError(RuntimeError):
    """All candidate launch methods for the Godot process failed."""

    def __init__(self, executable: str, attempts: Sequence[tuple[Sequence[str], OSError]]) -> None:
        self.executable = executable
        self.attempts = list(attempts)
        message_lines = [f"Could not launch Godot executable {executable!r}."]
        if len(self.attempts) > 1:
            for ordinal, (argv, error) in enumerate(self.attempts, start=1):
                message_lines.append(f"  attempt {ordinal}: {list(argv)!r}")
                message_lines.append(f"    -> {error!r}")
        elif self.attempts:
            message_lines.append(f"  {self.attempts[0][1]!r}")
        message_lines.append(
            "Install Godot 4.7.2 and put it on PATH, set the GODOT_PATH "
            "environment variable, or pass --godot-executable."
        )
        if is_wsl() and is_windows_executable(executable):
            message_lines.append(
                "WSL note: running a Windows .exe from WSL needs working WSL "
                "interop — [interop] enabled=true in /etc/wsl.conf, the "
                "WSLInterop binfmt registration (systemd-binfmt), and the "
                "execute bit on the .exe as seen from WSL (chmod +x). The "
                "project path is converted to Windows form automatically."
            )
        super().__init__("\n".join(message_lines))


class WindowsInterop:
    """Launch adapter for running Windows Godot binaries from WSL.

    Instantiate with the *resolved* Godot executable, use :meth:`windows_path`
    for every path that will be handed to the Godot process, and spawn through
    :meth:`popen` / :meth:`call` / :meth:`run` — those try the direct command
    first and, when the OS refuses it (broken binfmt registration, missing
    exec bit, ...), retry through ``cmd.exe /C call`` before giving up with a
    :class:`GodotLaunchError` that reports every attempt.

    Outside WSL, or when the executable is a Linux binary, the adapter is a
    pure pass-through: one candidate, unchanged argv, unchanged paths.
    """

    def __init__(self, executable: str | Path) -> None:
        self.executable = str(executable)
        if is_windows_shell(self.executable):
            raise ValueError(
                f"{self.executable!r} is a Windows shell, not a Godot binary. "
                "Pass the Godot executable itself — e.g. --godot-executable "
                "/mnt/<drive>/.../Godot_v4.7.2-stable_win64_console.exe or set "
                "GODOT_PATH. From WSL, SandboxAI converts the project path to "
                "Windows form and wraps cmd.exe around the launch "
                "automatically when a direct launch is not possible."
            )
        self.active: bool = is_wsl() and is_windows_executable(self.executable)

    def windows_path(self, path: str | Path) -> str:
        """`path` in the form the Windows Godot process must receive."""
        if not self.active:
            return str(path)
        return wsl_to_windows_path(path)

    @staticmethod
    def _windows_executable_name(executable: str) -> str:
        """The executable as cmd.exe must see it.

        Bare names (``godot.exe``) and already-Windows paths pass through —
        cmd resolves them against the *Windows* PATH / filesystem. POSIX paths
        (``/mnt/c/...``) are translated.
        """
        if looks_like_windows_path(executable):
            return executable
        if "/" not in executable and "\\" not in executable:
            return executable
        return wsl_to_windows_path(executable)

    def launch_candidates(self, command: Sequence[str]) -> list[list[str]]:
        """Ordered argv lists to try for `command` (preferred first).

        Candidate 1 is the command itself (direct WSL interop launch).
        Candidate 2 — only when driving a Windows binary from WSL — routes the
        same invocation through ``cmd.exe /C call``. ``call`` keeps the first
        character after ``/C`` from being a quote, which is what protects the
        line from cmd's first/last-quote stripping; stdio inheritance and the
        exit code are preserved.
        """
        argv = [str(part) for part in command]
        if not argv:
            raise ValueError("launch command must not be empty")
        if not self.active:
            return [argv]
        candidates = [argv]
        shell_form = ["cmd.exe", "/C", "call", self._windows_executable_name(argv[0]), *argv[1:]]
        if shell_form != argv:
            candidates.append(shell_form)
        return candidates

    def _launch_with_fallback(
        self,
        spawn: Callable[..., Any],
        command: Sequence[str],
        **kwargs: Any,
    ) -> Any:
        attempts: list[tuple[list[str], OSError]] = []
        for argv in self.launch_candidates(command):
            try:
                return spawn(argv, **kwargs)
            except OSError as exc:
                # Any OS-level refusal of one form (PermissionError for a
                # missing exec bit or blocked CreateProcess, ENOEXEC for a
                # missing binfmt registration, FileNotFoundError for a PATH
                # lookup that only works on the Windows side, ...) is a reason
                # to try the next launch form, not to give up.
                attempts.append((argv, exc))
        raise GodotLaunchError(self.executable, attempts) from attempts[-1][1]

    def popen(self, command: Sequence[str], **kwargs: Any) -> subprocess.Popen:
        return self._launch_with_fallback(subprocess.Popen, command, **kwargs)

    def call(self, command: Sequence[str], **kwargs: Any) -> int:
        return self._launch_with_fallback(subprocess.call, command, **kwargs)

    def run(self, command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return self._launch_with_fallback(subprocess.run, command, **kwargs)
