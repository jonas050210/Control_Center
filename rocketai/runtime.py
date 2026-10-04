"""Laufzeit-Helfer: eine Sperre pro Trainings-Run und ein Lebenszeichen.

Zwei Trainingsprozesse für denselben Run würden sich gegenseitig die
Checkpoints überschreiben. Deshalb hält jeder Trainer eine *Dateisperre*
(``trainer.lock``); das Betriebssystem gibt sie automatisch frei, wenn der
Prozess abstürzt oder beendet wird. Damit funktioniert die Sperre auch über
mehrere Prozesse hinweg (Weboberfläche *und* Kommandozeile) — anders als eine
Sperre, die nur im Speicher des Webservers liegt.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = [
    "AlreadyRunning",
    "TrainLock",
    "lock_owner",
    "process_alive",
    "read_pid_file",
]


class AlreadyRunning(RuntimeError):
    """Ein anderer Prozess trainiert diesen Run bereits."""

    def __init__(self, name: str, pid: int | None, since: float | None = None):
        detail = f" (Prozess {pid}" if pid else ""
        if detail and since:
            detail += f", seit {time.strftime('%H:%M:%S', time.localtime(since))}"
        detail += ")" if detail else ""
        super().__init__(f"'{name}' wird schon trainiert{detail}")


def read_pid_file(path: Path) -> dict[str, Any]:
    """Inhalt der Sperrdatei als ``{"pid": ..., "since": ...}`` (leer, wenn kaputt)."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    pid_text, _, rest = text.partition(" ")
    try:
        pid = int(pid_text)
    except ValueError:
        return {}
    info: dict[str, Any] = {"pid": pid}
    with contextlib.suppress(ValueError):
        info["since"] = float(rest)
    return info


def process_alive(pid: int) -> bool:
    """Läuft ein Prozess mit dieser Nummer? (POSIX und Windows)"""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":  # pragma: no cover - Windows-Pfad
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
            handle, ctypes.byref(exit_code)
        )
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        # 259 = STILL_ACTIVE
        return bool(ok) and exit_code.value == 259
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # existiert, gehört aber einem anderen Benutzer
    return True


def lock_owner(directory: Path) -> dict[str, Any] | None:
    """Wer trainiert diesen Run? ``None``, wenn niemand (mehr) trainiert.

    Nach einem Absturz bleibt die Sperrdatei liegen; deshalb wird zusätzlich
    geprüft, ob der eingetragene Prozess noch lebt.
    """
    info = read_pid_file(Path(directory) / "trainer.lock")
    if not info:
        return None
    return {**info, "alive": process_alive(int(info["pid"]))}


class TrainLock:
    """Exklusive Sperre für einen Run, gehalten von *einem* Prozess.

    Benutzung::

        with TrainLock(paths.root, name="my-bot"):
            Trainer(config).train()

    Die Sperrdatei bleibt nach dem Ende liegen (sie enthält nur den letzten
    Besitzer) — entscheidend ist die Dateisperre des Betriebssystems, nicht die
    Datei selbst.
    """

    def __init__(self, directory: Path, name: str = "") -> None:
        self.directory = Path(directory)
        self.name = name or self.directory.name
        self.path = self.directory / "trainer.lock"
        self._handle: Any = None
        self._locked = False

    # -- internals -------------------------------------------------------
    def _try_lock(self) -> bool:
        assert self._handle is not None
        if os.name == "nt":  # pragma: no cover - Windows-Pfad
            import msvcrt

            try:
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                return False
            return True
        import fcntl

        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(self) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":  # pragma: no cover - Windows-Pfad
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass

    # -- context manager -------------------------------------------------
    def __enter__(self) -> TrainLock:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        info = read_pid_file(self.path)
        stale_pid = info.get("pid")
        if not self._try_lock():
            # Wichtig (Windows): Die Datei wieder schließen, bevor der Fehler
            # fliegt. Ein offener Griff verhindert dort das Löschen der
            # Sperrdatei — dann bliebe die Sperre für immer stehen.
            self._handle.close()
            self._handle = None
            owner = stale_pid if stale_pid and process_alive(stale_pid) else None
            raise AlreadyRunning(self.name, owner, info.get("since"))
        self._locked = True
        # Ab hier gehören wir die Sperre: Besitzer eintragen und kürzen.
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"{os.getpid()} {time.time():.3f}\n".encode())
        self._handle.flush()
        return self

    def __exit__(self, *_error: object) -> None:
        if self._locked:
            self._unlock()
            self._locked = False
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        # Nur aufräumen, wenn die Sperrdatei uns gehört. Auf Windows kann ein
        # anderer Prozess die Datei noch offen haben — dann bleibt sie liegen
        # (was harmlos ist: entscheidend ist die Sperre, nicht die Datei).
        info = read_pid_file(self.path)
        if info.get("pid") == os.getpid():
            with contextlib.suppress(OSError):
                self.path.unlink(missing_ok=True)


def heartbeat_writer(
    path_writer: Callable[[Path, dict[str, Any]], None],
    path: Path,
    info: Callable[[], dict[str, Any]],
    every_seconds: float = 5.0,
) -> Callable[[], None]:
    """Gibt eine Funktion zurück, die höchstens alle ``every_seconds`` schreibt."""
    last = {"time": 0.0}

    def beat(force: bool = False) -> None:  # type: ignore[assignment]
        now = time.time()
        if not force and now - last["time"] < every_seconds:
            return
        last["time"] = now
        path_writer(path, info())

    return beat  # type: ignore[return-value]
