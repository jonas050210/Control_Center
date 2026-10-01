"""WSL -> Windows Godot launch regression tests.

Hermetic tests for python/sandboxai/wsl.py and its call sites. They pin down
the failure reported from a real WSL machine:

* the Godot *project* path was handed to the Windows Godot build in POSIX
  form (``/mnt/c/...``), which Windows cannot resolve -- it must be
  converted to ``C:\\...``;
* directly launching the ``.exe`` can be refused by the OS
  (``PermissionError`` & friends), which must fall back to
  ``cmd.exe /C call``; and
* a remembered/explicit Windows-form executable path (``C:\\...``) must
  resolve under WSL.

Everything here stubs the platform probes (sandboxai.wsl.is_wsl,
sandboxai.wsl._wslpath) so almost all of the suite runs identically on
Linux, native Windows and real WSL. No machine-specific paths are
hardcoded. A handful of tests mock is_wsl() to True while exercising a
*project path built from this host's real tempdir* - on Linux that is a
faithful stand-in for a real WSL machine (a genuine POSIX filesystem),
but on native Windows it is not: is_wsl() can only ever really be True
under a POSIX Python, so `pathlib.Path` there is always PurePosixPath,
never the WindowsPath this suite would actually get on a native Windows
CI runner. Those specific tests are skipped off POSIX; see their
docstrings/comments for exactly what breaks and why it cannot occur in
production.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sandboxai.contract import OBSERVATION_FIELD_COUNT
from sandboxai.wsl import (
    GodotLaunchError,
    WindowsInterop,
    is_windows_executable,
    is_windows_shell,
    is_wsl,
    looks_like_windows_path,
    normalize_host_path,
    windows_to_wsl_path,
    wsl_to_windows_path,
)


def _bridge_handshake(transport, payload):
    """Canned GodotProcessTransport bridge handshake response."""
    return {"ok": True, "spaces": {"observation_space": {"size": OBSERVATION_FIELD_COUNT}}}


def _fake_wslpath(flag: str, path: str) -> str:
    """Deterministic stand-in for the wslpath helper (both directions)."""
    if flag == "-w":
        return "C:\\wsl" + path.replace("/", "\\")
    if flag == "-u":
        return "/wsl" + path.replace("\\", "/").replace(":", "")
    raise AssertionError(f"unexpected wslpath flag {flag!r}")


class WslDetectionTests(unittest.TestCase):
    """is_wsl() starts with `if os.name != "posix": return False` - correct
    for production (WSL only exists under a POSIX Python), but it means
    these marker/proc-version checks only ever run on a real POSIX host
    unless os.name is also stubbed. Patching it here is what makes the
    suite "hermetic" the way the module docstring promises: identical
    results on Linux, native Windows and real WSL."""

    def test_is_wsl_via_environment_markers(self):
        with mock.patch("os.name", "posix"):
            with mock.patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=True):
                self.assertTrue(is_wsl())
            with mock.patch.dict(os.environ, {"WSL_INTEROP": "/run/WSL/8_interop"}, clear=True):
                self.assertTrue(is_wsl())

    def test_is_wsl_via_proc_version(self):
        with mock.patch("os.name", "posix"):
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch(
                    "sandboxai.wsl._proc_version_text",
                    return_value="Linux version 5.15.90.1-microsoft-standard-WSL2 (gcc x86_64)",
                ):
                    self.assertTrue(is_wsl())
                with mock.patch(
                    "sandboxai.wsl._proc_version_text",
                    return_value="Linux version 6.1.0-13-amd64 (debian-kernel@lists.debian.org)",
                ):
                    self.assertFalse(is_wsl())

    def test_is_wsl_without_proc_version(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("sandboxai.wsl._proc_version_text", return_value=""),
        ):
            self.assertFalse(is_wsl())

    def test_is_windows_executable(self):
        self.assertTrue(is_windows_executable("godot.exe"))
        self.assertTrue(is_windows_executable("Godot_v4.7.2-stable_win64_CONSOLE.EXE"))
        self.assertTrue(is_windows_executable("/mnt/c/tools/Godot/godot_console.exe"))
        self.assertTrue(is_windows_executable(r"C:\tools\godot.exe"))
        self.assertFalse(is_windows_executable("godot"))
        self.assertFalse(is_windows_executable("godot4"))
        self.assertFalse(is_windows_executable("/usr/bin/Godot_v4.7.2-stable_linux.x86_64"))

    def test_is_windows_shell(self):
        for name in (
            "cmd.exe",
            r"C:\Windows\System32\cmd.exe",
            "cmd",
            "powershell.exe",
            "pwsh",
            "pwsh.exe",
        ):
            self.assertTrue(is_windows_shell(name), name)
        self.assertFalse(is_windows_shell("godot.exe"))

    def test_looks_like_windows_path(self):
        self.assertTrue(looks_like_windows_path(r"C:\Users\tester\project"))
        self.assertTrue(looks_like_windows_path("c:/Users/tester"))
        self.assertTrue(looks_like_windows_path(r"\\wsl$\Ubuntu\home\tester"))
        self.assertFalse(looks_like_windows_path("/mnt/c/Users/tester"))
        self.assertFalse(looks_like_windows_path("relative/path"))
        self.assertFalse(looks_like_windows_path(""))


class WslPathTranslationTests(unittest.TestCase):
    """Path conversion with and without the wslpath helper available."""

    def test_wslpath_result_is_used_verbatim_and_stripped(self):
        # wslpath is the authority: it understands custom automount roots and
        # the \\wsl$ UNC form for files inside the Linux filesystem.
        with (
            mock.patch("sandboxai.wsl.shutil.which", return_value="/usr/bin/wslpath"),
            mock.patch(
                "sandboxai.wsl.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [], 0, stdout="  T:\\odd\\root\\x \n", stderr=""
                ),
            ) as runner,
        ):
            self.assertEqual(wsl_to_windows_path("/weird/mount/x"), r"T:\odd\root\x")
        self.assertEqual(runner.call_args[0][0][:2], ["/usr/bin/wslpath", "-w"])

    def test_wslpath_failure_falls_back_to_drive_mount_mapping(self):
        failing = mock.Mock(side_effect=subprocess.TimeoutExpired(cmd=["wslpath"], timeout=10))
        with (
            mock.patch("sandboxai.wsl.shutil.which", return_value="/usr/bin/wslpath"),
            mock.patch("sandboxai.wsl.subprocess.run", failing),
        ):
            self.assertEqual(
                wsl_to_windows_path("/mnt/c/Users/tester/Proj Folder"),
                r"C:\Users\tester\Proj Folder",
            )
        with (
            mock.patch("sandboxai.wsl.shutil.which", return_value="/usr/bin/wslpath"),
            mock.patch(
                "sandboxai.wsl.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="err"),
            ),
        ):
            self.assertEqual(wsl_to_windows_path("/mnt/c/Users/tester"), r"C:\Users\tester")

    def test_drive_mount_fallback_without_wslpath(self):
        with mock.patch("sandboxai.wsl._wslpath", return_value=None):
            self.assertEqual(wsl_to_windows_path("/mnt/c/Users/ai/project"), r"C:\Users\ai\project")
            self.assertEqual(wsl_to_windows_path("/mnt/d/work"), r"D:\work")
            self.assertEqual(wsl_to_windows_path("/mnt/z"), "Z:\\")
            self.assertEqual(wsl_to_windows_path("/mnt/z/"), "Z:\\")
            # Not a drive mount -> left alone (a Linux path must not be mangled).
            self.assertEqual(wsl_to_windows_path("/home/tester/project"), "/home/tester/project")
            self.assertEqual(wsl_to_windows_path("/mnt/cc/not-a-drive"), "/mnt/cc/not-a-drive")
            # Already-Windows input passes through.
            self.assertEqual(wsl_to_windows_path(r"C:\already\windows"), r"C:\already\windows")

    def test_windows_to_wsl_fallbacks(self):
        with mock.patch("sandboxai.wsl._wslpath", return_value=None):
            self.assertEqual(
                windows_to_wsl_path(r"C:\Users\tester\project"), "/mnt/c/Users/tester/project"
            )
            self.assertEqual(windows_to_wsl_path("d:/tools/godot"), "/mnt/d/tools/godot")
            self.assertEqual(
                windows_to_wsl_path(r"\\wsl$\Ubuntu-24.04\home\tester"),
                "/home/tester",
            )
            self.assertEqual(windows_to_wsl_path("/already/posix"), "/already/posix")

    def test_normalize_host_path_accepts_windows_form_only_under_wsl(self):
        with (
            mock.patch("sandboxai.wsl.is_wsl", return_value=True),
            mock.patch("sandboxai.wsl._wslpath", return_value=None),
        ):
            self.assertEqual(normalize_host_path(r"C:\proj\Sandbox"), "/mnt/c/proj/Sandbox")
            self.assertEqual(normalize_host_path("/mnt/c/proj"), "/mnt/c/proj")
        with mock.patch("sandboxai.wsl.is_wsl", return_value=False):
            # Native platforms must never rewrite a path, not even a
            # Windows-looking one.
            self.assertEqual(normalize_host_path(r"C:\proj\Sandbox"), r"C:\proj\Sandbox")


class WindowsInteropPlanningTests(unittest.TestCase):
    """Pure command-construction rules (nothing is spawned here)."""

    def test_native_platform_is_an_exact_passthrough(self):
        with mock.patch("sandboxai.wsl.is_wsl", return_value=False):
            interop = WindowsInterop("godot")
        self.assertFalse(interop.active)
        command = ["godot", "--headless", "--path", "/mnt/c/proj", "--script", "res://x.gd"]
        self.assertEqual(interop.launch_candidates(command), [command])
        self.assertEqual(interop.windows_path("/mnt/c/proj"), "/mnt/c/proj")

    def test_linux_godot_on_wsl_is_a_passthrough(self):
        # A *Linux* Godot binary runs natively under WSL and must receive
        # normal POSIX paths -- no conversion, no cmd.exe fallback.
        with mock.patch("sandboxai.wsl.is_wsl", return_value=True):
            interop = WindowsInterop("/usr/bin/godot")
        self.assertFalse(interop.active)
        command = ["/usr/bin/godot", "--headless", "--path", "/mnt/c/proj"]
        self.assertEqual(interop.launch_candidates(command), [command])

    def test_windows_godot_on_wsl_converts_paths_and_offers_cmd_fallback(self):
        with (
            mock.patch("sandboxai.wsl.is_wsl", return_value=True),
            mock.patch("sandboxai.wsl._wslpath", side_effect=_fake_wslpath),
        ):
            interop = WindowsInterop("/mnt/c/tooling/Godot v4/Godot_console.exe")
            project = interop.windows_path("/mnt/c/Users/ai/Sandbox AI")
            self.assertTrue(interop.active)
            self.assertEqual(project, r"C:\wsl\mnt\c\Users\ai\Sandbox AI")
            command = ["/mnt/c/tooling/Godot v4/Godot_console.exe", "--headless", "--path", project]
            candidates = interop.launch_candidates(command)
        self.assertEqual(len(candidates), 2)
        # Direct launch first, exactly as given.
        self.assertEqual(candidates[0], command)
        # cmd.exe fallback second: /C + call (call keeps cmd's quote-stripping
        # away from quoted paths) and the executable in Windows form.
        fallback = candidates[1]
        self.assertEqual(fallback[:3], ["cmd.exe", "/C", "call"])
        self.assertNotIn("/mnt/", fallback[3])
        self.assertEqual(fallback[3], r"C:\wsl\mnt\c\tooling\Godot v4\Godot_console.exe")
        # Quoting-safety invariant of the whole design: WSL owns all quoting
        # (MSVCRT rules), so no argv element may contain an embedded quote and
        # every token stays exactly one element (spaces included).
        for argv in candidates:
            self.assertGreater(len(argv), 1)
            for token in argv:
                self.assertNotIn('"', token)
        self.assertIn("Sandbox AI", fallback[fallback.index("--path") + 1])

    def test_windows_form_and_bare_executables_pass_through_to_cmd(self):
        with (
            mock.patch("sandboxai.wsl.is_wsl", return_value=True),
            mock.patch("sandboxai.wsl._wslpath", side_effect=_fake_wslpath),
        ):
            windows_form = WindowsInterop(r"C:\Godot\godot_console.exe")
            bare = WindowsInterop("godot.exe")
            command_win = [r"C:\Godot\godot_console.exe", "--headless"]
        self.assertEqual(
            windows_form.launch_candidates(command_win)[1],
            ["cmd.exe", "/C", "call", r"C:\Godot\godot_console.exe", "--headless"],
        )
        # cmd resolves a bare name against the *Windows* PATH.
        self.assertEqual(
            bare.launch_candidates(["godot.exe", "--headless"])[1],
            ["cmd.exe", "/C", "call", "godot.exe", "--headless"],
        )

    def test_windows_shell_is_rejected_with_guidance(self):
        for shell in ("cmd.exe", r"C:\Windows\System32\cmd.exe", "powershell", "pwsh.exe"):
            with self.assertRaises(ValueError) as ctx:
                WindowsInterop(shell)
            self.assertIn("not a Godot binary", str(ctx.exception))
            self.assertIn("--godot-executable", str(ctx.exception))

    def test_empty_command_is_rejected(self):
        with mock.patch("sandboxai.wsl.is_wsl", return_value=False):
            interop = WindowsInterop("godot")
        with self.assertRaises(ValueError):
            interop.launch_candidates([])


class WindowsInteropFallbackTests(unittest.TestCase):
    """The spawn-with-fallback behavior shared by popen/call/run."""

    def _active_interop(self):
        """A WSL-active WindowsInterop whose platform stubs stay patched for
        the rest of the test (launch-time conversion needs them too)."""
        for patcher in (
            mock.patch("sandboxai.wsl.is_wsl", return_value=True),
            mock.patch("sandboxai.wsl._wslpath", side_effect=_fake_wslpath),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        return WindowsInterop("/mnt/c/tools/Godot_console.exe")

    def test_popen_retries_after_permission_error_and_preserves_stdio_kwargs(self):
        interop = self._active_interop()
        process = mock.Mock()
        with mock.patch(
            "sandboxai.wsl.subprocess.Popen",
            side_effect=[
                PermissionError(13, "Permission denied", "/mnt/c/tools/Godot_console.exe"),
                process,
            ],
        ) as popen:
            result = interop.popen(
                ["/mnt/c/tools/Godot_console.exe", "--headless", "--path", r"C:\p"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
        self.assertIs(result, process)
        self.assertEqual(popen.call_count, 2)
        first, second = popen.call_args_list
        self.assertEqual(first[0][0][0], "/mnt/c/tools/Godot_console.exe")
        self.assertEqual(
            second[0][0][:4], ["cmd.exe", "/C", "call", r"C:\wsl\mnt\c\tools\Godot_console.exe"]
        )
        # The fallback must keep the exact stdio wiring of the direct launch.
        self.assertEqual(first[1], second[1])

    def test_popen_retries_after_missing_executable(self):
        interop = self._active_interop()
        with mock.patch(
            "sandboxai.wsl.subprocess.Popen",
            side_effect=[FileNotFoundError(2, "No such file or directory"), mock.Mock()],
        ) as popen:
            interop.popen([r"C:\path on windows only\godot.exe", "--headless"])
        self.assertEqual(popen.call_count, 2)

    def test_single_candidate_failure_raises_launch_error_without_retry(self):
        with mock.patch("sandboxai.wsl.is_wsl", return_value=False):
            interop = WindowsInterop("godot")
        with (
            mock.patch(
                "sandboxai.wsl.subprocess.Popen", side_effect=PermissionError(13, "denied")
            ) as popen,
            self.assertRaises(GodotLaunchError) as ctx,
        ):
            interop.popen(["godot", "--headless"])
        # Native behavior preserved: one attempt, no hidden side effects.
        self.assertEqual(popen.call_count, 1)
        self.assertIn("Could not launch Godot executable", str(ctx.exception))

    def test_exhausted_candidates_report_every_attempt_and_wsl_remedies(self):
        interop = self._active_interop()
        with (
            mock.patch(
                "sandboxai.wsl.subprocess.Popen",
                side_effect=[PermissionError(13, "denied"), OSError(8, "Exec format error")],
            ),
            self.assertRaises(GodotLaunchError) as ctx,
        ):
            interop.popen(["/mnt/c/tools/Godot_console.exe", "--headless"])
        message = str(ctx.exception)
        self.assertIn("attempt 1", message)
        self.assertIn("attempt 2", message)
        self.assertIn("cmd.exe", message)
        self.assertIn("WSL", message)  # interop/exec-bit remediation guidance
        self.assertIn("--godot-executable", message)

    def test_call_and_run_use_the_same_fallback(self):
        interop = self._active_interop()
        with mock.patch(
            "sandboxai.wsl.subprocess.call", side_effect=[OSError("denied"), 0]
        ) as call:
            self.assertEqual(interop.call(["/mnt/c/tools/Godot_console.exe"]), 0)
        self.assertEqual(call.call_args_list[1][0][0][0], "cmd.exe")
        completed = subprocess.CompletedProcess([], 0, stdout="4.7.2-stable\n", stderr="")
        with mock.patch(
            "sandboxai.wsl.subprocess.run",
            side_effect=[PermissionError(13, "denied"), completed],
        ) as runner:
            self.assertIs(interop.run(["/mnt/c/tools/Godot_console.exe", "--version"]), completed)
        self.assertEqual(runner.call_args_list[1][0][0][0], "cmd.exe")


class GodotTransportWslTests(unittest.TestCase):
    """End-to-end command construction of GodotProcessTransport under WSL."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.project = tmp / "Some Project Dir"
        self.project.mkdir()
        self.executable = tmp / "Godot_v4.7.2-stable_win64_console.exe"
        self.executable.touch()

    def _fake_process(self):
        process = mock.Mock()
        process.stdout = io.StringIO("")
        process.stderr = io.StringIO("")
        process.poll.return_value = 0  # exited: close() takes the quiet path
        return process

    def _make_transport(self, popen_side_effect, project_path=None):
        from sandboxai.godot_env import GodotProcessTransport

        with (
            mock.patch("sandboxai.wsl.is_wsl", return_value=True),
            mock.patch("sandboxai.wsl._wslpath", side_effect=_fake_wslpath),
            mock.patch("sandboxai.wsl.Path.exists", return_value=True),
            mock.patch(
                "sandboxai.godot_env.find_godot_executable", return_value=str(self.executable)
            ),
            mock.patch("sandboxai.wsl.subprocess.Popen", side_effect=popen_side_effect) as popen,
            mock.patch("sandboxai.godot_env.GodotProcessTransport.request", _bridge_handshake),
        ):
            transport = GodotProcessTransport(
                project_path=project_path if project_path is not None else self.project,
                godot_executable=str(self.executable),
                environment_count=1,
                request_timeout=1.0,
            )
        transport.close()
        return transport, popen

    @unittest.skipUnless(
        os.name == "posix",
        "assumes self.project/self.executable are real POSIX paths, as they "
        "would be inside real WSL; on native Windows they are WindowsPaths "
        "and the expected 'C:\\\\wsl' + path formula does not apply (see "
        "module docstring)",
    )
    def test_direct_launch_converts_project_path_to_windows_form(self):
        transport, popen = self._make_transport([self._fake_process()])
        self.assertEqual(popen.call_count, 1)
        argv = popen.call_args[0][0]
        self.assertEqual(argv[0], str(self.executable))
        self.assertEqual(
            argv[argv.index("--path") + 1], r"C:\wsl" + str(self.project).replace("/", "\\")
        )
        self.assertEqual(transport.executable, str(self.executable))

    @unittest.skipUnless(
        os.name == "posix",
        "assumes self.project/self.executable are real POSIX paths, as they "
        "would be inside real WSL; on native Windows they are WindowsPaths "
        "and the expected 'C:\\\\wsl' + path formula does not apply (see "
        "module docstring)",
    )
    def test_permission_error_falls_back_to_cmd_call_with_windows_paths(self):
        _transport, popen = self._make_transport(
            [PermissionError(13, "Permission denied", str(self.executable)), self._fake_process()]
        )
        self.assertEqual(popen.call_count, 2)
        fallback = popen.call_args_list[1][0][0]
        self.assertEqual(fallback[:3], ["cmd.exe", "/C", "call"])
        # The executable is translated to Windows form for cmd.exe ...
        self.assertEqual(fallback[3], r"C:\wsl" + str(self.executable).replace("/", "\\"))
        project_argument = fallback[fallback.index("--path") + 1]
        # ... the project path is Windows-form, exactly ONE argv element ...
        self.assertEqual(project_argument, r"C:\wsl" + str(self.project).replace("/", "\\"))
        self.assertNotIn("/mnt/", " ".join(fallback))
        # ... and nothing self-quotes: WSL's interop layer owns quoting.
        for token in fallback:
            self.assertNotIn('"', token)

    def test_all_failed_attempts_surface_actionable_error(self):
        with self.assertRaises(GodotLaunchError) as ctx:
            self._make_transport([PermissionError(13, "denied"), PermissionError(13, "denied")])
        message = str(ctx.exception)
        self.assertIn("Could not launch Godot executable", message)
        self.assertIn("attempt 2", message)
        self.assertIn("cmd.exe", message)

    def test_windows_form_project_path_is_accepted(self):
        with (
            mock.patch("sandboxai.wsl.is_wsl", return_value=True),
            mock.patch("sandboxai.wsl._wslpath", return_value=None),
        ):
            _transport, popen = self._make_transport(
                [self._fake_process()],
                project_path="C:\\wsl" + str(self.project).replace("/", "\\"),
            )
        # C:\... was normalized to the WSL-side path before exists()/resolve().
        argv = popen.call_args[0][0]
        self.assertIn("--path", argv)

    def test_windows_form_project_path_really_maps_to_the_checkout(self):
        windows_form = "C:\\wsl" + str(self.project).replace("/", "\\")
        desired_wsl = "/wsl" + str(self.project).replace("/", "\\").replace(":", "")

        def translate(flag, path):
            if flag == "-u" and path == windows_form:
                return desired_wsl
            if flag == "-w":
                return "C:\\wsl" + path.replace("/", "\\")
            return None

        from sandboxai.godot_env import GodotProcessTransport

        with (
            mock.patch("sandboxai.wsl.is_wsl", return_value=True),
            mock.patch("sandboxai.wsl._wslpath", side_effect=translate),
            mock.patch(
                "sandboxai.godot_env.find_godot_executable", return_value=str(self.executable)
            ),
            mock.patch("sandboxai.wsl.subprocess.Popen", return_value=self._fake_process()),
            mock.patch("sandboxai.godot_env.GodotProcessTransport.request", _bridge_handshake),
            mock.patch("sandboxai.wsl.Path.exists", return_value=True),
        ):
            transport = GodotProcessTransport(
                project_path=windows_form,
                godot_executable=str(self.executable),
                environment_count=1,
                request_timeout=1.0,
            )
        transport.close()
        self.assertTrue(transport.executable)


class ConfigWslResolutionTests(unittest.TestCase):
    """find_godot_executable must resolve Windows-form paths under WSL."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.checkout = tmp / "checkout"
        (self.checkout / ".sandboxai").mkdir(parents=True)
        (self.checkout / "project.godot").touch()
        self.settings_path = self.checkout / ".sandboxai" / "settings.json"
        self.executable = tmp / "Godot_v4.7.2-stable_win64_console.exe"
        self.executable.touch()
        for target, value in (("sandboxai.config._settings_path", self.settings_path),):
            patcher = mock.patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        which_patcher = mock.patch("sandboxai.config.shutil.which", return_value=None)
        which_patcher.start()
        self.addCleanup(which_patcher.stop)
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("GODOT_PATH", "GODOT_EXECUTABLE")
        }
        env_patcher = mock.patch.dict(os.environ, env, clear=True)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

    def _windows_form_of(self, path: Path) -> str:
        return "C:\\wsl" + str(path).replace("/", "\\")

    def _patched_translation(self, windows_form: str, wsl_form: str):
        def translate(path):
            return wsl_form if path == windows_form else path

        return mock.patch("sandboxai.config.windows_to_wsl_path", side_effect=translate)

    def test_remembered_windows_path_resolves_under_wsl(self):
        from sandboxai.config import find_godot_executable

        windows_form = self._windows_form_of(self.executable)
        self.settings_path.write_text(
            json.dumps({"godot_executable": windows_form}), encoding="utf-8"
        )
        with (
            mock.patch("sandboxai.config.is_wsl", return_value=True),
            self._patched_translation(windows_form, str(self.executable)),
        ):
            self.assertEqual(find_godot_executable("godot"), str(self.executable))

    def test_explicit_windows_path_resolves_under_wsl(self):
        from sandboxai.config import find_godot_executable

        windows_form = self._windows_form_of(self.executable)
        with (
            mock.patch("sandboxai.config.is_wsl", return_value=True),
            self._patched_translation(windows_form, str(self.executable)),
        ):
            self.assertEqual(find_godot_executable(windows_form), str(self.executable))

    def test_windows_path_is_ignored_off_wsl(self):
        from sandboxai.config import find_godot_executable

        windows_form = self._windows_form_of(self.executable)
        self.settings_path.write_text(
            json.dumps({"godot_executable": windows_form}), encoding="utf-8"
        )
        with (
            mock.patch("sandboxai.config.is_wsl", return_value=False),
            mock.patch("sandboxai.config._GODOT_CANDIDATES", ()),
        ):
            # Native Linux: no translation attempted, documented default.
            self.assertEqual(find_godot_executable("godot"), "godot")


class CliWslLaunchTests(unittest.TestCase):
    """The graphical record command under WSL."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.project = tmp / "checkout with space"
        self.project.mkdir()
        self.executable = tmp / "Godot_console.exe"
        self.executable.touch()

    def _cli_patches(self):
        return (
            mock.patch("sandboxai.wsl.is_wsl", return_value=True),
            mock.patch("sandboxai.wsl._wslpath", side_effect=_fake_wslpath),
            mock.patch("sandboxai.cli.save_godot_executable_setting", return_value=None),
            mock.patch("sandboxai.cli.load_godot_executable_setting", return_value=None),
        )

    @unittest.skipUnless(
        os.name == "posix",
        "build_record_command resolves project_path through "
        "normalize_host_path()+Path.resolve(); with is_wsl() mocked True "
        "that assumes a POSIX Path, as it would be under real WSL, but on "
        "native Windows Path.resolve() is WindowsPath.resolve() and "
        "invents a drive letter from cwd for the fake POSIX-ish "
        "intermediate value instead (see module docstring)",
    )
    def test_record_command_converts_project_and_output_paths(self):
        from sandboxai.cli import build_record_command

        patches = self._cli_patches()
        for patcher in patches:
            patcher.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        command = build_record_command(
            str(self.executable),
            str(self.project),
            str(self.project / "out" / "demo.jsonl"),
            5.0,
            2,
        )
        self.assertEqual(
            command[command.index("--path") + 1], r"C:\wsl" + str(self.project).replace("/", "\\")
        )
        output_value = command[command.index("--output") + 1]
        self.assertTrue(output_value.startswith("C:\\wsl"), output_value)
        self.assertNotIn("/mnt/", " ".join(command))

    def test_record_dispatch_retries_through_cmd_after_permission_error(self):
        from sandboxai.cli import main

        patches = self._cli_patches()
        for patcher in patches:
            patcher.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        with mock.patch(
            "sandboxai.cli.subprocess.call",
            side_effect=[PermissionError(13, "Permission denied"), 0],
        ) as call:
            exit_code = main(
                [
                    "record",
                    "--godot-executable",
                    str(self.executable),
                    "--project-path",
                    str(self.project),
                    "--output",
                    str(self.project / "out" / "demo.jsonl"),
                    "--duration",
                    "1",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(call.call_count, 2)
        fallback = call.call_args_list[1][0][0]
        self.assertEqual(fallback[:3], ["cmd.exe", "/C", "call"])
        self.assertNotIn("/mnt/", " ".join(fallback))

    def test_shell_as_executable_is_a_clean_cli_error(self):
        from sandboxai.cli import main

        with (
            mock.patch("sandboxai.wsl.is_wsl", return_value=True),
            mock.patch("sandboxai.wsl._wslpath", side_effect=_fake_wslpath),
            mock.patch("sandboxai.cli.save_godot_executable_setting", return_value=None) as save,
            mock.patch("sandboxai.cli.load_godot_executable_setting", return_value=None),
            mock.patch(
                "sandboxai.cli.shutil.which", return_value="/mnt/c/Windows/System32/cmd.exe"
            ),
        ):
            exit_code = main(["record", "--godot-executable", "cmd.exe"])
        self.assertEqual(exit_code, 1)
        # A shell workaround must never be remembered for later runs.
        self.assertFalse(save.called)


FAKE_WSL_BRIDGE_SOURCE = (
    r"""
import json, sys
OBS_DIM = """
    + str(OBSERVATION_FIELD_COUNT)
    + r"""
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    command = request.get("cmd")
    if command == "spaces":
        print(json.dumps({
            "ok": True,
            "action_space": {"type": "multi_discrete", "nvec": [3, 3, 3, 3, 2, 2], "dimension": 6},
            "observation_space": {"type": "structured_float_vector", "size": OBS_DIM,
                                  "shape": [OBS_DIM], "low": -1.0, "high": 1.0},
        }), flush=True)
    elif command == "ping":
        print(json.dumps({"ok": True, "pong": True}), flush=True)
    elif command == "close":
        print(json.dumps({"ok": True, "close": True}), flush=True)
        break
"""
)

# A fake `wslpath` with a deterministic, uniquely identifiable mapping:
# /posix/path <-> C:\fake\mnt\posix\path (both directions).
FAKE_WSLPATH_SOURCE = r"""#!/bin/sh
flag="$1"; p="$2"
case "$flag" in
  -w)
    rest=$(printf '%s' "$p" | tr '/' '\\')
    printf 'C:\\fake\\mnt%s\n' "$rest"
    ;;
  -u)
    case "$p" in
      C:\\fake\\mnt*)
        rest=${p#C:\\fake\\mnt}
        printf '%s\n' "$(printf '%s' "$rest" | tr '\\' '/')"
        ;;
      *)
        printf '%s\n' "$p"
        ;;
    esac
    ;;
  *) exit 1 ;;
esac
"""

# Emulates `cmd.exe /C call <prog> <args...>`: argv elements are forwarded
# intact and (like the Windows loader) the image runs regardless of any POSIX
# exec bit; the Windows image path is resolved back via wslpath -u, mirroring
# how a real cmd.exe opens C:\... directly.
FAKE_CMD_SOURCE = r"""#!/bin/sh
shift 2
prog="$1"; shift
{
  echo "CMDWRAPPER"
  for a in "$prog" "$@"; do printf 'ARG:%s\n' "$a"; done
} >> "$SANDBOXAI_FAKE_GODOT_LOG"
prog=$(wslpath -u "$prog")
exec sh "$prog" "$@"
"""

FAKE_GODOT_SOURCE_TEMPLATE = """#!/bin/sh
for a in "$@"; do printf 'ARG:%s\\n' "$a" >> "$SANDBOXAI_FAKE_GODOT_LOG"; done
exec '{pexec}' '{bridge}' "$@"
"""


@unittest.skipUnless(os.name == "posix", "fake Windows binaries rely on POSIX sh wrappers")
class WslEndToEndLaunchTests(unittest.TestCase):
    """Bringing up the real bridge transport through the WSL machinery.

    Instead of asserting on argv lists in isolation, these tests put a fake
    ``cmd.exe``, a fake ``wslpath`` and a Godot-named script on PATH and run
    the actual subprocess fallback chain — including the PermissionError
    trigger (the "Windows" .exe script deliberately loses its exec bit).
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.bin_dir = tmp / "winbin"
        self.bin_dir.mkdir()
        self.project = tmp / "Project Dir With Spaces"
        self.project.mkdir()
        self.log = tmp / "godot_launch.log"

        bridge = tmp / "fake_bridge.py"
        bridge.write_text(FAKE_WSL_BRIDGE_SOURCE, encoding="utf-8")
        godot = self.bin_dir / "Godot_v4.7.2-stable_win64_console.exe"
        godot.write_text(
            FAKE_GODOT_SOURCE_TEMPLATE.format(pexec=sys.executable, bridge=str(bridge)),
            encoding="utf-8",
        )
        self.godot = godot
        cmd = self.bin_dir / "cmd.exe"
        cmd.write_text(FAKE_CMD_SOURCE, encoding="utf-8")
        cmd.chmod(0o755)
        wslpath = self.bin_dir / "wslpath"
        wslpath.write_text(FAKE_WSLPATH_SOURCE, encoding="utf-8")
        wslpath.chmod(0o755)

        env = {
            "PATH": str(self.bin_dir) + os.pathsep + os.environ.get("PATH", ""),
            "SANDBOXAI_FAKE_GODOT_LOG": str(self.log),
        }
        env_patcher = mock.patch.dict(os.environ, env)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        wsl_patcher = mock.patch("sandboxai.wsl.is_wsl", return_value=True)
        wsl_patcher.start()
        self.addCleanup(wsl_patcher.stop)

    def _open_transport(self):
        from sandboxai.godot_env import GodotProcessTransport

        return GodotProcessTransport(
            project_path=self.project,
            godot_executable=str(self.godot),
            environment_count=1,
            request_timeout=15.0,
        )

    def _windows_project_form(self) -> str:
        return "C:\\fake\\mnt" + str(self.project).replace("/", "\\")

    def test_permission_error_falls_back_to_cmd_and_launches_bridge(self):
        # The "Windows" Godot has no exec bit: the direct launch is refused by
        # the OS, exactly like the reported PermissionError.
        self.godot.chmod(0o644)
        transport = self._open_transport()
        try:
            self.assertTrue(transport.request({"cmd": "ping"}).get("pong"))
        finally:
            transport.close()
        log = self.log.read_text(encoding="utf-8")
        # The cmd.exe wrapper ran (fallback), and every path reached Godot in
        # Windows form, as a single intact argv element (spaces included).
        self.assertIn("CMDWRAPPER", log)
        self.assertIn(f"ARG:{self._windows_project_form()}", log)
        self.assertNotIn("ARG:/mnt/", log)

    def test_direct_launch_when_exec_bit_is_present_converts_project_path(self):
        self.godot.chmod(0o755)  # binfmt interop would accept this directly
        transport = self._open_transport()
        try:
            self.assertTrue(transport.request({"cmd": "ping"}).get("pong"))
        finally:
            transport.close()
        log = self.log.read_text(encoding="utf-8")
        self.assertNotIn("CMDWRAPPER", log)  # direct candidate, no wrapper
        self.assertIn(f"ARG:{self._windows_project_form()}", log)


class RuntimeValidationWslTests(unittest.TestCase):
    def test_probe_version_falls_back_to_cmd_after_permission_error(self):
        from sandboxai.runtime_validation import RuntimeValidator

        completed = subprocess.CompletedProcess([], 0, stdout="4.7.2-stable\n", stderr="")
        with (
            mock.patch("sandboxai.wsl.is_wsl", return_value=True),
            mock.patch("sandboxai.wsl._wslpath", side_effect=_fake_wslpath),
            mock.patch(
                "sandboxai.wsl.subprocess.run",
                side_effect=[PermissionError(13, "Permission denied"), completed],
            ) as runner,
        ):
            version = RuntimeValidator(
                "/nonexistent-project", godot_executable="godot.exe"
            ).probe_version("/mnt/c/tools/Godot_console.exe")
        self.assertEqual(version, "4.7.2-stable")
        self.assertEqual(runner.call_count, 2)
        second = runner.call_args_list[1][0][0]
        self.assertEqual(second[:3], ["cmd.exe", "/C", "call"])


if __name__ == "__main__":
    unittest.main()
