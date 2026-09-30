# Running full validation on your own machine

`PROJECT.md` section 15 marks several P0/P1 roadmap items **PLANNED**
specifically because no session so far has had access to real target
hardware with a working Godot install: reproducing the full Python/Godot
suite, live runtime validation, and the five benchmark suites. This
environment (the one running the agent) has no GPU, 2 CPU cores, and no
Godot binary, so it cannot produce that evidence itself - and the project's
own rule is "never report estimates as measurements" (`PROJECT.md` section
16), so nothing here is guessed or filled in with placeholder numbers.

If your machine matches (or is close to) the documented target hardware -
**i7-12700F, RTX 4060 Ti, 32 GB RAM, Windows 11 + Ubuntu/WSL, Godot 4.7.2**
(`PROJECT.md` section 12) - you can generate that real evidence yourself.

## Quick start (WSL/Ubuntu + Linux Godot)

```bash
cd SandboxAI   # this repository, checked out under WSL or /mnt/c/...
tools/wsl/run_full_validation.sh /path/to/your/godot
# or, if the Godot binary sits somewhere under your Desktop/OneDrive:
tools/wsl/run_full_validation.sh
```

The script:

1. Finds/verifies the Godot executable (or tells you it couldn't and why -
   it never guesses).
2. Creates/reuses `.venv` and installs `sandboxai[test,training,lint]`.
3. Records a real system snapshot (`uname`, core count, RAM, `nvidia-smi`).
4. Runs, in order, and keeps going even if one step fails (partial real
   data is still real data):
   - `ruff check .`
   - the full Python test suite (`pytest python/tests -q`, training extras
     installed - this is the CI `full-tests` job, just on your machine)
   - the GDScript suite (`godot --headless --path . --script
     res://tests/run_tests.gd`)
   - `sandboxai validate-runtime` (live headless Godot validation)
   - `sandboxai smoke-test` (end-to-end Python/ML stack sanity check)
   - `sandboxai benchmark-suites` (the four comparable suites at
     1/4/8/16/32/64 environments)
   - `sandboxai benchmark --worker-counts 1,2,4,8` (the worker-process sweep)
5. Writes every log and every raw benchmark JSON file under
   `training/local-validation/<UTC timestamp>/`, plus a `SUMMARY.txt`.

## What to do with the results

Send back (or paste) `SUMMARY.txt` and, for anything that failed, its
`.log` file. For a full roadmap update, the whole
`training/local-validation/<timestamp>/` folder (zipped) is most useful -
it is not tracked in git (see `.gitignore`), so nothing there gets
committed by accident.

Once real numbers exist, the `PLANNED` benchmark/validation rows in
`PROJECT.md` section 15 get updated to reference them (with the exact
command and timestamp), never rewritten as prose claims without a file to
point at.

## Other environments

- **Native Windows** (no WSL): install Python 3.11+ and Godot 4.7.2 for
  Windows, then run the same commands from `tools/wsl/run_full_validation.sh`
  one by one in PowerShell (the script itself is bash-only; there is no
  native-Windows port of it yet - ask if you want one).
- **WSL driving a Windows Godot .exe**: pass the Windows path with
  `--godot-executable` (e.g. `/mnt/c/.../Godot_v4.7.2-stable_win64_console.exe`)
  to the individual `sandboxai` commands; `sandboxai.wsl` auto-detects this
  case and translates paths/launch strategy accordingly (see
  `python/tests/test_wsl_interop.py`).
