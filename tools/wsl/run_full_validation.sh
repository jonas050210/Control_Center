#!/usr/bin/env bash
# SandboxAI full local validation + benchmark run, for WSL (Ubuntu) with a
# Linux Godot build - the setup this repo's own author uses day to day
# (project files live on the Windows desktop/OneDrive, Python and Godot both
# run inside Ubuntu/WSL against the /mnt/c/... view of them).
#
# This exists to turn PROJECT.md's still-PLANNED items (full suite
# reproduction, live runtime validation, benchmark sweeps on real target
# hardware) into real, measured numbers - nothing in this script estimates
# or fabricates anything; every result file it writes came from an actual
# command run on this machine.
#
# Usage:
#   tools/wsl/run_full_validation.sh [path/to/godot]
#   GODOT_EXECUTABLE=/mnt/c/Users/you/OneDrive/Desktop/Godot_v4.7.2-stable_linux.x86_64 \
#     tools/wsl/run_full_validation.sh
#
# If no path is given, this script searches a few common Desktop/OneDrive
# locations for something named "godot*". If it can't find exactly one
# candidate, it stops and asks you to pass the path explicitly - it will
# never guess.
#
# OneDrive note: if the Godot binary lives inside a OneDrive-synced folder
# and Windows "Files On-Demand" is on, the file can be a cloud-only
# placeholder until it has actually been opened once from Windows/Explorer
# ("Always keep on this device"). If this script reports the executable is
# missing/won't run even though it's clearly there, that is the first thing
# to check.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT" || { echo "[SandboxAI] Could not locate the repository root."; exit 1; }

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="$REPO_ROOT/training/local-validation/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/SUMMARY.txt"
: > "$SUMMARY"

log() { echo "[SandboxAI] $*" | tee -a "$SUMMARY"; }

# ---------------------------------------------------------------------------
# 1. Resolve the Godot executable.
# ---------------------------------------------------------------------------
GODOT_EXECUTABLE="${1:-${GODOT_EXECUTABLE:-}}"
if [ -z "$GODOT_EXECUTABLE" ]; then
    log "No Godot path given; searching common Desktop/OneDrive locations..."
    mapfile -t CANDIDATES < <(
        find "$HOME/Desktop" /mnt/c/Users/*/Desktop /mnt/c/Users/*/OneDrive*/Desktop \
            -maxdepth 3 -iname "godot*" -type f 2>/dev/null | sort -u
    )
    if [ "${#CANDIDATES[@]}" -eq 1 ]; then
        GODOT_EXECUTABLE="${CANDIDATES[0]}"
        log "Found exactly one candidate: $GODOT_EXECUTABLE"
    elif [ "${#CANDIDATES[@]}" -gt 1 ]; then
        log "Found multiple candidates, refusing to guess:"
        printf '  %s\n' "${CANDIDATES[@]}" | tee -a "$SUMMARY"
        log "Re-run with the exact path as the first argument."
        exit 1
    else
        log "No candidate found under \$HOME/Desktop or /mnt/c/Users/*/[OneDrive]*/Desktop."
        log "Re-run as: tools/wsl/run_full_validation.sh /path/to/godot"
        exit 1
    fi
fi
if [ ! -f "$GODOT_EXECUTABLE" ]; then
    log "ERROR: '$GODOT_EXECUTABLE' does not exist or is not a regular file."
    log "If it lives in OneDrive: open it once from Windows Explorer and choose"
    log "'Always keep on this device' (Files On-Demand can leave it as a cloud"
    log "placeholder that WSL sees but cannot actually read/execute)."
    exit 1
fi
chmod +x "$GODOT_EXECUTABLE" 2>/dev/null || true
log "Using Godot executable: $GODOT_EXECUTABLE"
"$GODOT_EXECUTABLE" --version > "$RESULTS_DIR/godot_version.txt" 2>&1
log "Godot --version: $(cat "$RESULTS_DIR/godot_version.txt")"

# ---------------------------------------------------------------------------
# 2. Python environment.
# ---------------------------------------------------------------------------
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    log "Activating existing .venv"
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
else
    log "Creating .venv"
    python3 -m venv "$REPO_ROOT/.venv"
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi
log "Installing sandboxai with test+training+lint extras (safe to re-run)"
pip install -q -e ".[test,training,lint]" 2>&1 | tail -20 | tee "$RESULTS_DIR/pip_install.log" >/dev/null

# ---------------------------------------------------------------------------
# 3. System snapshot (real, not estimated - for provenance next to results).
# ---------------------------------------------------------------------------
{
    echo "date_utc: $TIMESTAMP"
    uname -a
    echo "--- nproc ---"; nproc
    echo "--- /proc/meminfo (MemTotal) ---"; grep MemTotal /proc/meminfo
    echo "--- nvidia-smi ---"; nvidia-smi 2>&1 || echo "(nvidia-smi not available)"
    echo "--- python ---"; python --version
} > "$RESULTS_DIR/system_info.txt" 2>&1
log "System snapshot written to $RESULTS_DIR/system_info.txt"

# ---------------------------------------------------------------------------
# 4. Run each step, capturing output + exit code, without stopping the whole
#    run just because one step fails - partial real data beats none.
# ---------------------------------------------------------------------------
run_step() {
    local name="$1"; shift
    local log_file="$RESULTS_DIR/${name}.log"
    log "--- running: $name ---"
    log "  command: $*"
    if "$@" > "$log_file" 2>&1; then
        log "  OK (see ${name}.log)"
        echo "$name: OK" >> "$SUMMARY"
    else
        local code=$?
        log "  FAILED (exit $code, see ${name}.log)"
        echo "$name: FAILED (exit $code)" >> "$SUMMARY"
    fi
}

run_step "01_ruff" ruff check .
run_step "02_pytest_full_suite" env PYTHONPATH=python python -m pytest python/tests -q
run_step "03_godot_gdscript_tests" "$GODOT_EXECUTABLE" --headless --path "$REPO_ROOT" --script res://tests/run_tests.gd
run_step "04_validate_runtime" python -m sandboxai validate-runtime \
    --godot-executable "$GODOT_EXECUTABLE" --project-path "$REPO_ROOT" --json
run_step "05_smoke_test" python -m sandboxai smoke-test --device auto
run_step "06_benchmark_suites" python -m sandboxai benchmark-suites \
    --godot-executable "$GODOT_EXECUTABLE" --project-path "$REPO_ROOT" \
    --output-dir "$RESULTS_DIR/benchmark-suites"
run_step "07_benchmark_worker_sweep" python -m sandboxai benchmark \
    --godot-executable "$GODOT_EXECUTABLE" --project-path "$REPO_ROOT" \
    --worker-counts 1,2,4,8 --output-dir "$RESULTS_DIR/benchmark-worker-sweep"

log ""
log "=== DONE ==="
log "All results, logs and raw benchmark JSON are under:"
log "  $RESULTS_DIR"
log "Zip/tar that whole folder and send it back (or paste SUMMARY.txt plus"
log "any FAILED step's .log) so the real numbers can replace the PLANNED"
log "roadmap entries in PROJECT.md - nothing gets written there from a guess."
cat "$SUMMARY"
