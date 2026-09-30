#!/usr/bin/env bash
# SandboxAI CPU-vs-CUDA throughput comparison at a fixed, known-good
# environment/worker configuration.
#
# Why this exists: earlier hand-timed runs suggested CPU training was about
# twice as fast as CUDA on this hardware, which is plausible - the policy
# network is tiny (84 -> 128 -> 128) and at that size the per-step kernel
# launch and host<->device round trip cost more than the arithmetic they
# save. "Plausible" is not "measured", though, and PROJECT.md section 16
# forbids reporting estimates as measurements. This script produces the
# measurement.
#
# It runs the same training three ways, changing nothing but the device
# placement:
#
#   1. cpu     --device cpu   --inference-device cpu    (everything on CPU)
#   2. hybrid  --device cuda  --inference-device cpu    (PPO updates on the
#              GPU, per-step rollout inference on the CPU - this is what
#              --inference-device was added for)
#   3. cuda    --device cuda  --inference-device cuda   (everything on GPU)
#
# Same seed, same environment count, same worker count, same step budget,
# so wall-clock time is the only thing that differs. Each run writes into
# its own run directory; nothing is overwritten and nothing is averaged
# away.
#
# Usage:
#   tools/wsl/compare_device_throughput.sh [path/to/godot]
#   STEPS=100000 ENV_COUNT=48 ENV_WORKERS=4 \
#     tools/wsl/compare_device_throughput.sh /path/to/godot
#
# Environment overrides (all optional):
#   STEPS        total training steps per run   (default 100000)
#   ENV_COUNT    parallel environments          (default 48)
#   ENV_WORKERS  environment worker processes   (default 4)
#   SEED         RNG seed, shared by all runs   (default 1234)
#   MODES        which runs to do               (default "cpu hybrid cuda")
#
# The Godot lookup, the OneDrive placeholder caveat and the "never guess a
# path" rule are the same as run_full_validation.sh; see that script's
# header for the details.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT" || { echo "[SandboxAI] Could not locate the repository root."; exit 1; }

STEPS="${STEPS:-100000}"
ENV_COUNT="${ENV_COUNT:-48}"
ENV_WORKERS="${ENV_WORKERS:-4}"
SEED="${SEED:-1234}"
MODES="${MODES:-cpu hybrid cuda}"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="$REPO_ROOT/training/device-comparison/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/SUMMARY.txt"
: > "$SUMMARY"

log() { echo "[SandboxAI] $*" | tee -a "$SUMMARY"; }

# ---------------------------------------------------------------------------
# 1. Resolve the Godot executable (same contract as run_full_validation.sh).
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
    else
        log "ERROR: found ${#CANDIDATES[@]} candidates; refusing to guess."
        for candidate in "${CANDIDATES[@]:-}"; do log "  - $candidate"; done
        log "Re-run as: tools/wsl/compare_device_throughput.sh /path/to/godot"
        exit 1
    fi
fi

if [ ! -f "$GODOT_EXECUTABLE" ]; then
    log "ERROR: '$GODOT_EXECUTABLE' does not exist or is not a regular file."
    log "If it lives in OneDrive, open it once from Windows so it is not a"
    log "cloud-only placeholder, then try again."
    exit 1
fi
chmod +x "$GODOT_EXECUTABLE" 2>/dev/null || true
log "Using Godot executable: $GODOT_EXECUTABLE"
"$GODOT_EXECUTABLE" --version > "$RESULTS_DIR/godot_version.txt" 2>&1
log "Godot --version: $(cat "$RESULTS_DIR/godot_version.txt")"

# ---------------------------------------------------------------------------
# 2. Record what the numbers were measured on. A throughput figure without
#    the machine behind it is not reproducible.
# ---------------------------------------------------------------------------
{
    echo "=== uname -a ==="; uname -a
    echo; echo "=== CPU ==="; lscpu 2>/dev/null | head -20
    echo; echo "=== RAM ==="; free -h 2>/dev/null
    echo; echo "=== nvidia-smi ==="; nvidia-smi 2>&1 | head -15
    echo; echo "=== torch ==="
    python -c 'import torch; print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")' 2>&1
} > "$RESULTS_DIR/system_snapshot.txt" 2>&1
log "System snapshot written to system_snapshot.txt"

CUDA_AVAILABLE="$(python -c 'import torch; print("yes" if torch.cuda.is_available() else "no")' 2>/dev/null || echo "unknown")"
log "CUDA available to torch: $CUDA_AVAILABLE"

log ""
log "Configuration: steps=$STEPS env-count=$ENV_COUNT env-workers=$ENV_WORKERS seed=$SEED"
log ""

# ---------------------------------------------------------------------------
# 3. The runs.
# ---------------------------------------------------------------------------
declare -A ELAPSED
declare -A STATUS

run_mode() {
    local mode="$1" device inference
    case "$mode" in
        cpu)    device="cpu";  inference="cpu"  ;;
        hybrid) device="cuda"; inference="cpu"  ;;
        cuda)   device="cuda"; inference="cuda" ;;
        *) log "Unknown mode '$mode', skipping."; return ;;
    esac

    if [ "$device" = "cuda" ] && [ "$CUDA_AVAILABLE" != "yes" ]; then
        log "SKIP $mode: torch reports no CUDA device."
        STATUS[$mode]="skipped (no CUDA)"
        ELAPSED[$mode]="-"
        return
    fi

    local run_id="devcmp-${mode}-${TIMESTAMP}"
    local logfile="$RESULTS_DIR/${mode}.log"
    log "RUN $mode: --device $device --inference-device $inference"

    local start end
    start="$(date +%s)"
    sandboxai train \
        --godot-executable "$GODOT_EXECUTABLE" \
        --project-path "$REPO_ROOT" \
        --steps "$STEPS" \
        --env-count "$ENV_COUNT" \
        --env-workers "$ENV_WORKERS" \
        --seed "$SEED" \
        --device "$device" \
        --inference-device "$inference" \
        --run-id "$run_id" \
        > "$logfile" 2>&1
    local code=$?
    end="$(date +%s)"

    ELAPSED[$mode]="$((end - start))"
    if [ "$code" -eq 0 ]; then
        STATUS[$mode]="ok"
        log "     done in ${ELAPSED[$mode]}s"
    else
        STATUS[$mode]="FAILED (exit $code)"
        log "     FAILED after ${ELAPSED[$mode]}s (exit $code) - see ${mode}.log"
        log "     last lines:"
        tail -15 "$logfile" | sed 's/^/       /' | tee -a "$SUMMARY"
    fi
}

for mode in $MODES; do
    run_mode "$mode"
done

# ---------------------------------------------------------------------------
# 4. The table.
# ---------------------------------------------------------------------------
log ""
log "=========================================================="
log "Results: $STEPS steps, $ENV_COUNT envs, $ENV_WORKERS workers"
log "=========================================================="
printf "%-8s %-12s %-14s %s\n" "mode" "wall clock" "steps/second" "status" | tee -a "$SUMMARY"
for mode in $MODES; do
    secs="${ELAPSED[$mode]:-"-"}"
    # A rate is only meaningful for a run that actually completed its step
    # budget. Printing "100000 steps / 3 seconds" for a run that died three
    # seconds in would be a fabricated measurement.
    if [ "${STATUS[$mode]:-}" = "ok" ] && [ "$secs" != "-" ] && [ "$secs" -gt 0 ] 2>/dev/null; then
        rate="$(python -c "print(f'{$STEPS/$secs:.1f}')" 2>/dev/null || echo "-")"
        printf "%-8s %-12s %-14s %s\n" "$mode" "${secs}s" "$rate" "${STATUS[$mode]}" | tee -a "$SUMMARY"
    else
        printf "%-8s %-12s %-14s %s\n" "$mode" "-" "-" "${STATUS[$mode]:-not run}" | tee -a "$SUMMARY"
    fi
done

log ""
log "Per-run training output: sandboxai inspect-runs --root training --limit 3"
log "Everything from this comparison: $RESULTS_DIR"
log ""
log "Send SUMMARY.txt and system_snapshot.txt for analysis. If a run failed,"
log "its full log is in the matching <mode>.log next to them."
