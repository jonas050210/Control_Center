# SandboxAI M0 bootstrap (Windows 11, Python 3.11, RTX 4060 Ti)
#
# One-time setup from a FRESH CLONE, run from the repo root:
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
#
# What it does:
#   1. checks the Python 3.11 launcher
#   2. creates .venv (if missing) and upgrades pip
#   3. installs PyTorch with CUDA 12.4 (if missing) + pinned requirements
#   4. clones the pinned godot_rl_agents_examples + SandboxAI overlay
#   5. downloads + installs the Godot 4.3 export templates (if missing, ~1.2 GB)
#   6. imports assets and exports build\fps_windows.exe / build\virtualcamera_windows.exe
#
# Skips any step that is already done; safe to re-run.
# Requires the Godot 4.3 executables in the repo root (already the case) and
# git on PATH.  Use -SkipTemplates to skip step 5 (e.g. manual template install).
#
# NOTE: we deliberately use $ErrorActionPreference = 'Continue' (the default):
# with 'Stop', stderr output of native tools (python/pip/py) can abort the
# script on EXPECTED failures (e.g. the torch pre-check before installing).
# Every step instead checks $LASTEXITCODE / Test-Path explicitly.

param([switch]$SkipTemplates)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Info($msg)  { Write-Host "[setup] $msg"  -ForegroundColor Cyan }
function Fail($msg)  { Write-Host "[setup] $msg"  -ForegroundColor Red; exit 1 }
function Step($msg)  { Write-Host ""; Write-Host "== $msg ==" -ForegroundColor White }

# ---------------------------------------------------------------- 1. Python
Step "Python 3.11"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Fail "Windows Python launcher 'py' not found. Install Python 3.11 from https://www.python.org/downloads/ (tick 'Add to PATH')."
}
$pyVer = py -3.11 --version
if ($LASTEXITCODE -ne 0) { Fail "Python 3.11 is not installed (py -3.11 failed). Install it from https://www.python.org/downloads/" }
Info "found: $pyVer"

# ------------------------------------------------------------------ 2. venv
Step "virtual environment"
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Info "creating .venv"
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "could not create .venv" }
} else {
    Info ".venv already exists"
}
Info "upgrading pip"
& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed (check your internet connection)" }

# ------------------------------------------------------- 3a. torch (cu124)
Step "PyTorch (CUDA 12.4)"
& $py -c "import torch; assert '+cu124' in torch.__version__, torch.__version__"
if ($LASTEXITCODE -ne 0) {
    Info "installing torch with CUDA 12.4 (~2.5 GB download, one time)"
    & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { Fail "torch install failed - check your internet connection" }
} else {
    $v = & $py -c "import torch; print(torch.__version__)"
    Info "already installed: torch $v"
}

# ------------------------------------------------------- 3b. pinned deps
Step "pinned requirements (data pipeline, godot-rl, sb3, gymnasium)"
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "requirements install failed" }

# -------------------------------------------------- 4. examples (pinned)
Step "godot_rl_agents_examples (pinned commit + SandboxAI overlay)"
& $py feasibility\setup_examples.py
if ($LASTEXITCODE -ne 0) { Fail "setup_examples.py failed - is git installed and on PATH?" }

# ------------------------------------------- 5. Godot export templates
Step "Godot 4.3 export templates"
$tplDir = Join-Path $env:APPDATA "Godot\export_templates\4.3.stable"
$tplExe = Join-Path $tplDir "windows_release_x86_64.exe"
if ($SkipTemplates) {
    Info "skipped (-SkipTemplates)"
} elseif (Test-Path $tplExe) {
    Info "already installed"
} else {
    Info "downloading Godot 4.3.stable export templates (~1.2 GB, one time)"
    $zip = Join-Path $env:TEMP "godot_4.3_export_templates.zip"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest `
            -Uri "https://github.com/godotengine/godot/releases/download/4.3-stable/Godot_v4.3-stable_export_templates.tpz" `
            -OutFile $zip -ErrorAction Stop
    } catch {
        Fail ("template download failed: " + $_.Exception.Message + "`nFallback: open the Godot editor -> Editor/Manage Export Templates... -> Download and Install, then re-run this script.")
    }
    Info "extracting (takes a minute)"
    $tmp = Join-Path $env:TEMP "godot_4.3_templates_extract"
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    New-Item -ItemType Directory -Force -Path $tplDir | Out-Null
    Copy-Item "$tmp\templates\*" $tplDir -Force
    Remove-Item -Recurse -Force $tmp, $zip -ErrorAction SilentlyContinue
    if (-not (Test-Path $tplExe)) {
        Fail "template extraction did not produce windows_release_x86_64.exe - install manually via the Godot editor (Editor/Manage Export Templates...), then re-run."
    }
    Info "installed to $tplDir"
}

# ------------------------------------------------------- 6. import/export
Step "import assets + export environments"
& $py feasibility\export_envs.py
if ($LASTEXITCODE -ne 0) { Fail "export_envs.py failed (see output above)" }

# ------------------------------------------------------------------ done
Step "GPU"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} else {
    Info "nvidia-smi not on PATH - GPU/VRAM reporting in benchmarks will be skipped"
}

Write-Host ""
Write-Host "Bootstrap complete. Benchmarks (run from the repo root, venv active):" -ForegroundColor Green
Write-Host "  .venv\Scripts\activate"
Write-Host "  python feasibility\benchmark_env.py --env_path build\fps_windows.exe --speedup 30 --seconds 30"
Write-Host "  python feasibility\benchmark_env.py --env_path build\fps_windows.exe --speedup 30 --n_parallel 4 --seconds 30"
Write-Host "  python feasibility\train_ppo.py --env_path build\fps_windows.exe --timesteps 50000 --n_parallel 2"
Write-Host "  python feasibility\benchmark_env.py --env_path build\virtualcamera_windows.exe --viz --speedup 30 --seconds 30"
Write-Host "  python feasibility\train_ppo.py --env_path build\virtualcamera_windows.exe --viz --timesteps 20000"
Write-Host ""
Write-Host "Fill the results table in feasibility\README.md afterwards (see PROJECT.md)." -ForegroundColor Green
