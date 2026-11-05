# run_local.ps1 - one-command launcher for IGCA on Windows. 
#
#   CPU machine : creates .venv, installs the CPU stack, runs the test suite and the
#                 tiny-random Falcon demo (exact zero-gate identity + grounding in ~1 min).
#   CUDA machine: installs the CUDA stack and runs falcon_local.py on the model you name.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File run_local.ps1
#   powershell -ExecutionPolicy Bypass -File run_local.ps1 -Model tiiuae/falcon-7b -Data flickr8k -Steps 2000
#   powershell -ExecutionPolicy Bypass -File run_local.ps1 -Model tiiuae/falcon-40b -Data flickr8k -Steps 59500 -Variant ungated
param(
    [string]$Model = "",
    [ValidateSet("synthetic", "flickr8k")][string]$Data = "synthetic",
    [ValidateSet("gated", "ungated")][string]$Variant = "gated",
    [int]$Steps = -1,
    [string]$Venv = ".venv"
)

$ErrorActionPreference = "Stop"
Write-Host "======================================================================"
Write-Host " IGCA - zero-gated cross-attention - local launcher"
Write-Host "======================================================================"

$hasNvidia = $false
try { if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi -L | Out-Host; $hasNvidia = $true } } catch {}
if (-not $hasNvidia) { Write-Host "No NVIDIA GPU detected -> CPU verification mode." }

if (-not (Test-Path "$Venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment at $Venv ..."
    py -3 -m venv $Venv
}
$py = "$Venv\Scripts\python.exe"
& $py -m pip install -q --upgrade pip
if ($hasNvidia) {
    & $py -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121
    & $py -m pip install -q -r requirements.txt bitsandbytes
} else {
    & $py -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    & $py -m pip install -q -r requirements.txt
}

if (-not $hasNvidia -and $Model -eq "") {
    Write-Host "`n--- test suite ---"
    & $py -m pytest -q
    Write-Host "`n--- tiny-random Falcon: gated (IGCA) ---"
    & $py falcon_local.py --tiny-random --variant gated --out-dir runs\tiny_gated
    Write-Host "`n--- tiny-random Falcon: un-gated baseline ---"
    & $py falcon_local.py --tiny-random --variant ungated --out-dir runs\tiny_ungated
    exit $LASTEXITCODE
}

if ($Model -eq "") { Write-Host "Pass -Model <hf-id> for a GPU run."; exit 1 }
$cmd = @("falcon_local.py", "--model", $Model, "--data", $Data, "--variant", $Variant)
if ($Steps -ge 0) { $cmd += @("--steps", "$Steps") }
Write-Host "Running: $py $($cmd -join ' ')"
& $py @cmd
$code = $LASTEXITCODE
switch ($code) {
    0 { Write-Host "RESULT: run completed. See runs\<name>\run_manifest.json." }
    2 { Write-Host "RESULT: zero-gate identity FAILED - fix wiring before training." }
    3 { Write-Host "RESULT: model could not be loaded on this machine (no fallback attempted)." }
    default { Write-Host "RESULT: exited with code $code." }
}
exit $code
