# ============================================================================
# DIAD2024 corrected re-evaluation (time-ordered windows, capture-grouped split,
# ground-truth labels). Runs all stages and logs to experiments\reeval_diad\logs.
#
# Usage (from any folder):
#   $env:DIAD_RAW = "<path>\cic_iot_diad_2024"   # raw CIC IoT-DIAD 2024 CSVs
#   powershell -ExecutionPolicy Bypass -File scripts\reeval_diad\run_reeval_diad.ps1
#   ... -Smoke                 quick end-to-end check (a few minutes)
#   ... -Seeds 42,123,2026     choose seeds (default: 42,123,2026)
#   ... -SkipData -SkipDetector  reuse data / detectors already built
# Follow progress live in another window:
#   Get-Content experiments\reeval_diad\logs\run.log -Wait
# ============================================================================
param(
    [int[]]$Seeds = @(42, 123, 2026),
    [switch]$Smoke,
    [switch]$SkipData,
    [switch]$SkipDetector,
    [switch]$Cpu          # run on the CPU (avoids the GPU driver's large memory reservation)
)
$ErrorActionPreference = "Stop"
$Scripts = $PSScriptRoot
$Root    = Join-Path (Split-Path (Split-Path $PSScriptRoot)) "experiments\reeval_diad"
if ($Smoke) { $Root = "$Root`_smoke"; $Seeds = @(42) }
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$RunLog = Join-Path $Logs "run.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
if ($Cpu) { $env:CUDA_VISIBLE_DEVICES = "-1" } else { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }

function Stage([string]$Name, [string[]]$PyArgs) {
    $t0 = Get-Date
    "==== [$($t0.ToString('yyyy-MM-dd HH:mm:ss'))] START $Name" | Tee-Object -FilePath $RunLog -Append
    # PowerShell 5.1 turns native stderr into errors under "Stop"; relax it for python.
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & python @PyArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $RunLog -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) {
        "==== FAILED $Name (exit $code)" | Tee-Object -FilePath $RunLog -Append
        exit $code
    }
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    "==== DONE  $Name in $mins min" | Tee-Object -FilePath $RunLog -Append
}

$SmokeFlag = @(); if ($Smoke) { $SmokeFlag = @("--smoke") }
"#### Re-evaluation started $(Get-Date)  root=$Root  seeds=$($Seeds -join ',')  device=$(if ($Cpu) {'CPU'} else {'GPU'})" | Tee-Object -FilePath $RunLog -Append

if (-not $SkipData)     { Stage "1 build dataset"   (@("$Scripts\build_dataset.py", "--out", "$Root\data") + $SmokeFlag) }
if (-not $SkipDetector) {
    $ep = if ($Smoke) { "2" } else { "20" }
    Stage "2 train detectors" @("$Scripts\train_detectors.py", "--root", $Root, "--epochs", $ep)
}
foreach ($s in $Seeds) {
    Stage "3 train agents (seed $s)" (@("$Scripts\train_agents.py", "--root", $Root, "--seed", "$s") + $SmokeFlag)
    Stage "4 evaluate (seed $s)"     @("$Scripts\evaluate.py", "--root", $Root, "--seed", "$s")
}
if (-not $Smoke) { Stage "5 summarise all seeds" @("$Scripts\summarise.py", $Root) }
"#### Re-evaluation finished $(Get-Date)" | Tee-Object -FilePath $RunLog -Append
