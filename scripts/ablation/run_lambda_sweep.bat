@echo off
REM ============================================================================
REM Lambda ablation sweep for Strategy B
REM
REM Runs train_rl_multiagent_StrategyB.py for 3 new values of LAMBDA_FUZZY:
REM   0.00, 0.10, 0.50
REM (the 0.25 cell is reused from the canonical results_StrategyB/ folder)
REM
REM Each run writes to experiments/results_StrategyB_lam{value}/{dataset}/results.json
REM
REM Default behaviour: CICIoT2023 only (4 runs). To also run DIAD2024, change
REM the DATASET variable below to "all" or run this script twice.
REM
REM Estimated wall-clock time: ~4.5 hours for CICIoT2023 alone on a single GPU
REM (3 runs x ~90 min each, which you measured). Double it for both datasets.
REM We skip lambda=0.25 because the canonical results_StrategyB/ folder
REM already holds that cell.
REM ============================================================================

setlocal enabledelayedexpansion

REM Change to the project root so that paths in train_rl_multiagent_StrategyB
REM (PROJECT = ../.. relative to the script) resolve correctly.
cd /d "%~dp0\..\.."

REM Which dataset to sweep over. Options: ciciot2023 | diad2024 | all
set DATASET=ciciot2023

REM The lambda values to test. 0.00 disables the fuzzy shaping term
REM entirely and reproduces the B3 (FedProx+MOSAR only) behaviour.
REM
REM NOTE: lambda=0.25 is the default used by the canonical
REM experiments/results_StrategyB/ run, so we do NOT re-run it here.
REM The aggregator (aggregate_lambda_results.py) automatically reads
REM the 0.25 cell from the canonical folder.
set LAMBDAS=0.00 0.10 0.50

set LOG=scripts\ablation\lambda_sweep.log
echo =============================================================== > "%LOG%"
echo Lambda sweep started at %DATE% %TIME% >> "%LOG%"
echo Dataset: %DATASET% >> "%LOG%"
echo Lambdas: %LAMBDAS% >> "%LOG%"
echo =============================================================== >> "%LOG%"

for %%L in (%LAMBDAS%) do (
    set LAM=%%L
    REM Build the suffix "_lam0.10" (strip the "0." if you prefer; we keep
    REM it so the folder name is unambiguous).
    set SUFFIX=_lam%%L

    echo.
    echo =============================================================
    echo   RUNNING: lambda = %%L, dataset = %DATASET%, suffix = !SUFFIX!
    echo =============================================================
    echo [%DATE% %TIME%] Starting lambda=%%L >> "%LOG%"

    python scripts\training\train_rl_multiagent_StrategyB.py ^
        --dataset %DATASET% ^
        --lambda-fuzzy %%L ^
        --suffix !SUFFIX!

    if errorlevel 1 (
        echo [%DATE% %TIME%] FAILED lambda=%%L with errorlevel %errorlevel% >> "%LOG%"
        echo ERROR: run for lambda=%%L failed. See above.
    ) else (
        echo [%DATE% %TIME%] OK lambda=%%L >> "%LOG%"
        echo SUCCESS: run for lambda=%%L completed.
    )
)

echo.
echo =============================================================
echo   Lambda sweep finished.
echo =============================================================
echo [%DATE% %TIME%] Sweep completed >> "%LOG%"
echo Log file: %LOG%
echo.
echo Next step: run
echo     python scripts\ablation\aggregate_lambda_results.py
echo to collect the results into a single CSV for the paper.

endlocal
