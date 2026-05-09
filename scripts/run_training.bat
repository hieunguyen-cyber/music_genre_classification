@echo off
:: Equivalent of run_training.sh
:: Exits on error (equivalent of set -euo pipefail)

:: Change to project root (parent of scripts/)
pushd "%~dp0\.."

python src/main.py --stage train %*
if %ERRORLEVEL% neq 0 ( popd & exit /b %ERRORLEVEL% )

popd
