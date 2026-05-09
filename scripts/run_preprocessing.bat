@echo off
:: Equivalent of run_preprocessing.sh
:: Exits on error (equivalent of set -euo pipefail)

:: Change to project root (parent of scripts/)
pushd "%~dp0\.."

python src/main.py --stage preprocess %*
if %ERRORLEVEL% neq 0 ( popd & exit /b %ERRORLEVEL% )

popd
