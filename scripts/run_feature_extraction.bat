@echo off
:: Equivalent of run_feature_extraction.sh
:: Exits on error (equivalent of set -euo pipefail)

:: Change to project root (parent of scripts/)
pushd "%~dp0\.."

python src/main.py --stage feature %*
if %ERRORLEVEL% neq 0 ( popd & exit /b %ERRORLEVEL% )

popd
