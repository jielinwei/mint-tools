@echo off
setlocal

if "%~1"=="" (
  echo Usage: run_audit_windows.bat SOURCE_DIR [OUTPUT_DIR]
  exit /b 1
)

set SOURCE_DIR=%~1
set OUTPUT_DIR=%~2

if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=.

python scripts\audit_mint_rt_dicom.py --source-dir "%SOURCE_DIR%" --output-dir "%OUTPUT_DIR%" --overwrite
python scripts\analyze_mint_folder_patterns.py --source-dir "%SOURCE_DIR%" --output-dir "%OUTPUT_DIR%\outputs"
