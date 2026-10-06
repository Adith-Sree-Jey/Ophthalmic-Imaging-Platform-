@echo off
setlocal
if defined OPHTHALMIC_PYTHON (
  set "PYTHON_EXE=%OPHTHALMIC_PYTHON%"
) else if exist "%~dp0.venv\Scripts\python.exe" (
  set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
  set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)

cd /d "%~dp0"
"%PYTHON_EXE%" -m uvicorn main:app --host 0.0.0.0 --port 8000
