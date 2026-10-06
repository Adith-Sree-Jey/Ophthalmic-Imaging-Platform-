$dotVenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if ($env:OPHTHALMIC_PYTHON) {
  $python = $env:OPHTHALMIC_PYTHON
} elseif (Test-Path $dotVenvPython) {
  $python = $dotVenvPython
} elseif (Test-Path $venvPython) {
  $python = $venvPython
} else {
  $python = "python"
}

Set-Location -LiteralPath $PSScriptRoot
& $python -m uvicorn main:app --host 0.0.0.0 --port 8000
