$ErrorActionPreference = "Stop"

Write-Host "[doctor] core python:" 
python -c "import sys; print(sys.executable)"
python -c "import transformers, tokenizers; print('transformers', transformers.__version__, 'tokenizers', tokenizers.__version__)"

$mineruPython = if ($env:MINERU_PYTHON) { $env:MINERU_PYTHON } else { ".venv-mineru\Scripts\python.exe" }
Write-Host "[doctor] MINERU_PYTHON=$mineruPython"
if (-not (Test-Path $mineruPython)) {
  Write-Host "[doctor] ERROR: MINERU_PYTHON not found" -ForegroundColor Red
  exit 1
}
& $mineruPython -c "import sys; print(sys.executable)"
& $mineruPython -c "import mineru; print('mineru OK')"
Write-Host "[doctor] done"
