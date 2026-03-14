$ErrorActionPreference = "Stop"
python -m venv .venv-mineru
.\.venv-mineru\Scripts\python -m pip install --upgrade pip
.\.venv-mineru\Scripts\pip install -r requirements-mineru.txt
Write-Host "[setup_mineru] done"
