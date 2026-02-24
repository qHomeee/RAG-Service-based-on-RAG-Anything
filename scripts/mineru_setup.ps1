$ErrorActionPreference = "Stop"

if (-Not (Test-Path ".venv-mineru")) {
  python -m venv .venv-mineru
}

.\.venv-mineru\Scripts\python.exe -m pip install -U pip
.\.venv-mineru\Scripts\pip.exe install -r requirements-mineru.txt

.\.venv-mineru\Scripts\python.exe -c "from app.mineru_runner import resolve_mineru_python, check_mineru_ready; p=resolve_mineru_python(); ok, msg=check_mineru_ready(p); print({'ok': ok, 'mineru_python': str(p), 'detail': msg})"
