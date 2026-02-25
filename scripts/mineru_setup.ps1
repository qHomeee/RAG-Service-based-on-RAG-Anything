$ErrorActionPreference = "Stop"

if (-Not (Test-Path ".venv-mineru")) {
  python -m venv .venv-mineru
}

.\.venv-mineru\Scripts\python.exe -m pip install -U pip
.\.venv-mineru\Scripts\pip.exe install -r requirements-mineru.txt
$env:DISABLE_MINERU_LLM = "1"

.\.venv-mineru\Scripts\python.exe -c "from app.mineru_runner import resolve_mineru_python, check_mineru_env; p=resolve_mineru_python(); ok, msg=check_mineru_env(p); print({'ok': ok, 'mineru_python': str(p), 'detail': msg})"
