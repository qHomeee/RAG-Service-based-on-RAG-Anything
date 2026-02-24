$ErrorActionPreference = "Stop"
if (-Not (Test-Path ".venv-mineru")) {
  python -m venv .venv-mineru
}
& .\.venv-mineru\Scripts\Activate.ps1
pip install -r requirements-mineru.txt
python -c "from app.parser import mineru_doctor; import os, json; print(json.dumps(mineru_doctor(r'.venv-mineru\Scripts\python.exe'), ensure_ascii=False, indent=2))"
