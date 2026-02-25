from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("rag_service")


@dataclass(frozen=True)
class MineruRunResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    output_dir: Path


class MineruUnavailableError(RuntimeError):
    pass


class MineruRunError(RuntimeError):
    pass


def resolve_mineru_python() -> Path:
    env_val = os.getenv("MINERU_PYTHON")
    if not env_val:
        raise MineruUnavailableError("MINERU_PYTHON environment variable is required")
    return Path(env_val)


def check_mineru_env(mineru_python: Path) -> tuple[bool, str]:
    if not mineru_python.exists():
        return False, f"MinerU python not found: {mineru_python}"

    proc = subprocess.run(
        [str(mineru_python), "-c", "import dill, shapely, pyclipper, torch, transformers; print('ok')"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err
    return True, "ok"


def build_mineru_command(mineru_python: Path, pdf_path: Path, output_dir: Path, *, text_only: bool = False) -> list[str]:
    cmd = [
        str(mineru_python),
        "-m",
        "app.mineru_offline_cli",
        "--path",
        str(pdf_path.resolve()),
        "--output",
        str(output_dir.resolve()),
        "-b",
        "pipeline",
        "-d",
        "cpu",
    ]
    if text_only:
        cmd += ["-m", "txt", "-f", "false", "-t", "false"]
    return cmd


def run_mineru(mineru_python: Path, pdf_path: Path, output_dir: Path, *, timeout_s: int | None = None, text_only: bool = False) -> MineruRunResult:
    cmd = build_mineru_command(mineru_python, pdf_path, output_dir, text_only=text_only)
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["DISABLE_MINERU_LLM"] = "1"

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    duration_s = round(time.perf_counter() - start, 3)

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    pdf_size_bytes = pdf_path.stat().st_size if pdf_path.exists() else None

    logger.info("mineru_python", extra={"mineru_python": str(mineru_python)})
    logger.info("mineru_cmd", extra={"cmd": cmd})
    logger.info("mineru_output_dir", extra={"path": str(output_dir)})
    logger.info("mineru_exec_stats", extra={"path": str(pdf_path), "duration_s": duration_s, "pdf_size_bytes": pdf_size_bytes})
    logger.info("mineru_returncode", extra={"returncode": proc.returncode, "path": str(pdf_path)})
    logger.info("mineru_stdout_tail", extra={"stdout_tail": stdout[-4000:], "path": str(pdf_path)})
    if stderr:
        logger.info("mineru_stderr_tail", extra={"stderr_tail": stderr[-4000:], "path": str(pdf_path)})

    if proc.returncode != 0 or _stderr_indicates_failure(stderr):
        raise MineruRunError(f"mineru failed\ncmd={' '.join(cmd)}\nstderr={stderr[-4000:]}")

    return MineruRunResult(cmd=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr, output_dir=output_dir)


def _stderr_indicates_failure(stderr: str) -> bool:
    lowered = (stderr or "").lower()
    return any(sig in lowered for sig in ["traceback", "modulenotfounderror", "importerror", "error |", "error"])


def extract_missing_module(stderr: str) -> str | None:
    m = re.search(r'No module named ["\']([^"\']+)["\']', stderr or "")
    return m.group(1) if m else None
