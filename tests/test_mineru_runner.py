import subprocess
from pathlib import Path

import pytest

from app.mineru_runner import MineruRunError, run_mineru


class _Proc:
    returncode = 0
    stdout = "ok"
    stderr = ""


def test_run_mineru_passes_configured_timeout(tmp_path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "parsed"

    def fake_run(cmd, *, capture_output, text, check, env, timeout):
        assert timeout == 7200
        return _Proc()

    monkeypatch.setattr("app.mineru_runner.subprocess.run", fake_run)

    result = run_mineru(
        Path("python"),
        pdf_path,
        output_dir,
        timeout_s=7200,
    )

    assert result.returncode == 0


def test_run_mineru_raises_clear_error_on_timeout(tmp_path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "parsed"

    def fake_run(cmd, *, capture_output, text, check, env, timeout):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr("app.mineru_runner.subprocess.run", fake_run)

    with pytest.raises(MineruRunError, match="timed out after 30s"):
        run_mineru(
            Path("python"),
            pdf_path,
            output_dir,
            timeout_s=30,
        )
