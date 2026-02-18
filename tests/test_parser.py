from pathlib import Path

from app.parser import MineruCliCaps, RAGAnythingParser, log_dependency_compatibility
from app.schemas import ParsedElement


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_logs_fallback_reason_when_mineru_empty(tmp_path, monkeypatch, caplog):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(parser, "_run_mineru_subprocess", lambda *_args, **_kwargs: {"elements": []})

    with caplog.at_level("INFO", logger="rag_service"):
        elements, mode = parser.parse_file_with_mode("sample.txt", sample)

    assert mode == "fallback"
    assert elements == [ParsedElement(element_index=0, type="text", content="hello", meta={"fallback": True})]
    assert any("parser_fallback_used" in rec.message for rec in caplog.records)


def test_logs_warning_when_mineru_throws(tmp_path, monkeypatch, caplog):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(parser, "_run_mineru_subprocess", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with caplog.at_level("WARNING", logger="rag_service"):
        elements, reason = parser._parse_with_mineru(sample)

    assert elements is None
    assert reason.startswith("exception:")
    assert any("mineru_execution_error" in rec.message for rec in caplog.records)


def test_detect_cli_caps_without_json(monkeypatch):
    def fake_run(cmd, capture_output, text, timeout, check):
        if "parse_doc" in cmd:
            return _Proc(stdout="Usage: parse_doc [OPTIONS]\n  --output-dir TEXT\n  --disable-image\n")
        return _Proc(stdout="Usage: client [COMMAND]\n")

    monkeypatch.setattr("app.parser.subprocess.run", fake_run)
    RAGAnythingParser._detect_mineru_cli_caps.cache_clear()
    caps = RAGAnythingParser._detect_mineru_cli_caps("python")

    assert caps.supports_json is False
    assert caps.output_dir_flag == "--output-dir"


def test_build_command_without_json_flag():
    caps = MineruCliCaps(
        supports_json=False,
        output_dir_flag="--output-dir",
        disable_image_flag="--disable-image",
        disable_table_flag=None,
        disable_equation_flag=None,
    )
    cmd = RAGAnythingParser._build_mineru_command(
        path=Path("a.pdf"),
        text_only=True,
        output_dir=Path("out"),
        caps=caps,
    )
    assert "--json" not in cmd
    assert "--output-dir" in cmd
    assert "--disable-image" in cmd


def test_mineru_subprocess_reads_output_dir_artifacts(tmp_path, monkeypatch):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4")

    caps = MineruCliCaps(
        supports_json=False,
        output_dir_flag="--output-dir",
        disable_image_flag=None,
        disable_table_flag=None,
        disable_equation_flag=None,
    )
    monkeypatch.setattr(parser, "_detect_mineru_cli_caps", lambda _python: caps)

    def fake_run(cmd, capture_output, text, timeout, check):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        (out_dir / "result.json").write_text('{"elements":[{"type":"text","text":"ok"}]}', encoding="utf-8")
        return _Proc(returncode=0, stdout="")

    monkeypatch.setattr("app.parser.subprocess.run", fake_run)
    data = parser._run_mineru_subprocess(sample, text_only=False)
    assert data["elements"][0]["text"] == "ok"


def test_dependency_mismatch_retries_text_only(tmp_path, monkeypatch, caplog):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    calls: list[bool] = []

    def fake_run(path, *, text_only: bool):
        calls.append(text_only)
        if not text_only:
            raise RuntimeError("UnimerMBartForCausalLM.forward() got an unexpected keyword argument 'cache_position'")
        return {"elements": [{"type": "text", "text": "retry ok"}]}

    monkeypatch.setattr(parser, "_run_mineru_subprocess", fake_run)

    with caplog.at_level("WARNING", logger="rag_service"):
        elements, reason = parser._parse_with_mineru(sample)

    assert calls == [False, True]
    assert reason == "ok_text_only_retry"
    assert elements == [{"type": "text", "text": "retry ok"}]
    assert any("dependency_mismatch" in rec.message for rec in caplog.records)
    assert any("parser_degraded_mode_used" in rec.message for rec in caplog.records)


def test_dependency_compatibility_logs_mismatch(monkeypatch, caplog):
    versions = {
        "transformers": "4.57.0",
        "torch": "2.10.0",
        "mineru": "2.0.6",
        "raganything": "1.2.9",
        "sentence-transformers": "2.2.2",
        "huggingface-hub": "0.17.3",
        "accelerate": None,
    }

    monkeypatch.setattr("app.parser._safe_version", lambda pkg: versions.get(pkg))

    with caplog.at_level("ERROR", logger="rag_service"):
        log_dependency_compatibility()

    assert any("dependency_mismatch" in rec.message for rec in caplog.records)
