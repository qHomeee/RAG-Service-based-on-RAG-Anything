from pathlib import Path

from app.parser import MineruCliCaps, RAGAnythingParser, check_mineru_runtime, log_dependency_compatibility, mineru_run_and_validate, _module_to_package
from app.schemas import ParsedElement
from app.config import settings


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_logs_fallback_reason_when_mineru_empty(tmp_path, monkeypatch, caplog):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    monkeypatch.setattr("app.parser.check_mineru_env", lambda _py: (True, "ok"))
    monkeypatch.setattr("app.parser.resolve_mineru_python", lambda: Path("python"))
    monkeypatch.setattr(parser, "_run_mineru_subprocess", lambda *_args, **_kwargs: {"elements": []})

    with caplog.at_level("INFO", logger="rag_service"):
        elements, mode = parser.parse_file_with_mode("sample.txt", sample)

    assert mode == "fallback"
    assert elements == [ParsedElement(element_index=0, type="text", content="hello", meta={"fallback": True})]


def test_logs_warning_when_mineru_throws(tmp_path, monkeypatch, caplog):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    monkeypatch.setattr("app.parser.check_mineru_env", lambda _py: (True, "ok"))
    monkeypatch.setattr("app.parser.resolve_mineru_python", lambda: Path("python"))
    monkeypatch.setattr(parser, "_run_mineru_subprocess", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with caplog.at_level("WARNING", logger="rag_service"):
        elements, reason = parser._parse_with_mineru(sample)

    assert elements is None
    assert reason.startswith("exception:")
    assert any("mineru_execution_error" in rec.message for rec in caplog.records)


def test_detect_cli_caps_without_json(monkeypatch):
    def fake_run(cmd, capture_output, text, check=False, timeout=None, env=None):
        return _Proc(stdout="Usage: client [OPTIONS]\n  -p, --path TEXT\n  -o, --output TEXT\n  -m TEXT\n  -f TEXT\n  -t TEXT\n")

    monkeypatch.setattr("app.parser.subprocess.run", fake_run)
    RAGAnythingParser._detect_mineru_cli_caps.cache_clear()
    caps = RAGAnythingParser._detect_mineru_cli_caps("python")

    assert caps.path_flag == "--path"
    assert caps.output_dir_flag == "--output"


def test_build_command_uses_required_flags_and_no_parse_doc():
    caps = MineruCliCaps(
        path_flag="-p",
        output_dir_flag="-o",
        disable_image_flag="--disable-image",
        disable_table_flag=None,
        disable_equation_flag=None,
        mode_flag="-m",
        formula_flag="-f",
        table_flag="-t",
        backend_flag="-b",
        device_flag="-d",
    )
    cmd = RAGAnythingParser._build_mineru_command(
        path=Path("a.pdf"),
        text_only=True,
        output_dir=Path("out"),
        caps=caps,
    )
    assert "parse_doc" not in cmd
    assert "-p" in cmd
    assert "-o" in cmd
    assert "--disable-image" in cmd
    assert "-m" in cmd and "txt" in cmd
    assert "-f" in cmd and "false" in cmd
    assert "-t" in cmd and "false" in cmd


def test_mineru_subprocess_reads_output_dir_artifacts_recursively(tmp_path, monkeypatch):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4")

    caps = MineruCliCaps(
        path_flag="-p",
        output_dir_flag="-o",
        disable_image_flag=None,
        disable_table_flag=None,
        disable_equation_flag=None,
        mode_flag=None,
        formula_flag=None,
        table_flag=None,
        backend_flag=None,
        device_flag=None,
    )
    monkeypatch.setattr(parser, "_detect_mineru_cli_caps", lambda _python: caps)
    monkeypatch.setattr(settings, "storage_parsed", str(tmp_path / "parsed"))
    monkeypatch.setenv("MINERU_PYTHON", "python")

    def fake_run(cmd, capture_output, text, check=False, timeout=None, env=None):
        out_dir = Path(cmd[cmd.index("--output") + 1])
        nested = out_dir / "nested" / "level"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "result.json").write_text('{"elements":[{"type":"text","text":"ok"}]}', encoding="utf-8")
        return _Proc(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr("app.parser.subprocess.run", fake_run)
    data = parser._run_mineru_subprocess(sample, text_only=False, reindex=True)
    assert data["elements"][0]["text"] == "ok"


def test_dependency_mismatch_retries_text_only(tmp_path, monkeypatch, caplog):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    calls: list[bool] = []

    def fake_run(path, *, text_only: bool, reindex: bool = False):
        calls.append(text_only)
        if not text_only:
            raise RuntimeError("UnimerMBartForCausalLM.forward() got an unexpected keyword argument 'cache_position'")
        return {"elements": [{"type": "text", "text": "retry ok"}]}

    monkeypatch.setattr("app.parser.check_mineru_env", lambda _py: (True, "ok"))
    monkeypatch.setattr("app.parser.resolve_mineru_python", lambda: Path("python"))
    monkeypatch.setattr(parser, "_run_mineru_subprocess", fake_run)

    with caplog.at_level("WARNING", logger="rag_service"):
        elements, reason = parser._parse_with_mineru(sample)

    assert calls == [False, True]
    assert reason == "ok_text_only_retry"
    assert elements == [{"type": "text", "text": "retry ok"}]
    assert any("parser_degraded_mode_used" in rec.message for rec in caplog.records)


def test_nonzero_returncode_retries_once(tmp_path, monkeypatch):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    calls: list[bool] = []

    def fake_run(path, *, text_only: bool, reindex: bool = False):
        calls.append(text_only)
        if len(calls) == 1:
            raise RuntimeError("mineru_returncode=2\nstderr=boom")
        return {"elements": [{"type": "text", "text": "retry ok"}]}

    monkeypatch.setattr("app.parser.check_mineru_env", lambda _py: (True, "ok"))
    monkeypatch.setattr("app.parser.resolve_mineru_python", lambda: Path("python"))
    monkeypatch.setattr(parser, "_run_mineru_subprocess", fake_run)
    elements, reason = parser._parse_with_mineru(sample)
    assert calls == [False, True]
    assert reason == "ok_text_only_retry"
    assert elements == [{"type": "text", "text": "retry ok"}]


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


def test_mineru_uses_cached_output_when_not_reindex(tmp_path, monkeypatch):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4")

    parsed_root = tmp_path / "parsed"
    monkeypatch.setattr(settings, "storage_parsed", str(parsed_root))
    out_dir = parsed_root / sample.stem / "subdir"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.md").write_text("cached markdown", encoding="utf-8")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("subprocess must not run when cached artifacts exist")

    monkeypatch.setattr("app.parser.subprocess.run", fail_run)

    data = parser._run_mineru_subprocess(sample, text_only=False, reindex=False)
    assert data["elements"][0]["text"] == "cached markdown"


def test_check_mineru_runtime_logs_missing_dependency(monkeypatch, caplog):
    calls = []

    def fake_run(cmd, capture_output, text, check=False, timeout=None, env=None):
        calls.append(cmd)
        return _Proc(returncode=1, stderr="ModuleNotFoundError: No module named 'torch'")

    monkeypatch.setattr("app.parser.subprocess.run", fake_run)

    with caplog.at_level("WARNING", logger="rag_service"):
        result = check_mineru_runtime("python")

    assert result["ok"] is False
    assert result["missing"][0]["module"] == "torch"
    assert result["missing"][0]["pip"] == "torch"
    assert any("mineru_missing_dependency" in rec.message for rec in caplog.records)


def test_mineru_run_and_validate_fails_on_traceback_even_zero_returncode(tmp_path, monkeypatch):
    out_dir = tmp_path / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nested").mkdir()
    (out_dir / "nested" / "result.md").write_text("ok", encoding="utf-8")

    def fake_run(cmd, capture_output, text, check=False, timeout=None, env=None):
        return _Proc(returncode=0, stdout="", stderr="Traceback (most recent call last): boom")

    monkeypatch.setattr("app.parser.subprocess.run", fake_run)

    try:
        mineru_run_and_validate(cmd=["python", "-m", "mineru.cli.client"], output_dir=out_dir, source_path=tmp_path / "a.pdf")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "mineru_returncode=0" in str(exc)


def test_retry_runs_only_once_on_failure(tmp_path, monkeypatch):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4")

    calls: list[bool] = []

    def fake_run(path, *, text_only: bool, reindex: bool = False):
        calls.append(text_only)
        raise RuntimeError("mineru_returncode=1\nstderr=ERROR")

    monkeypatch.setattr("app.parser.check_mineru_env", lambda _py: (True, "ok"))
    monkeypatch.setattr("app.parser.resolve_mineru_python", lambda: Path("python"))
    monkeypatch.setattr(parser, "_run_mineru_subprocess", fake_run)
    elements, reason = parser._parse_with_mineru(sample)

    assert calls == [False, True]
    assert elements is None
    assert reason.startswith("dependency_mismatch:")


def test_check_mineru_runtime_checks_doclayout_import(monkeypatch):
    outputs = iter([
        _Proc(returncode=1, stderr="ModuleNotFoundError: No module named 'doclayout_yolo'"),
    ])

    def fake_run(cmd, capture_output, text, check=False, timeout=None, env=None):
        return next(outputs)

    monkeypatch.setattr("app.parser.subprocess.run", fake_run)
    result = check_mineru_runtime("python")
    assert result["missing"][0]["module"] == "doclayout_yolo"
    assert result["missing"][0]["pip"] == "doclayout-yolo"


def test_mineru_run_and_validate_detects_missing_module_mapping(tmp_path, monkeypatch, caplog):
    out_dir = tmp_path / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nested").mkdir()
    (out_dir / "nested" / "result.md").write_text("ok", encoding="utf-8")

    def fake_run(cmd, capture_output, text, check=False, timeout=None, env=None):
        return _Proc(returncode=0, stdout="", stderr="ModuleNotFoundError: No module named 'doclayout_yolo'")

    monkeypatch.setattr("app.parser.subprocess.run", fake_run)

    with caplog.at_level("ERROR", logger="rag_service"):
        try:
            mineru_run_and_validate(cmd=["python", "-m", "mineru.cli.client"], output_dir=out_dir, source_path=tmp_path / "a.pdf")
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass

    assert any("mineru_missing_dependency_detected" in rec.message for rec in caplog.records)


def test_module_to_package_mapping_known_cases():
    assert _module_to_package("fast_langdetect") == "fast-langdetect"
    assert _module_to_package("doclayout_yolo") == "doclayout-yolo"
    assert _module_to_package("rapid_table") == "rapid-table"


def test_mineru_run_and_validate_fails_on_module_not_found_zero_returncode(tmp_path, monkeypatch):
    out_dir = tmp_path / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nested").mkdir()
    (out_dir / "nested" / "result.md").write_text("ok", encoding="utf-8")

    def fake_run(cmd, capture_output, text, check=False, timeout=None, env=None):
        return _Proc(returncode=0, stdout="", stderr="ModuleNotFoundError: No module named 'ultralytics'")

    monkeypatch.setattr("app.parser.subprocess.run", fake_run)

    try:
        mineru_run_and_validate(cmd=["python", "-m", "mineru.cli.client"], output_dir=out_dir, source_path=tmp_path / "a.pdf")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "ultralytics" in str(exc)


def test_parser_skips_mineru_when_doctor_missing_deps(tmp_path, monkeypatch):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("fallback content", encoding="utf-8")

    monkeypatch.setattr("app.parser.check_mineru_env", lambda _py: (False, "ModuleNotFoundError: No module named ultralytics"))
    monkeypatch.setattr("app.parser.resolve_mineru_python", lambda: Path("python"))

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("MinerU should be skipped when doctor reports missing deps")

    monkeypatch.setattr(parser, "_run_mineru_subprocess", should_not_run)

    elems, mode = parser.parse_file_with_mode("sample.txt", sample)
    assert mode == "fallback"
    assert elems and elems[0].content == "fallback content"


def test_check_mineru_runtime_alias_kept(monkeypatch):
    calls = {"n": 0}

    def fake_run(cmd, capture_output, text, check=False, timeout=None, env=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Proc(returncode=0, stdout="ok", stderr="")
        return _Proc(returncode=0, stdout="2.0.6\n2.0\n4.35\n0.17\n0.1\n0.1\n8.4\n0.1\n", stderr="")

    monkeypatch.setattr("app.parser.subprocess.run", fake_run)
    assert check_mineru_runtime("python")["ok"] is True


def test_mineru_doctor_reports_rapid_table_mapping(monkeypatch):
    def fake_run(cmd, capture_output, text, check=False, timeout=None, env=None):
        return _Proc(returncode=1, stderr="ModuleNotFoundError: No module named 'rapid_table'")

    monkeypatch.setattr("app.parser.subprocess.run", fake_run)
    result = check_mineru_runtime("python")
    assert result["ok"] is False
    assert result["missing"][0]["module"] == "rapid_table"
    assert result["missing"][0]["pip"] == "rapid-table"


def test_parse_file_falls_back_when_mineru_stderr_module_not_found(tmp_path, monkeypatch):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("fallback content", encoding="utf-8")

    monkeypatch.setattr("app.parser.check_mineru_env", lambda _py: (True, "ok"))
    monkeypatch.setattr("app.parser.resolve_mineru_python", lambda: Path("python"))

    calls = {"n": 0}

    def fail_run(*_args, **_kwargs):
        calls["n"] += 1
        raise RuntimeError("mineru_returncode=0\nstderr=ModuleNotFoundError: No module named 'ultralytics'")

    monkeypatch.setattr(parser, "_run_mineru_subprocess", fail_run)
    elems, mode = parser.parse_file_with_mode("sample.txt", sample)

    assert calls["n"] == 0
    assert mode == "fallback"
    assert elems and elems[0].content == "fallback content"
