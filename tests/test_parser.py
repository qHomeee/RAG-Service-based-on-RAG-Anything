from pathlib import Path

from app.parser import RAGAnythingParser
from app.schemas import ParsedElement


def test_logs_fallback_reason_when_rag_anything_empty(tmp_path, monkeypatch, caplog):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    def fake_parse_with_rag(_: Path):
        return None, "empty_elements"

    monkeypatch.setattr(parser, "_parse_with_rag_anything", fake_parse_with_rag)

    with caplog.at_level("INFO", logger="rag_service"):
        elements, mode = parser.parse_file_with_mode("sample.txt", sample)

    assert mode == "fallback"
    assert elements == [ParsedElement(element_index=0, type="text", content="hello", meta={"fallback": True})]
    assert any("parser_fallback_used" in rec.message and "empty_elements" in str(rec.__dict__) for rec in caplog.records)


def test_logs_warning_when_rag_anything_throws(tmp_path, caplog):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    with caplog.at_level("WARNING", logger="rag_service"):
        elements, reason = parser._parse_with_rag_anything(sample)

    # Runtime without rag_anything should emit warning and return explicit reason.
    assert elements is None
    assert reason.startswith("exception:")
    assert any("rag_anything_parse_failed" in rec.message for rec in caplog.records)


def test_load_pipeline_module_supports_raganything_alias(monkeypatch):
    parser = RAGAnythingParser()

    calls = []

    class DummyModule:
        class ParsingPipeline:  # pragma: no cover - shape only
            pass

    def fake_import(name: str):
        calls.append(name)
        if name == "rag_anything.pipeline":
            raise ModuleNotFoundError("legacy name unavailable")
        if name == "raganything.pipeline":
            return DummyModule
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("app.parser.importlib.import_module", fake_import)

    module = parser._load_rag_pipeline_module()
    assert module is DummyModule
    assert calls == ["rag_anything.pipeline", "raganything.pipeline"]
