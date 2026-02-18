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


def test_logs_warning_when_rag_anything_throws(tmp_path, monkeypatch, caplog):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    def fake_loader():
        def _raise(_: str):
            raise RuntimeError("boom")

        return _raise

    monkeypatch.setattr(parser, "_load_rag_parse_callable", fake_loader)

    with caplog.at_level("WARNING", logger="rag_service"):
        elements, reason = parser._parse_with_rag_anything(sample)

    assert elements is None
    assert reason.startswith("exception:")
    assert any("rag_anything_parse_failed" in rec.message for rec in caplog.records)


def test_load_parse_callable_prefers_raganything_root(monkeypatch):
    parser = RAGAnythingParser()
    calls = []

    class DummyModule:
        class RAGAnything:
            def parse_document(self, _: str):  # pragma: no cover - shape only
                return {"elements": [{"type": "text", "text": "ok"}]}

    def fake_import(name: str):
        calls.append(name)
        if name == "raganything":
            return DummyModule
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("app.parser.importlib.import_module", fake_import)

    parse_callable = parser._load_rag_parse_callable()
    assert callable(parse_callable)
    assert calls == ["raganything"]


def test_load_parse_callable_supports_parser_module_class(monkeypatch):
    parser = RAGAnythingParser()

    class DummyParserModule:
        class DocumentParser:
            def parse_file(self, _: str):  # pragma: no cover - shape only
                return {"elements": [{"type": "text", "text": "ok"}]}

    def fake_import(name: str):
        if name == "raganything":
            raise ModuleNotFoundError(name)
        if name == "raganything.parser":
            return DummyParserModule
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("app.parser.importlib.import_module", fake_import)

    parse_callable = parser._load_rag_parse_callable()
    assert callable(parse_callable)


def test_parse_with_rag_anything_accepts_list_payload(tmp_path, monkeypatch):
    parser = RAGAnythingParser()
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    def fake_loader():
        return lambda _: [{"type": "text", "text": "a"}]

    monkeypatch.setattr(parser, "_load_rag_parse_callable", fake_loader)
    elements, reason = parser._parse_with_rag_anything(sample)

    assert reason == "ok"
    assert elements == [{"type": "text", "text": "a"}]
