from pathlib import Path

from app.parser import RAGAnythingParser
from app.schemas import ParsedElement
from app.service import RagService


class _DummyDB:
    def commit(self):
        return None


class _DummyDoc:
    def __init__(self, doc_id: str = "doc-1"):
        self.doc_id = doc_id


class _FakeRepo:
    def __init__(self):
        self.db = _DummyDB()
        self.docs = []
        self.fragments = []

    def upsert_document(self, source_uri, title, collection, meta, reindex):
        self.docs.append((source_uri, title, collection, meta, reindex))
        return _DummyDoc()

    def insert_fragment_with_embeddings(self, doc, fragment):
        self.fragments.append(fragment)
        return 1


def test_ingest_pipeline_uses_mineru_subprocess_output(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    sample = raw / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4")

    parser = RAGAnythingParser()

    def fake_run_mineru(path: Path, *, text_only: bool, reindex: bool = False):
        assert text_only is False
        return {"elements": [{"type": "text", "text": "Заголовок\n\nТекст документа", "page": 1}]}

    monkeypatch.setattr("app.parser.check_mineru_ready", lambda _py: (True, "ok"))
    monkeypatch.setattr("app.parser.resolve_mineru_python", lambda: Path("python"))
    monkeypatch.setattr(parser, "_run_mineru_subprocess", fake_run_mineru)

    repo = _FakeRepo()
    service = RagService(parser=parser, repository=repo)

    stats = service.ingest(str(raw), collection="default", reindex=False)

    assert stats["indexed_docs"] == 1
    assert stats["indexed_fragments"] >= 1
    assert stats["indexed_vectors"] >= 1
    assert repo.docs[0][3]["parse_mode"] == "rag_anything"
