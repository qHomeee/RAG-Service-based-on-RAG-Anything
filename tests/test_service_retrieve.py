from app.repository import RetrievalRow
from app.service import RagService


class _FakeParser:
    pass


class _FakeRepository:
    def retrieve(self, query: str, top_k: int, min_score: float, collection: str, source_uris: list[str] | None):
        return [
            RetrievalRow(
                fragment_id="f1",
                source_uri="doc.md",
                title="doc.md",
                type="text",
                page=1,
                snippet="short preview",
                score=0.95,
                text="full fragment text that should be returned in API snippet",
            )
        ]


def test_retrieve_returns_full_fragment_in_snippet_field():
    service = RagService(parser=_FakeParser(), repository=_FakeRepository())
    hits = service.retrieve("query", 5, 0.2, "default", None, return_text=False)
    assert hits[0]["snippet"] == "full fragment text that should be returned in API snippet"
    assert hits[0]["text"] is None
