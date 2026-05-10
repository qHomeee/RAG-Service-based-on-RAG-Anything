from app.repository import RetrievalRow
from app.service import RagService


class _FakeParser:
    pass


class _FakeRepository:
    def retrieve(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
        include_toc: bool = False,
    ):
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
                dense_score=0.8,
                lexical_score=0.7,
                phrase_score=0.65,
                subject_score=0.9,
                section_score=0.4,
                rerank_score=0.6,
                final_score=0.75,
                lexical_overlap=0.5,
                document_score=0.9,
                rrf_score=0.4,
                exact_phrases=["query 1"],
                matched_phrases=["query 1"],
                missing_required_modifiers=[],
                wrong_entity_modifier=False,
                phrase_score_before_penalty=0.8,
                phrase_score_after_penalty=0.65,
                is_toc=False,
                toc_filtered=False,
                toc_penalty_applied=False,
            )
        ]


def test_retrieve_returns_full_fragment_in_snippet_field():
    service = RagService(parser=_FakeParser(), repository=_FakeRepository())
    hits = service.retrieve("query", 5, 0.2, "default", None, return_text=False)
    assert hits[0]["snippet"] == "full fragment text that should be returned in API snippet"
    assert hits[0]["text"] is None
    assert hits[0]["score"] == 0.75
    assert hits[0]["dense_score"] == 0.8
    assert hits[0]["lexical_score"] == 0.7
    assert hits[0]["phrase_score"] == 0.65
    assert hits[0]["subject_score"] == 0.9
    assert hits[0]["section_score"] == 0.4
    assert hits[0]["rerank_score"] == 0.6
    assert hits[0]["final_score"] == 0.75
    assert hits[0]["lexical_overlap"] == 0.5
    assert hits[0]["document_score"] == 0.9
    assert hits[0]["rrf_score"] == 0.4
    assert hits[0]["exact_phrases"] == ["query 1"]
    assert hits[0]["matched_phrases"] == ["query 1"]
    assert hits[0]["missing_required_modifiers"] == []
    assert hits[0]["wrong_entity_modifier"] is False
    assert hits[0]["phrase_score_before_penalty"] == 0.8
    assert hits[0]["phrase_score"] == hits[0]["phrase_score_after_penalty"]
    assert hits[0]["is_toc"] is False
    assert hits[0]["toc_filtered"] is False
    assert hits[0]["toc_penalty_applied"] is False
