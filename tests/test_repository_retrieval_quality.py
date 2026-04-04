import pytest

from app.config import settings
from app.repository import RetrievalRow, expand_query, normalize_query, rerank_by_keyword_relevance, retrieve_multi_query


def _row(fragment_id: str, score: float, text: str) -> RetrievalRow:
    return RetrievalRow(
        fragment_id=fragment_id,
        source_uri="s",
        title="t",
        type="text",
        page=None,
        snippet=text,
        score=score,
        text=text,
    )


def test_normalize_query_removes_topic_prefix():
    assert normalize_query("тема урока: османская империя") == "османская империя"
    assert normalize_query("урок:  Первая мировая  ") == "первая мировая"


def test_expand_query_builds_thematic_variants_for_short_query():
    original = settings.query_expansion_enabled
    settings.query_expansion_enabled = True
    try:
        variants = expand_query("османская империя")
        assert variants[0] == "османская империя"
        assert len(variants) >= 2
        assert any("султан" in variant for variant in variants)
    finally:
        settings.query_expansion_enabled = original


def test_multi_query_fusion_uses_weighted_max_score():
    calls = {
        "q1": [_row("a", 1.0, "a text"), _row("b", 0.7, "b text")],
        "q2": [_row("a", 0.9, "a text"), _row("c", 0.8, "c text")],
    }
    merged = retrieve_multi_query(retrieve_fn=lambda q: calls[q], queries=["q1", "q2"])

    assert set(merged) == {"a", "b", "c"}
    assert merged["a"][0] == 1.0
    assert merged["c"][0] == pytest.approx(0.72)


def test_keyword_rerank_penalizes_irrelevant_hits():
    hits = [
        _row("good", 1.0, "Османская империя, султан и Стамбул"),
        _row("bad", 1.0, "Британская армия и реформы промышленности"),
    ]

    reranked = rerank_by_keyword_relevance("османская империя", hits)

    assert reranked[0].fragment_id == "good"
    assert reranked[0].score > reranked[1].score
