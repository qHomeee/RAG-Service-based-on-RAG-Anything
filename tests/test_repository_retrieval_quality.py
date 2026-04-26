import pytest

from app.config import settings
from app.repository import (
    RetrievalRow,
    _query_terms_for_scoring,
    apply_adaptive_threshold,
    lexical_overlap,
    mmr_select,
    normalize_query,
    rerank_by_keyword_relevance,
    retrieve_multi_query,
    score_retrieval_candidates,
    select_final_hits,
)


def _row(fragment_id: str, dense_score: float, text: str, source_uri: str = "s") -> RetrievalRow:
    return RetrievalRow(
        fragment_id=fragment_id,
        source_uri=source_uri,
        title=source_uri,
        type="text",
        page=None,
        snippet=text,
        score=dense_score,
        text=text,
        dense_score=dense_score,
    )


def test_normalize_query_removes_topic_prefix():
    assert normalize_query("тема урока: османская империя") == "османская империя"
    assert normalize_query("урок:  Первая мировая  ") == "первая мировая"


def test_query_terms_keep_numbers_roman_numerals_and_short_terms():
    terms = _query_terms_for_scoring("Александр I, 1812 год, НН в причастиях")
    assert "александр" in terms
    assert "i" in terms
    assert "1812" in terms
    assert "нн" in terms
    assert "в" not in terms


def test_lexical_overlap_uses_light_russian_normalization():
    terms = _query_terms_for_scoring("реформы Сперанского")
    assert lexical_overlap(terms, "Проект реформ Сперанского") > 0.5


def test_multi_query_fusion_uses_weighted_max_score():
    calls = {
        "q1": [_row("a", 1.0, "a text"), _row("b", 0.7, "b text")],
        "q2": [_row("a", 0.9, "a text"), _row("c", 0.8, "c text")],
    }
    merged = retrieve_multi_query(retrieve_fn=lambda q: calls[q], queries=["q1", "q2"])

    assert set(merged) == {"a", "b", "c"}
    assert merged["a"][0] == 1.0
    assert merged["c"][0] == pytest.approx(0.72)


def test_keyword_rerank_penalizes_irrelevant_hits_without_query_hardcode():
    hits = [
        _row("good", 0.7, "Османская империя, султан и Стамбул"),
        _row("bad", 0.7, "Британская армия и реформы промышленности"),
    ]

    reranked = rerank_by_keyword_relevance("османская империя", hits)

    assert reranked[0].fragment_id == "good"
    assert reranked[0].score > reranked[1].score
    assert all(0.0 <= hit.score <= 1.0 for hit in reranked)


def test_anti_noise_filter_drops_zero_overlap_low_dense_candidates():
    candidates = [
        _row("history", 0.45, "Александр I. Реформы Сперанского и Негласный комитет.", "history.pdf"),
        _row("noise", 0.5, "Одуванчики, техника безопасности и бытовые инструкции.", "biology.pdf"),
    ]

    scored, rejected = score_retrieval_candidates("Александр I и его правление", candidates)

    assert [row.fragment_id for row in scored] == ["history"]
    assert rejected[0]["fragment_id"] == "noise"
    assert rejected[0]["rejection_reason"] in {"low_dense_no_lexical_overlap", "no_query_terms_in_chunk"}


def test_min_score_is_applied_to_final_score_on_zero_to_one_scale():
    candidates = [
        _row("good", 0.6, "Правописание НН в причастиях и отглагольных прилагательных."),
        _row("weak", 0.2, "Техника эвакуации при чрезвычайных ситуациях."),
    ]
    scored, _ = score_retrieval_candidates("правописание Н и НН в причастиях", candidates)
    selected = select_final_hits(scored, top_k=15, min_score=0.35)

    assert selected
    assert all(0.0 <= hit.score <= 1.0 for hit in selected)
    assert all(hit.score >= 0.35 for hit in selected)
    assert all(hit.fragment_id != "weak" for hit in selected)


def test_adaptive_threshold_cuts_score_tail():
    hits = [
        _row("a", 0.9, "a"),
        _row("b", 0.82, "b"),
        _row("tail", 0.35, "tail"),
    ]
    hits = [hit if hit.final_score else hit.__class__(**{**hit.__dict__, "final_score": hit.score}) for hit in hits]

    selected = apply_adaptive_threshold(hits)

    assert [hit.fragment_id for hit in selected] == ["a", "b"]


def test_mmr_keeps_relevant_results_but_reduces_duplicates():
    hits = [
        _row("a", 0.9, "реформы сперанского государственный совет министерства"),
        _row("dup", 0.88, "реформы сперанского государственный совет министерства"),
        _row("b", 0.84, "отечественная война 1812 года внешняя политика"),
    ]
    hits = [hit if hit.final_score else hit.__class__(**{**hit.__dict__, "final_score": hit.score}) for hit in hits]

    selected = mmr_select(hits, top_k=2, query_terms=_query_terms_for_scoring("реформы Сперанского 1812"))

    assert selected[0].fragment_id == "a"
    assert selected[1].fragment_id == "b"


@pytest.mark.parametrize(
    ("query", "wanted_source", "blocked_sources"),
    [
        ("Александр I и его правление", "history.pdf", {"russian.pdf", "civil-defense.pdf"}),
        ("Отечественная война 1812 года", "history.pdf", {"russian.pdf", "civil-defense.pdf"}),
        ("реформы Сперанского", "history.pdf", {"russian.pdf", "civil-defense.pdf"}),
        ("правописание Н и НН в причастиях", "russian.pdf", {"history.pdf", "civil-defense.pdf"}),
        ("виды придаточных предложений", "russian.pdf", {"history.pdf", "civil-defense.pdf"}),
        ("АСДНР при чрезвычайных ситуациях", "civil-defense.pdf", {"history.pdf", "russian.pdf"}),
        ("средства гражданской обороны", "civil-defense.pdf", {"history.pdf", "russian.pdf"}),
    ],
)
def test_mixed_textbook_smoke_queries_do_not_fill_top_k_with_other_subjects(query, wanted_source, blocked_sources):
    candidates = [
        _row(
            "history-1",
            0.5,
            "Александр I, реформы Сперанского, Негласный комитет, Отечественная война 1812 года.",
            "history.pdf",
        ),
        _row(
            "history-2",
            0.48,
            "Внутренняя и внешняя политика России, Венский конгресс, крепостное право.",
            "history.pdf",
        ),
        _row(
            "russian-1",
            0.5,
            "Правописание Н и НН в причастиях. Виды придаточных предложений в русском языке.",
            "russian.pdf",
        ),
        _row(
            "civil-1",
            0.5,
            "АСДНР при чрезвычайных ситуациях. Средства гражданской обороны и защита населения.",
            "civil-defense.pdf",
        ),
        _row("noise", 0.35, "Одуванчики, садовая техника и бытовые советы.", "other.pdf"),
    ]
    scored, _ = score_retrieval_candidates(query, candidates)
    selected = select_final_hits(scored, top_k=15, min_score=0.35)

    assert selected
    assert len(selected) < 15
    assert selected[0].source_uri == wanted_source
    assert all(0.0 <= hit.score <= 1.0 for hit in selected)
    assert all(hit.score >= 0.35 for hit in selected)
    assert not ({hit.source_uri for hit in selected} & blocked_sources)
