from pathlib import Path

import pytest

from app.document_intelligence import analyze_query, build_document_profile, is_toc_text
from app.config import settings
from app.repository import (
    RetrievalRow,
    MAX_DB_SNIPPET_CHARS,
    _bm25_scores,
    _normalize_rerank_scores,
    _query_terms_for_scoring,
    apply_adaptive_threshold,
    anchor_phrase_score,
    document_level_scores,
    lexical_overlap,
    mmr_select,
    normalize_query,
    normalize_context,
    phrase_match_score,
    rerank_by_keyword_relevance,
    retrieve_multi_query,
    score_retrieval_candidates,
    select_final_hits,
    topical_match_score,
    weighted_phrase_term_score,
)
from app.schemas import ParsedElement


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


def test_db_snippet_preview_fits_legacy_varchar_limit():
    snippet = normalize_context("слово " * 200, MAX_DB_SNIPPET_CHARS)

    assert len(snippet) <= 450
    assert snippet.endswith("слово")


def test_query_terms_keep_numbers_roman_numerals_and_short_terms():
    terms = _query_terms_for_scoring("Александр I, 1812 год, НН в причастиях")
    assert "александр" in terms
    assert "i" not in terms
    assert "1812" in terms
    assert "нн" in terms
    assert "н" not in terms
    assert "в" not in terms


@pytest.mark.parametrize(
    ("query", "subject"),
    [
        ("Александр I и его правление", "history"),
        ("правописание Н и НН в причастиях", "russian_language"),
        ("АСДНР при чрезвычайных ситуациях", "safety"),
        ("решить квадратное уравнение", "math"),
        ("фотосинтез", "biology"),
    ],
)
def test_query_understanding_detects_subjects(query, subject):
    analysis = analyze_query(query)

    assert analysis["primary_subject"] == subject
    assert analysis["subject_confidence"] >= 0.55
    assert analysis["key_terms"]
    assert all(item["subject"] for item in analysis["detected_subjects"])


def test_query_understanding_extracts_required_exact_entity_phrase():
    analysis = analyze_query("Александр I и его правление")

    assert "александр i" in analysis["exact_phrases"]


def test_document_profile_extracts_subject_grade_doc_type_sections_and_keywords():
    profile = build_document_profile(
        source_uri="biology/Biologiya_9klass_uchebnik.pdf",
        title="Biologiya_9klass_uchebnik.pdf",
        file_path=Path("Biologiya_9klass_uchebnik.pdf"),
        parsed_elements=[
            ParsedElement(
                element_index=0,
                type="text",
                content="# Фотосинтез\n\nБиология. Растения, хлорофилл, клетка.",
                page=1,
            )
        ],
        collection="default",
    )

    assert profile["subject"] == "biology"
    assert profile["grade"] == 9
    assert profile["doc_type"] == "textbook"
    assert profile["language"] == "ru"
    assert "Фотосинтез" in profile["section_titles"]
    assert "фотосинтез" in profile["keywords"]


def test_lexical_overlap_uses_light_russian_normalization():
    terms = _query_terms_for_scoring("реформы Сперанского")
    assert lexical_overlap(terms, "Проект реформ Сперанского") > 0.5


def test_lexical_overlap_respects_token_boundaries_and_weak_roman_tokens():
    terms = _query_terms_for_scoring("Александр I и его правление")

    assert lexical_overlap(terms, "I. Деепричастия обозначают добавочное действие.") == 0
    assert lexical_overlap(terms, "Направление мысли в тексте.") == 0
    assert lexical_overlap(terms, "Указать казакам на- правление, по которому нужно идти.") == 0
    assert lexical_overlap(terms, "Правление Александра I.") == 1.0


def test_phrase_match_boosts_name_plus_roman_number_only_as_phrase():
    assert phrase_match_score("Александр I и его правление", "Александр I начал правление") > 0
    assert phrase_match_score("Александр I и его правление", "I. Деепричастия и обстоятельства") == 0


def test_required_entity_modifier_penalizes_partial_and_wrong_entities():
    query = "Александр I и его правление"
    candidates = [
        _row("exact", 0.6, "Внутренняя политика Александра I и реформы Сперанского.", "history.pdf"),
        _row("partial", 0.8, "Александр укреплял государственную власть.", "history.pdf"),
        _row("wrong-ii", 0.85, "Правление Александра II и преобразования второй половины XIX века.", "history.pdf"),
        _row("wrong-iii", 0.82, "Александр III и его внутренняя политика.", "history.pdf"),
        _row("patronymic", 0.86, "Победоносцев, Александр Александрович и консервативная политика.", "history.pdf"),
    ]

    scored, rejected = score_retrieval_candidates(query, candidates, apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}

    assert not rejected
    assert by_id["exact"].phrase_score >= 0.75
    assert "александр i" in by_id["exact"].exact_phrases
    assert by_id["exact"].matched_phrases == ["александр i"]
    assert by_id["exact"].phrase_score == by_id["exact"].phrase_score_after_penalty
    assert by_id["partial"].phrase_score <= 0.2
    assert by_id["partial"].phrase_score == by_id["partial"].phrase_score_after_penalty
    assert by_id["wrong-ii"].phrase_score <= 0.2
    assert by_id["wrong-ii"].phrase_score == by_id["wrong-ii"].phrase_score_after_penalty
    assert by_id["wrong-iii"].phrase_score <= 0.2
    assert by_id["patronymic"].phrase_score <= 0.2
    assert by_id["wrong-ii"].wrong_entity_modifier is True
    assert by_id["wrong-iii"].wrong_entity_modifier is True
    assert by_id["patronymic"].wrong_entity_modifier is True
    assert by_id["wrong-ii"].score < by_id["partial"].score
    assert by_id["wrong-ii"].phrase_score_before_penalty > by_id["wrong-ii"].phrase_score_after_penalty


@pytest.mark.parametrize(
    ("query", "matching_text", "wrong_text"),
    [
        ("витамин B12", "Дефицит витамина B12 вызывает анемию.", "Витамин B6 относится к другой группе."),
        ("глава 5", "Глава 5 посвящена основным понятиям.", "Глава 6 посвящена повторению."),
        ("закон 44-ФЗ", "Федеральный закон 44-ФЗ регулирует закупки.", "Закон 223-ФЗ регулирует другую область."),
        ("правописание Н и НН в причастиях", "Правописание Н и НН в причастиях зависит от условий.", "Правописание одной Н в кратких формах."),
        ("АСДНР при чрезвычайных ситуациях", "АСДНР при чрезвычайных ситуациях включает спасательные работы.", "Меры защиты населения при чрезвычайных ситуациях."),
    ],
)
def test_required_exact_phrase_modifiers_are_universal(query, matching_text, wrong_text):
    scored, _ = score_retrieval_candidates(
        query,
        [_row("matching", 0.55, matching_text), _row("wrong", 0.8, wrong_text)],
        apply_noise_filter=False,
    )
    by_id = {hit.fragment_id: hit for hit in scored}

    assert by_id["matching"].phrase_score >= 0.75
    assert by_id["wrong"].phrase_score <= 0.2
    assert by_id["matching"].matched_phrases
    assert by_id["wrong"].missing_required_modifiers


def test_wrong_entity_modifier_is_rejected_from_alexander_i_top8():
    query = "Александр I и его правление"
    candidates = [
        _row("exact-1", 0.55, "Александр I и его правление: Негласный комитет и первые реформы.", "history.pdf"),
        _row("exact-2", 0.53, "Внутренняя политика Александра I и проекты Сперанского.", "history.pdf"),
        _row("exact-3", 0.52, "Александр I, Отечественная война 1812 года и внешняя политика.", "history.pdf"),
        _row("exact-4", 0.51, "Венский конгресс и дипломатия Александра I после войны.", "history.pdf"),
        _row("context-1", 0.5, "Реформы Сперанского, Негласный комитет и крепостное право.", "history.pdf"),
        _row("context-2", 0.49, "Внутренняя политика, внешняя политика и Венский конгресс.", "history.pdf"),
        _row("context-3", 0.48, "Отечественная война 1812 года и послевоенное устройство Европы.", "history.pdf"),
        _row("context-4", 0.47, "Крестьянский вопрос, военные поселения и государственные реформы.", "history.pdf"),
        _row(
            "pobedonostsev",
            0.9,
            "Победоносцев, Александр Александрович и Александр II в общественной мысли XIX века.",
            "history.pdf",
        ),
    ]

    scored, rejected = score_retrieval_candidates(query, candidates)
    selected = select_final_hits(scored, top_k=8, min_score=0.0)

    assert "pobedonostsev" not in [hit.fragment_id for hit in selected]
    assert any(item["fragment_id"] == "pobedonostsev" for item in rejected)
    assert all(
        hit.matched_phrases or hit.phrase_score_after_penalty > 0 or "context" in hit.fragment_id
        for hit in selected
    )


def test_phrase_term_weighting_prefers_reign_context_over_anchor_only():
    query = "Александр I и его правление"
    anchor_only = "Экспедиция к Антарктиде была снаряжена при поддержке императора Александра I."
    reign_context = "Александр I и его правление: реформы, внутренняя политика и крепостное право."
    thematic_context = "Реформы Сперанского, внешняя политика, Венский конгресс и крепостное право."

    assert anchor_phrase_score(query, anchor_only) == 1.0
    assert topical_match_score(query, anchor_only) == 0.0
    assert topical_match_score(query, reign_context) > 0.9
    assert topical_match_score(query, thematic_context) > 0.5
    assert weighted_phrase_term_score(query, reign_context) > weighted_phrase_term_score(query, anchor_only)
    assert weighted_phrase_term_score(query, thematic_context) > 0


def test_anchor_only_alexander_i_ranks_below_reign_and_thematic_hits():
    query = "Александр I и его правление"
    candidates = [
        _row(
            "antarctica",
            0.75,
            "Экспедиция Беллинсгаузена и Лазарева к Антарктиде была снаряжена по указу Александра I.",
            "history.pdf",
        ),
        _row(
            "reign",
            0.58,
            "Александр I и его правление: реформы Сперанского, внутренняя политика и крепостное право.",
            "history.pdf",
        ),
        _row(
            "policy",
            0.56,
            "Реформы Сперанского, внешняя политика, Венский конгресс и крепостное право.",
            "history.pdf",
        ),
    ]

    scored, rejected = score_retrieval_candidates(query, candidates)
    by_id = {hit.fragment_id: hit for hit in scored}

    assert not rejected
    assert scored[0].fragment_id == "reign"
    assert by_id["reign"].score > by_id["antarctica"].score
    assert by_id["policy"].score > by_id["antarctica"].score


def test_subject_mismatch_penalizes_other_subject_dense_hits():
    query = "фотосинтез"
    analysis = analyze_query(query)
    biology = _row("biology", 0.55, "Фотосинтез: хлорофилл, свет, углекислый газ и кислород.", "biology.pdf")
    biology.subject_score = 0.9
    history = _row("history", 0.95, "Отечественная война 1812 года и реформы государственного управления.", "history.pdf")
    history.subject_score = 0.12

    scored, rejected = score_retrieval_candidates(query, [history, biology], query_analysis=analysis)

    assert scored[0].fragment_id == "biology"
    assert any(item["fragment_id"] == "history" and item["rejection_reason"] == "subject_mismatch" for item in rejected)


def test_section_heading_boosts_relevant_chunk_text():
    query = "квадратное уравнение"
    analysis = analyze_query(query)
    candidate = _row("math", 0.45, "Формула и примеры решения.", "math.pdf")
    candidate.subject_score = 0.9
    candidate.meta = {"section_path": ["Алгебра", "Квадратное уравнение"], "subject": "math"}

    scored, rejected = score_retrieval_candidates(query, [candidate], query_analysis=analysis)

    assert not rejected
    assert scored[0].section_score > 0.5
    assert scored[0].phrase_score > 0.4


def test_table_of_contents_is_penalized_below_content_chunk():
    query = "Отечественная война 1812 года"
    analysis = analyze_query(query)
    toc = _row("toc", 0.8, "Оглавление\nОтечественная война 1812 года 34\nРеформы 40", "history.pdf")
    toc.meta = {"is_toc": True, "subject": "history"}
    toc.subject_score = 0.9
    content = _row("content", 0.55, "Отечественная война 1812 года: причины, ход войны и Бородинское сражение.", "history.pdf")
    content.subject_score = 0.9

    scored, _ = score_retrieval_candidates(query, [toc, content], query_analysis=analysis)
    by_id = {hit.fragment_id: hit for hit in scored}

    assert by_id["content"].score > by_id["toc"].score


def test_flat_table_of_contents_page_is_detected_and_excluded_from_final_hits():
    query = "Александр I и его правление"
    toc_text = (
        "Убийство Павла I 46 Ледовый переход Барклая де Толли 1809 г. "
        "Детство и юность М.М. Сперанского 8 Батарея генерала Раевского 10 "
        "Александр I на Венском конгрессе 12 Н.Н. Новосильцев 14 "
        "Русско-французский союз 70 В.О. Ключевский 72 "
        "Товарищество передвижных художественных выставок 74 Могучая кучка 76"
    )
    toc = _row("769f3ab9efdb18894d6360316bce4c14", 0.9, toc_text, "history.pdf")
    toc.page = 4
    content = _row(
        "content",
        0.55,
        "Александр I и его правление: Негласный комитет, реформы Сперанского, внешняя политика и Венский конгресс.",
        "history.pdf",
    )

    assert is_toc_text(toc_text, page=4)
    scored, _ = score_retrieval_candidates(query, [toc, content], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}
    selected = select_final_hits(scored, top_k=8, min_score=0.0)

    assert by_id["769f3ab9efdb18894d6360316bce4c14"].is_toc is True
    assert by_id["769f3ab9efdb18894d6360316bce4c14"].toc_penalty_applied is True
    assert "769f3ab9efdb18894d6360316bce4c14" not in [hit.fragment_id for hit in selected]
    assert selected[0].fragment_id == "content"


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
    assert rejected[0]["rejection_reason"] in {"low_lexical_overlap", "low_document_score"}


def test_dense_only_candidate_cannot_receive_high_final_score():
    candidate = _row("dense-only", 0.82, "Деепричастия несовершенного вида и пунктуация.", "Russkiy_yazyk_2019.pdf")
    scored, rejected = score_retrieval_candidates("Александр I и его правление", [candidate], apply_noise_filter=False)

    assert not rejected
    assert scored[0].lexical_overlap == 0
    assert scored[0].lexical_score == 0
    assert scored[0].score < 0.35


def test_bm25_for_russian_deeprichastie_text_is_near_zero_for_history_query():
    query_terms = _query_terms_for_scoring("Александр I и его правление")
    rows = [
        _row(
            "russian",
            0.82,
            "I. Деепричастия несовершенного вида. Указать казакам на- правление, по которому нужно идти.",
            "Russkiy_yazyk_2019.pdf",
        ),
        _row(
            "history",
            0.58,
            "Александр I и его правление. Реформы и внутренняя политика.",
            "Istoriya_Rossii.pdf",
        ),
    ]

    scores = _bm25_scores(query_terms, rows)

    assert scores["russian"] < 0.1
    assert scores["history"] > 0.8


def test_document_score_does_not_get_high_from_dense_only_document_match():
    scores = document_level_scores(["Russkiy_yazyk_2019.pdf"], [])

    assert scores["Russkiy_yazyk_2019.pdf"] < 0.2


def test_effective_document_score_is_capped_without_local_lexical_support():
    candidate = _row(
        "russian",
        0.82,
        "I. Деепричастия несовершенного вида. Указать казакам на- правление, по которому нужно идти.",
        "Russkiy_yazyk_2019.pdf",
    )
    candidate.document_score = 0.98

    scored, rejected = score_retrieval_candidates("Александр I и его правление", [candidate])

    assert not scored
    assert rejected[0]["document_score"] < 0.2
    assert rejected[0]["lexical_score"] < 0.1


def test_rerank_scores_are_relative_not_always_saturated():
    scores = _normalize_rerank_scores([10.0, 9.9, 1.0], 3)

    assert scores == [1.0, pytest.approx(0.9889, rel=1e-4), 0.0]


def test_min_score_is_applied_to_final_score_on_zero_to_one_scale():
    candidates = [
        _row("good", 0.6, "Правописание Н и НН в причастиях и отглагольных прилагательных."),
        _row("weak", 0.2, "Техника эвакуации при чрезвычайных ситуациях."),
    ]
    scored, _ = score_retrieval_candidates("правописание Н и НН в причастиях", candidates)
    selected = select_final_hits(scored, top_k=15, min_score=0.35)

    assert selected
    assert all(0.0 <= hit.score <= 1.0 for hit in selected)
    assert all(hit.score >= 0.35 for hit in selected)
    assert all(hit.fragment_id != "weak" for hit in selected)


def test_single_letter_n_does_not_drive_russian_language_query_match():
    query = "правописание Н и НН в причастиях"
    candidates = [
        _row("bad-letter", 0.8, "Н. Н. Иванов написал текст об истории государства.", "history.pdf"),
        _row("good", 0.55, "Правописание Н и НН в причастиях и отглагольных прилагательных.", "russian.pdf"),
    ]

    scored, rejected = score_retrieval_candidates(query, candidates)
    selected = select_final_hits(scored, top_k=5, min_score=0.35)

    assert selected[0].fragment_id == "good"
    assert all(hit.fragment_id != "bad-letter" for hit in selected)
    assert any(item["fragment_id"] == "bad-letter" for item in rejected)


def test_asdnr_query_does_not_match_history_on_common_words():
    query = "АСДНР при чрезвычайных ситуациях"
    candidates = [
        _row("history", 0.7, "История России при Александре I: политика в сложных ситуациях.", "history.pdf"),
        _row("civil", 0.55, "АСДНР при чрезвычайных ситуациях: спасательные работы и защита населения.", "civil-defense.pdf"),
    ]

    scored, rejected = score_retrieval_candidates(query, candidates)
    selected = select_final_hits(scored, top_k=5, min_score=0.35)

    assert selected[0].fragment_id == "civil"
    assert all(hit.fragment_id != "history" for hit in selected)
    assert any(item["fragment_id"] == "history" for item in rejected)


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


def test_alexander_i_debug_smoke_excludes_russian_and_generic_pdfs_from_top5():
    candidates = [
        _row(
            "history-1",
            0.58,
            "Александр I и его правление. Реформы Сперанского, Негласный комитет, Отечественная война 1812 года.",
            "Istoriya_Rossii_9klass.pdf",
        ),
        _row(
            "history-2",
            0.55,
            "Внутренняя политика Александра I, крепостное право, Венский конгресс и внешняя политика.",
            "Istoriya_Rossii_9klass.pdf",
        ),
        _row(
            "russian-noise",
            0.82,
            "Деепричастия несовершенного вида, обособление обстоятельств и синтаксический разбор.",
            "Russkiy_yazyk_2019.pdf",
        ),
        _row(
            "generic-noise",
            0.79,
            "Техника безопасности, средства индивидуальной защиты и порядок эвакуации.",
            "pub_1167883.pdf",
        ),
    ]

    scored, rejected = score_retrieval_candidates("Александр I и его правление", candidates)
    selected = select_final_hits(scored, top_k=15, min_score=0.35)

    top5_sources = [hit.source_uri for hit in selected[:5]]
    assert selected[0].source_uri == "Istoriya_Rossii_9klass.pdf"
    assert "Russkiy_yazyk_2019.pdf" not in top5_sources
    assert "pub_1167883.pdf" not in top5_sources
    assert all(0.0 <= hit.score <= 1.0 for hit in selected)
    assert all(hit.dense_score is not None for hit in selected)
    assert all(hit.lexical_score is not None for hit in selected)
    assert all((hit.final_score or hit.score) is not None for hit in selected)
    rejected_sources = {item["source_uri"] for item in rejected}
    assert {"Russkiy_yazyk_2019.pdf", "pub_1167883.pdf"} <= rejected_sources
    russian_rejection = next(item for item in rejected if item["source_uri"] == "Russkiy_yazyk_2019.pdf")
    assert russian_rejection["lexical_score"] < 0.1
    assert russian_rejection["document_score"] < 0.2
