from pathlib import Path

import pytest

from app.document_intelligence import (
    analyze_query,
    build_document_profile,
    detect_chunk_type,
    detect_chunk_type_details,
    infer_section_title,
    is_toc_text,
    text_quality_flags,
)
from app.config import settings
from app.repository import (
    RagRepository,
    RetrievalRow,
    MAX_DB_SNIPPET_CHARS,
    _bm25_scores,
    _balanced_prerank,
    _candidate_quality,
    _normalize_rerank_scores,
    _metadata_with_inferred_section,
    _apply_explicit_source_scope,
    _query_terms_for_scoring,
    answer_focus_alignment_score,
    apply_adaptive_threshold,
    anchor_phrase_score,
    document_level_scores,
    lexical_overlap,
    mmr_select,
    normalize_query,
    normalize_context,
    normalize_for_required_term,
    phrase_match_score,
    required_term_match_score,
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
        ("Морфологический разбор", "russian_language"),
        ("синтаксический разбор предложения", "russian_language"),
        ("АСДНР при чрезвычайных ситуациях", "safety"),
        ("средства гражданской обороны", "safety"),
        ("решить квадратное уравнение", "math"),
        ("производная функции", "math"),
        ("фотосинтез", "biology"),
        ("митоз клетки", "biology"),
        ("условия протекания реакций ионного обмена", "chemistry"),
        ("как устранить жёсткость воды", "chemistry"),
        ("Почему раствор хлороводорода проводит электрический ток?", "chemistry"),
        ("специализация экономического района", "geography"),
        ("топливно-энергетический комплекс России", "geography"),
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


@pytest.mark.parametrize(
    ("query", "required_term"),
    [
        ("морфологический разбор", "морфологический разбор"),
        ("синтаксический разбор", "синтаксический разбор"),
        ("квадратное уравнение", "квадратное уравнение"),
        ("гражданская оборона", "гражданская оборона"),
        ("отечественная война", "отечественная война"),
        ("клеточное дыхание", "клеточное дыхание"),
    ],
)
def test_query_understanding_extracts_multiword_required_terms_for_concepts(query, required_term):
    analysis = analyze_query(query)

    assert analysis["query_type"] == "concept_lookup"
    assert analysis["required_terms"] == [required_term]


def test_required_term_score_requires_full_multiword_term():
    required_terms = ["морфологический разбор"]

    assert "морфологический разбор" in normalize_for_required_term("Морфологический разбор имени числительного I. Часть речи")

    exact = required_term_match_score(required_terms, "Морфологический разбор причастия")
    exact_numeral = required_term_match_score(required_terms, "Морфологический разбор имени числительного I. Часть речи")
    exact_with_steps = required_term_match_score(required_terms, "Морфологический разбор междометия I. Часть речи. II. Морфологические признаки.")
    exact_hyphenated = required_term_match_score(required_terms, "Морфологи-\nческий разбор глагола")
    lemma = required_term_match_score(required_terms, "План морфологического разбора глагола")
    partial = required_term_match_score(required_terms, "Рецензия — отзыв, письменный разбор произведения")
    syntax_partial = required_term_match_score(required_terms, "См. разбор сложносочинённого предложения")
    other_partial = required_term_match_score(required_terms, "Морфологические признаки причастия")

    assert exact.score == 1.0
    assert exact.match_type == "full_phrase_prefix"
    assert exact_numeral.score == 1.0
    assert exact_numeral.match_type == "full_phrase_prefix"
    assert exact_with_steps.score == 1.0
    assert exact_with_steps.match_type == "full_phrase_prefix"
    assert exact_hyphenated.score == 1.0
    assert exact_hyphenated.match_type == "full_phrase_prefix"
    assert lemma.score == 1.0
    assert lemma.match_type == "full_phrase"
    assert partial.score <= 0.25
    assert partial.match_type == "partial"
    assert syntax_partial.score <= 0.25
    assert syntax_partial.match_type == "partial"
    assert other_partial.score <= 0.25


@pytest.mark.parametrize(
    ("query", "query_type"),
    [
        ("Морфологический разбор", "concept_lookup"),
        ("квадратное уравнение", "concept_lookup"),
        ("фотосинтез", "concept_lookup"),
        ("гражданская оборона", "concept_lookup"),
        ("правописание Н и НН в причастиях", "rule_lookup"),
        ("упражнения на морфологический разбор", "exercise_lookup"),
        ("тренировочные задания по причастиям", "exercise_lookup"),
        ("решить квадратное уравнение", "problem_solving"),
    ],
)
def test_query_understanding_detects_retrieval_intent(query, query_type):
    assert analyze_query(query)["query_type"] == query_type


def test_query_understanding_detects_answer_focus_and_required_entities():
    analysis = analyze_query("Какие цели преследовали реформы Танзимат в Османской империи?")

    assert analysis["query_type"] == "explanation"
    assert analysis["answer_focus"] == "goal"
    assert analysis["required_entities"] == ["Танзимат", "Османской"]


def test_query_understanding_does_not_treat_kakovy_as_an_entity():
    analysis = analyze_query("Каковы причины Младотурецкой революции 1908 года?")

    assert analysis["named_entities"] == ["Младотурецкой"]
    assert analysis["required_entities"] == ["Младотурецкой"]


@pytest.mark.parametrize(
    "query",
    [
        "Какими средствами выражается сравнение?",
        "От чего зависит выбор знака препинания в бессоюзном предложении?",
    ],
)
def test_query_understanding_does_not_treat_question_openers_as_entities(query):
    analysis = analyze_query(query)

    assert analysis["required_entities"] == []


def test_query_understanding_recognizes_implicit_goal_wording():
    analysis = analyze_query("Кто такие младотурки и чего они добивались?")

    assert analysis["query_type"] == "explanation"
    assert analysis["answer_focus"] == "goal"


def test_location_question_is_explanation_not_required_concept_phrase():
    analysis = analyze_query(
        "Где сосредоточены металлургические предприятия Поволжья?"
    )

    assert analysis["query_type"] == "explanation"
    assert analysis["required_terms"] == []
    assert "Где" not in analysis["required_entities"]


def test_cause_alignment_does_not_reward_reversed_consequence():
    query = "Каковы причины революции 1908 года?"

    cause = answer_focus_alignment_score(
        query,
        "Глубокий кризис и недовольство стали предпосылками революции 1908 года.",
    )
    consequence = answer_focus_alignment_score(
        query,
        "Революция 1908 года вызвала международный кризис.",
    )

    assert cause > 0.5
    assert consequence == 0.0


def test_cause_alignment_recognizes_explanatory_historical_prose():
    query = "Почему Османская империя ослабла к середине XIX века?"

    direct = answer_focus_alignment_score(
        query,
        (
            "Положение осложняли поражения Османской империи и нежелание элиты "
            "модернизировать государство к середине XIX века."
        ),
    )
    partial = answer_focus_alignment_score(
        query,
        "К середине XIX века движение на Балканах всё больше ослабляло Турцию.",
    )

    assert direct > partial


def test_cause_alignment_recognizes_scientific_explanation():
    query = "Почему раствор хлороводорода проводит электрический ток?"
    evidence = (
        "Под действием молекул воды молекулы хлороводорода распадаются "
        "на катионы водорода и хлорид-анионы. Благодаря появлению в растворе "
        "заряженных частиц он проводит электрический ток."
    )

    assert answer_focus_alignment_score(query, evidence) > 0.5


def test_final_scoring_does_not_treat_neighbor_context_as_target_evidence():
    query = "Какие цели преследовали реформы Танзимат?"
    leaked = _row(
        "neighbor-leak",
        0.9,
        "Соседний фрагмент: целью реформ Танзимата было укрепление власти.",
    )
    leaked.fragment_text = "Справочная хронология событий."
    leaked.expanded_from_neighbors = True
    leaked.meta = {
        "expanded_text": leaked.text,
        "expanded_from_neighbors": True,
    }
    exact = _row(
        "exact",
        0.7,
        "Целью реформ Танзимата было укрепление центральной власти.",
    )

    scored, _ = score_retrieval_candidates(
        query,
        [leaked, exact],
        rerank_raw_scores=[0.5, 0.5],
        apply_noise_filter=False,
    )
    by_id = {hit.fragment_id: hit for hit in scored}

    assert by_id["neighbor-leak"].required_entity_score == 0.0
    assert by_id["neighbor-leak"].answer_alignment_score == 0.0
    assert scored[0].fragment_id == "exact"


def test_query_understanding_detects_out_of_domain_school_subjects():
    assert analyze_query("Сформулируй закон Ома для электрической цепи")["primary_subject"] == "physics"
    assert analyze_query("Площадь равнобедренного треугольника")["primary_subject"] == "math"


def test_retrieve_does_not_remove_filter_when_document_router_rejects_everything():
    repository = object.__new__(RagRepository)
    repository.validate_embedding_compatibility = lambda collection: None
    repository._document_prefilter = lambda *args, **kwargs: (
        [],
        {},
        {},
        [],
        [{"reason": "subject_mismatch"}],
    )

    def unexpected_recall(*args, **kwargs):
        raise AssertionError("fragment recall must not run without an eligible document")

    repository._dense_recall = unexpected_recall
    repository._keyword_recall = unexpected_recall

    assert repository.retrieve(
        query="закон Ома",
        top_k=3,
        min_score=0.2,
        collection="default",
        source_uris=None,
    ) == []


@pytest.mark.parametrize(
    ("text", "chunk_type"),
    [
        ("408. Выполните морфологический разбор причастий. Спишите предложения.", "exercise"),
        ("А4. Часть речи неправильно определена. А5. Верно определены морфологические признаки у слова.", "test_question"),
        ("А1. Укажите верный ответ. 1) существительное 2) глагол 3) союз 4) частица", "test_question"),
        ("Приложение Планы разборов 238 Морфологический разбор имени существительного 239", "navigation_index"),
        (
            "Фонетика. Графика. Орфография 167 Лексикология. Фразеология. Орфография 172 "
            "Морфемика. Словообразование. Орфография 181 Морфология. Орфография 186 "
            "Синтаксис. Пунктуация 204 Употребление знаков препинания 222",
            "navigation_index",
        ),
        (
            "§ 15. Запятая и точка с запятой в бессоюзном сложном предложении 123 "
            "§ 16. Двоеточие в бессоюзном сложном предложении 125 "
            "§ 17. Тире в бессоюзном сложном предложении 130 Реферат 136",
            "navigation_index",
        ),
        (
            "§ 20. Роль языка в жизни общества. Язык как исторически развивающееся явление 150 "
            "§ 21. Русский литературный язык и его стили 159",
            "navigation_index",
        ),
        (
            "Османская империя и Иран. В чём причина революций начала XX в.? "
            "• Танзимат • садразам • бабизм • Большая игра "
            "1839 г. — начало Танзимата 1908 г. — Младотурецкая революция "
            "1905—1911 гг. — Конституционная революция в Иране",
            "navigation_index",
        ),
        ("Порядок морфологического разбора имени существительного. I. Часть речи. II. Морфологические признаки. III. Синтаксическая роль.", "schema_or_plan"),
        ("Квадратное уравнение - это уравнение вида ax2 + bx + c = 0.", "definition"),
        ("Правило: в полных причастиях пишется НН при наличии приставки.", "rule"),
        ("Например, фотосинтез происходит на свету.", "example"),
    ],
)
def test_chunk_type_detection_is_intent_aware(text, chunk_type):
    assert detect_chunk_type(text) == chunk_type


def test_runtime_navigation_detection_overrides_stale_index_metadata():
    row = _row(
        "chapter-opener",
        0.9,
        (
            "Османская империя и Иран. В чём причина революций начала XX в.? "
            "• Танзимат • садразам • бабизм • Большая игра "
            "1839 г. — начало Танзимата 1908 г. — Младотурецкая революция "
            "1905—1911 гг. — Конституционная революция в Иране"
        ),
    )
    row.meta = {"chunk_type": "rule", "chunk_type_reason": "stale_index_value"}

    scored, _ = score_retrieval_candidates(
        "Почему Османская империя ослабла?",
        [row],
        apply_noise_filter=False,
    )

    assert scored[0].chunk_type == "navigation_index"
    assert scored[0].is_navigation_index is True


def test_numbered_definition_examples_are_not_misclassified_as_exercise():
    text = (
        "Придаточные изъяснительные отвечают на падежные вопросы. "
        "Они относятся к словам со значением речи, мысли или чувства. "
        "Это чаще всего глаголы речи и мысли. "
        "Например: 1) Я сказал, что приду. 2) Он подумал, что успеет. "
        "3) Я рад, что вы пришли. 4) Говорили, будто его видели. "
        "5) Известно, что поезд прибыл. 6) Она спросила, придём ли мы."
    )
    assert detect_chunk_type(text) == "definition"


def test_expanded_context_preserves_curated_section_title():
    row = _row(
        "heading",
        0.8,
        "305. Прочитайте текст. Выделите авторские знаки и объясните их постановку.",
    )
    row.meta = {"section_title": "§ 19. Авторские знаки препинания"}

    meta = _metadata_with_inferred_section(row)

    assert meta["section_title"] == "§ 19. Авторские знаки препинания"
    assert meta["section_title_reason"] == "meta:section_title"


def test_section_title_extraction_rejects_plan_steps_test_labels_and_numeric_noise():
    assert infer_section_title("I. Часть речи. Общее грамматическое значение") is None
    assert infer_section_title("II. Морфологические признаки") is None
    assert infer_section_title("III. Синтаксическая роль") is None
    assert infer_section_title("А5. Верно определены морфологические признаки") is None
    assert infer_section_title("1,") is None
    assert infer_section_title("1, 2.") is None
    assert infer_section_title("1, 2. См. разбор сложносочинённого предложения") is None
    assert infer_section_title("Рецензия (от нем") is None
    assert infer_section_title("Морфологический разбор имени существительного") == "Морфологический разбор имени существительного"
    assert detect_chunk_type_details("1, 2. См. разбор сложносочинённого предложения")["chunk_type"] == "navigation_index"


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


def test_subject_score_uses_chunk_metadata_when_document_prefilter_is_absent():
    analysis = analyze_query("Морфологический разбор")
    russian = _row("ru", 0.55, "Выполните морфологический разбор причастий.", "russian.pdf")
    russian.meta = {"subject": "russian_language"}
    history = _row("history", 0.9, "История России и реформы государственного управления.", "history.pdf")
    history.meta = {"subject": "history"}

    scored, rejected = score_retrieval_candidates("Морфологический разбор", [history, russian], query_analysis=analysis)
    by_id = {hit.fragment_id: hit for hit in scored}

    assert by_id["ru"].subject_score >= 0.8
    assert any(item["fragment_id"] == "history" and item["subject_score"] <= 0.25 for item in rejected)


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
    assert "wrong_entity_modifier_penalty" in by_id["wrong-ii"].penalties_applied
    assert by_id["wrong-ii"].score < by_id["exact"].score
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


def test_year_phrase_accepts_common_russian_abbreviation():
    scored, rejected = score_retrieval_candidates(
        "Отечественная война 1812 года",
        [
            _row(
                "year-abbreviation",
                0.7,
                "Партизан Отечественной войны 1812 г. участвовал в заграничном походе.",
                "history.pdf",
            )
        ],
    )

    assert rejected == []
    assert scored[0].matched_phrases == ["война 1812 года"]


def test_year_phrase_accepts_nearby_caption_layout_but_keeps_year_required():
    query = "Каковы причины Младотурецкой революции 1908 года?"
    scored, _ = score_retrieval_candidates(
        query,
        [
            _row(
                "relevant",
                0.75,
                (
                    "Оппозиционное движение возглавили младотурки. "
                    "Младотурецкая революция. Литография С. Христидиса. 1908 г."
                ),
                "history.pdf",
            ),
            _row(
                "wrong-year",
                0.9,
                "Младотурецкая революция завершилась в 1909 г.",
                "history.pdf",
            ),
        ],
        rerank_raw_scores=[1.0, 0.4],
        apply_noise_filter=False,
    )
    by_id = {hit.fragment_id: hit for hit in scored}

    assert by_id["relevant"].matched_phrases == ["революции 1908 года"]
    assert by_id["relevant"].missing_required_modifiers == []
    assert by_id["wrong-year"].missing_required_modifiers == ["революции 1908 года"]
    assert scored[0].fragment_id == "relevant"


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


def test_explicit_source_scope_overrides_heuristic_subject_mismatch():
    query = "Почему проводник создаёт электрический ток?"
    analysis = analyze_query(query)
    assert analysis["primary_subject"] == "physics"
    scoped = _apply_explicit_source_scope(
        analysis,
        source_uris=["chemistry.pdf"],
    )
    chemistry = _row(
        "chemistry",
        0.75,
        "Раствор хлороводорода проводит электрический ток благодаря образованию ионов.",
        "chemistry.pdf",
    )
    chemistry.subject_score = 0.12

    scored, rejected = score_retrieval_candidates(
        query,
        [chemistry],
        query_analysis=scoped,
    )

    assert not rejected
    assert scored[0].fragment_id == "chemistry"
    assert scoped["subject_filter_overridden"] is True


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


def test_section_score_uses_parent_heading_and_section_path():
    candidate = _row("ru-section", 0.4, "Выполните разбор слов и укажите признаки.", "russian.pdf")
    candidate.meta = {
        "subject": "russian_language",
        "section_title": "Морфологический разбор имени существительного",
        "section_path": ["Русский язык", "Морфологический разбор имени существительного"],
        "parent_heading": "Морфологический разбор имени существительного",
    }

    scored, rejected = score_retrieval_candidates("Морфологический разбор", [candidate])

    assert not rejected
    assert scored[0].section_score >= 0.7
    assert scored[0].subject_score >= 0.8


def test_concept_lookup_prefers_schema_or_plan_over_exercise_chunks():
    query = "Морфологический разбор"
    schema = _row(
        "schema",
        0.45,
        "Порядок морфологического разбора имени существительного. I. Часть речи. II. Морфологические признаки. III. Синтаксическая роль.",
        "russian.pdf",
    )
    schema.meta = {
        "subject": "russian_language",
        "section_title": "Морфологический разбор имени существительного",
        "section_path": ["Русский язык", "Морфологический разбор имени существительного"],
    }
    exercise = _row(
        "exercise",
        0.9,
        "408. Выполните морфологический разбор выделенных причастий. Спишите предложения и подчеркните основы.",
        "russian.pdf",
    )
    exercise.meta = {"subject": "russian_language"}

    scored, rejected = score_retrieval_candidates(query, [exercise, schema], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}

    assert not rejected
    assert by_id["schema"].chunk_type == "schema_or_plan"
    assert by_id["schema"].intent_boost_applied is True
    assert by_id["schema"].full_phrase_match is True
    assert by_id["schema"].concept_boost_applied is True
    assert by_id["schema"].schema_or_rule_boost_applied is True
    assert by_id["exercise"].chunk_type == "exercise"
    assert by_id["exercise"].exercise_penalty_applied is True
    assert by_id["exercise"].exercise_demoted_for_concept_lookup is True
    assert by_id["exercise"].full_term_boost_applied is False
    assert scored[0].fragment_id == "schema"
    assert by_id["schema"].score > by_id["exercise"].score


def test_required_term_full_phrase_prefix_for_schema_heading():
    query = "морфологический разбор"
    schema = _row(
        "schema-numeral",
        0.45,
        "Морфологический разбор имени числительного I. Часть речи. II. Морфологические признаки.",
        "russian.pdf",
    )
    schema.meta = {"subject": "russian_language"}

    scored, rejected = score_retrieval_candidates(query, [schema], apply_noise_filter=False)
    hit = scored[0]

    assert not rejected
    assert hit.full_phrase_match is True
    assert hit.missing_required_terms == []
    assert hit.required_term_score == 1.0
    assert hit.required_term_match_type == "full_phrase_prefix"
    assert hit.term_penalty_applied is False
    assert hit.full_term_boost_applied is True


def test_required_term_full_phrase_exercise_is_still_demoted_for_concept_lookup():
    query = "морфологический разбор"
    exercise = _row(
        "exercise",
        0.95,
        "Выполните морфологический разбор деепричастий из 1-го и 3-го предложений.",
        "russian.pdf",
    )
    exercise.meta = {"subject": "russian_language"}

    scored, rejected = score_retrieval_candidates(query, [exercise], apply_noise_filter=False)
    hit = scored[0]

    assert not rejected
    assert hit.full_phrase_match is True
    assert hit.chunk_type == "exercise"
    assert hit.exercise_demoted_for_concept_lookup is True
    assert hit.full_term_boost_applied is False


def test_required_term_missing_modifier_for_generic_review_definition():
    query = "морфологический разбор"
    review = _row("review", 0.95, "Рецензия - письменный разбор произведения.", "russian.pdf")
    review.meta = {"subject": "russian_language"}

    scored, rejected = score_retrieval_candidates(query, [review], apply_noise_filter=False)
    hit = scored[0]

    assert not rejected
    assert hit.full_phrase_match is False
    assert "морфологический" in hit.missing_required_terms
    assert hit.term_penalty_applied is True


def test_required_term_missing_modifier_for_wrong_analysis_type():
    query = "синтаксический разбор"
    morphology = _row("morphology", 0.95, "Морфологический разбор имени существительного.", "russian.pdf")
    morphology.meta = {"subject": "russian_language"}

    scored, rejected = score_retrieval_candidates(query, [morphology], apply_noise_filter=False)
    hit = scored[0]

    assert not rejected
    assert hit.full_phrase_match is False
    assert "синтаксический" in hit.missing_required_terms
    assert hit.term_penalty_applied is True


def test_required_term_full_phrase_for_math_theorem():
    query = "теорема Пифагора"
    theorem = _row("theorem", 0.45, "Теорема Пифагора: квадрат гипотенузы равен сумме квадратов катетов.", "math.pdf")
    theorem.meta = {"subject": "math"}

    scored, rejected = score_retrieval_candidates(query, [theorem], apply_noise_filter=False)
    hit = scored[0]

    assert not rejected
    assert hit.full_phrase_match is True
    assert hit.missing_required_terms == []
    assert hit.required_term_score == 1.0
    assert hit.required_term_match_type == "full_phrase_prefix"


def test_concept_scoring_preserves_differences_between_full_phrase_schemas_exercises_and_partials():
    query = "морфологический разбор"
    complete_schema = _row(
        "complete-schema",
        0.65,
        "Морфологический разбор глагола I. Часть речи. 1. Начальная форма. "
        "2. Постоянные признаки. 3. Непостоянные признаки. III. Синтаксическая роль. "
        "Указывается вид, переходность, возвратность, спряжение, наклонение, время, лицо или род, число.",
        "russian.pdf",
    )
    complete_schema.meta = {"subject": "russian_language", "quality_score": 1.0}
    short_schema = _row(
        "short-schema",
        0.65,
        "Морфологический разбор имени существительного I. Часть речи. Общее грамматическое значение. II. Морфологические признаки.",
        "russian.pdf",
    )
    short_schema.meta = {"subject": "russian_language", "quality_score": 0.5, "is_too_short": True}
    exercise = _row(
        "exercise",
        0.95,
        "Выполните морфологический разбор деепричастий из 1-го и 3-го предложений.",
        "russian.pdf",
    )
    exercise.meta = {"subject": "russian_language"}
    generic_definition = _row("generic-definition", 0.95, "Рецензия - письменный разбор произведения.", "russian.pdf")
    generic_definition.meta = {"subject": "russian_language"}

    scored, rejected = score_retrieval_candidates(query, [generic_definition, exercise, short_schema, complete_schema], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}

    assert not rejected
    assert by_id["complete-schema"].score >= 0.9
    assert by_id["complete-schema"].score < 1.0
    assert by_id["complete-schema"].score > by_id["short-schema"].score
    assert by_id["short-schema"].score > by_id["exercise"].score
    assert by_id["exercise"].score > by_id["generic-definition"].score
    assert by_id["complete-schema"].full_phrase_match is True
    assert by_id["complete-schema"].term_penalty_applied is False
    assert by_id["complete-schema"].concept_full_phrase_boost_value > 0
    assert by_id["complete-schema"].section_title_boost_value > 0
    assert by_id["complete-schema"].schema_or_rule_boost_value > 0
    assert by_id["short-schema"].quality_penalty_value > by_id["complete-schema"].quality_penalty_value
    assert by_id["short-schema"].boundary_penalty_value > 0
    assert by_id["exercise"].exercise_penalty_value > 0
    assert by_id["exercise"].exercise_demoted_for_concept_lookup is True
    assert by_id["generic-definition"].missing_required_terms == ["морфологический"]
    assert by_id["generic-definition"].term_penalty_applied is True


def test_morphological_concept_query_excludes_test_and_navigation_from_top_reference_hits():
    query = "Морфологический разбор"
    schema_noun = _row(
        "schema-noun",
        0.48,
        "Морфологический разбор имени существительного. I. Часть речи. II. Морфологические признаки. III. Синтаксическая роль.",
        "russian.pdf",
    )
    schema_noun.meta = {
        "subject": "russian_language",
        "section_title": "Морфологический разбор имени существительного",
        "section_path": ["Русский язык", "Морфологический разбор имени существительного"],
    }
    schema_verb = _row(
        "schema-verb",
        0.46,
        "Морфологический разбор глагола. I. Часть речи. II. Постоянные и непостоянные признаки. III. Синтаксическая роль.",
        "russian.pdf",
    )
    schema_verb.meta = {
        "subject": "russian_language",
        "section_title": "Морфологический разбор глагола",
        "section_path": ["Русский язык", "Планы разборов", "Морфологический разбор глагола"],
    }
    test_question = _row(
        "test-a4-a5",
        0.99,
        "А4. Часть речи неправильно определена. А5. Верно определены морфологические признаки у слова.",
        "russian.pdf",
    )
    test_question.meta = {"subject": "russian_language"}
    navigation = _row(
        "nav-plans",
        0.98,
        "Приложение Планы разборов 238 Морфологический разбор имени существительного 239 Морфологический разбор глагола 240",
        "russian.pdf",
    )
    navigation.page = 4
    navigation.meta = {"subject": "russian_language"}
    exercise = _row(
        "exercise",
        0.97,
        "408. Выполните морфологический разбор причастий. Спишите предложения и подчеркните основы.",
        "russian.pdf",
    )
    exercise.meta = {"subject": "russian_language"}

    scored, rejected = score_retrieval_candidates(query, [test_question, navigation, exercise, schema_noun, schema_verb], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}
    selected = select_final_hits(scored, top_k=5, min_score=0.0)
    selected_ids = [hit.fragment_id for hit in selected]

    assert not rejected
    assert selected_ids[:2] == ["schema-noun", "schema-verb"]
    assert "test-a4-a5" not in selected_ids[:5]
    assert "nav-plans" not in selected_ids
    assert by_id["test-a4-a5"].chunk_type == "test_question"
    assert by_id["test-a4-a5"].test_question_penalty_applied is True
    assert by_id["test-a4-a5"].section_score == 0
    assert by_id["nav-plans"].chunk_type == "navigation_index"
    assert by_id["nav-plans"].is_navigation_index is True
    assert by_id["schema-noun"].chunk_type == "schema_or_plan"
    assert by_id["schema-noun"].schema_boost_applied is True
    assert by_id["schema-noun"].section_score >= 0.8
    assert by_id["schema-verb"].section_score >= 0.8


def test_multiword_concept_required_term_ranks_exact_schemas_above_generic_definition():
    query = "морфологический разбор"
    review_definition = _row(
        "review-definition",
        0.99,
        "Рецензия — отзыв, письменный разбор и оценка художественного или научного произведения.",
        "russian.pdf",
    )
    review_definition.meta = {"subject": "russian_language"}
    schema_participle = _row(
        "schema-participle",
        0.45,
        "Морфологический разбор причастия. I. Часть речи. II. Морфологические признаки. III. Синтаксическая роль.",
        "russian.pdf",
    )
    schema_participle.meta = {"subject": "russian_language"}
    schema_verb = _row(
        "schema-verb",
        0.44,
        "Морфологический разбор глагола. I. Часть речи. II. Постоянные и непостоянные признаки. III. Синтаксическая роль.",
        "russian.pdf",
    )
    schema_verb.meta = {"subject": "russian_language"}
    partial = _row(
        "partial-morphology",
        0.8,
        "Морфологические признаки причастия: вид, время, возвратность и синтаксическая роль.",
        "russian.pdf",
    )
    partial.meta = {"subject": "russian_language"}
    exercise = _row(
        "exercise-exact",
        0.99,
        "408. Выполните морфологический разбор причастий. Спишите предложения и подчеркните основы.",
        "russian.pdf",
    )
    exercise.meta = {"subject": "russian_language", "is_toc": True}

    scored, rejected = score_retrieval_candidates(query, [exercise, review_definition, partial, schema_participle, schema_verb], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}
    selected = select_final_hits(scored, top_k=10, min_score=0.0)
    selected_ids = [hit.fragment_id for hit in selected]

    assert not rejected
    assert selected_ids[:2] == ["schema-participle", "schema-verb"]
    assert "review-definition" not in selected_ids[:10]
    assert by_id["schema-participle"].required_terms == ["морфологический разбор"]
    assert by_id["schema-participle"].required_term_score == 1.0
    assert by_id["schema-participle"].required_term_match_type == "full_phrase_prefix"
    assert by_id["schema-participle"].schema_boost_applied is True
    assert by_id["schema-participle"].intent_boost_applied is True
    assert by_id["schema-participle"].full_term_boost_applied is True
    assert by_id["schema-participle"].full_phrase_match is True
    assert by_id["schema-participle"].concept_boost_applied is True
    assert by_id["schema-participle"].schema_or_rule_boost_applied is True
    assert by_id["schema-participle"].section_score >= 0.8
    assert by_id["schema-participle"].inferred_section_title == "Морфологический разбор причастия"
    assert by_id["schema-verb"].required_term_score == 1.0
    assert by_id["schema-verb"].schema_boost_applied is True
    assert by_id["review-definition"].chunk_type == "definition"
    assert by_id["review-definition"].required_term_score <= 0.25
    assert by_id["review-definition"].missing_required_terms == ["морфологический"]
    assert by_id["review-definition"].intent_boost_applied is False
    assert by_id["review-definition"].term_penalty_applied is True
    assert by_id["review-definition"].section_score == 0
    assert by_id["partial-morphology"].required_term_score <= 0.25
    assert by_id["partial-morphology"].term_penalty_applied is True
    assert by_id["exercise-exact"].required_term_score == 1.0
    assert by_id["exercise-exact"].full_phrase_match is True
    assert by_id["exercise-exact"].full_term_boost_applied is False
    assert by_id["exercise-exact"].exercise_penalty_applied is True
    assert by_id["exercise-exact"].exercise_demoted_for_concept_lookup is True
    assert by_id["exercise-exact"].is_toc is False
    assert by_id["exercise-exact"].low_text_quality_reason != "toc"
    assert by_id["exercise-exact"].score < by_id["schema-participle"].score
    assert "exercise-exact" not in selected_ids[:5]


def test_morphological_concept_top5_prefers_schema_pages_over_exercises():
    query = "морфологический разбор"
    schemas = []
    for idx, (name, page) in enumerate(
        [
            ("имени существительного", 238),
            ("глагола", 239),
            ("причастия", 240),
            ("деепричастия", 240),
            ("междометия", 241),
        ],
        start=1,
    ):
        schema = _row(
            f"schema-{idx}",
            0.42,
            f"Морфологический разбор {name}. I. Часть речи. II. Морфологические признаки. III. Синтаксическая роль.",
            "russian.pdf",
        )
        schema.page = page
        schema.meta = {"subject": "russian_language"}
        schemas.append(schema)
    exercise = _row(
        "exercise-high-dense",
        0.99,
        "408. Выполните морфологический разбор выделенных слов. Спишите предложения и объясните правописание.",
        "russian.pdf",
    )
    exercise.meta = {"subject": "russian_language"}

    scored, rejected = score_retrieval_candidates(query, [exercise, *schemas], apply_noise_filter=False)
    selected = select_final_hits(scored, top_k=5, min_score=0.0)

    assert not rejected
    assert len(selected) == 5
    assert all(hit.chunk_type == "schema_or_plan" for hit in selected)
    assert all(hit.required_term_score == 1.0 for hit in selected)
    assert all(hit.schema_boost_applied for hit in selected)
    assert {hit.page for hit in selected} <= {238, 239, 240, 241}
    assert "exercise-high-dense" not in [hit.fragment_id for hit in selected]


@pytest.mark.parametrize(
    ("query", "exact_text", "generic_text"),
    [
        (
            "морфологический разбор",
            "Морфологический разбор имени числительного. I. Часть речи. II. Морфологические признаки.",
            "1, 2. См. разбор сложносочинённого предложения.",
        ),
        (
            "синтаксический разбор",
            "Синтаксический разбор предложения. I. Вид предложения. II. Грамматическая основа.",
            "Письменный разбор текста помогает понять его композицию.",
        ),
        (
            "квадратное уравнение",
            "Квадратное уравнение — это уравнение вида ax2 + bx + c = 0.",
            "Линейное уравнение и способы его решения.",
        ),
        (
            "гражданская оборона",
            "Гражданская оборона включает защиту населения и средства оповещения.",
            "Оборона государства связана с военной безопасностью.",
        ),
    ],
)
def test_multiword_concept_required_terms_prefer_full_term_over_generic_token_match(query, exact_text, generic_text):
    exact = _row("exact", 0.45, exact_text, "subject.pdf")
    generic = _row("generic", 0.95, generic_text, "subject.pdf")

    scored, rejected = score_retrieval_candidates(query, [generic, exact], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}

    assert not rejected
    assert by_id["exact"].required_term_score >= 0.8
    assert by_id["exact"].full_term_boost_applied is True
    assert by_id["generic"].required_term_score <= 0.25
    assert by_id["generic"].term_penalty_applied is True
    assert scored[0].fragment_id == "exact"


def test_exercise_lookup_allows_exercise_chunks_to_rank_high():
    query = "упражнения на морфологический разбор"
    schema = _row(
        "schema",
        0.55,
        "Порядок морфологического разбора имени существительного. I. Часть речи. II. Морфологические признаки.",
        "russian.pdf",
    )
    schema.meta = {"subject": "russian_language"}
    exercise = _row(
        "exercise",
        0.55,
        "408. Выполните морфологический разбор причастий. Найдите грамматические признаки и синтаксическую роль.",
        "russian.pdf",
    )
    exercise.meta = {"subject": "russian_language"}

    scored, rejected = score_retrieval_candidates(query, [schema, exercise], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}

    assert not rejected
    assert by_id["exercise"].chunk_type == "exercise"
    assert by_id["exercise"].intent_boost_applied is True
    assert by_id["exercise"].exercise_penalty_applied is False
    assert scored[0].fragment_id == "exercise"


def test_exercise_lookup_does_not_penalize_test_or_exercise_chunks():
    query = "упражнения на морфологический разбор"
    test_question = _row(
        "test-question",
        0.65,
        "А1. Укажите верный ответ. 1) имя существительное 2) глагол 3) союз 4) частица.",
        "russian.pdf",
    )
    test_question.meta = {"subject": "russian_language"}
    exercise = _row(
        "exercise",
        0.6,
        "408. Выполните морфологический разбор причастий. Найдите грамматические признаки.",
        "russian.pdf",
    )
    exercise.meta = {"subject": "russian_language"}
    schema = _row(
        "schema",
        0.55,
        "Морфологический разбор причастия. I. Часть речи. II. Морфологические признаки. III. Синтаксическая роль.",
        "russian.pdf",
    )
    schema.meta = {"subject": "russian_language", "section_title": "Морфологический разбор причастия"}

    scored, rejected = score_retrieval_candidates(query, [schema, exercise, test_question], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}
    selected = select_final_hits(scored, top_k=5, min_score=0.0)

    assert not rejected
    assert by_id["exercise"].exercise_penalty_applied is False
    assert by_id["test-question"].test_question_penalty_applied is False
    assert by_id["exercise"].intent_boost_applied is True
    assert by_id["test-question"].intent_boost_applied is True
    assert any(hit.chunk_type in {"exercise", "test_question"} for hit in selected[:5])


def test_syntax_analysis_concept_query_prefers_real_plan_over_random_exercise():
    query = "синтаксический разбор предложения"
    schema = _row(
        "syntax-schema",
        0.48,
        "Синтаксический разбор предложения. I. Вид предложения. II. Грамматическая основа. III. Второстепенные члены.",
        "russian.pdf",
    )
    schema.meta = {
        "subject": "russian_language",
        "section_title": "Синтаксический разбор предложения",
        "section_path": ["Русский язык", "Синтаксический разбор предложения"],
    }
    exercise = _row(
        "syntax-exercise",
        0.93,
        "317. Спишите предложения. Выполните синтаксический разбор одного предложения и подчеркните члены предложения.",
        "russian.pdf",
    )
    exercise.meta = {"subject": "russian_language"}

    scored, rejected = score_retrieval_candidates(query, [exercise, schema], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}

    assert not rejected
    assert by_id["syntax-schema"].chunk_type == "schema_or_plan"
    assert by_id["syntax-schema"].section_score >= 0.8
    assert by_id["syntax-exercise"].chunk_type == "exercise"
    assert by_id["syntax-exercise"].exercise_penalty_applied is True
    assert scored[0].fragment_id == "syntax-schema"


def test_math_concept_query_prefers_theorem_explanation_over_exercise():
    query = "теорема Пифагора"
    theorem = _row(
        "theorem",
        0.45,
        "Теорема Пифагора: квадрат гипотенузы равен сумме квадратов катетов. Формула используется в прямоугольном треугольнике.",
        "math.pdf",
    )
    theorem.meta = {"subject": "math"}
    exercise = _row(
        "exercise",
        0.98,
        "27. Решите задачу, используя теорему Пифагора. Найдите неизвестную сторону треугольника.",
        "math.pdf",
    )
    exercise.meta = {"subject": "math"}

    scored, rejected = score_retrieval_candidates(query, [exercise, theorem], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}

    assert not rejected
    assert by_id["theorem"].required_term_score >= 0.8
    assert by_id["theorem"].full_phrase_match is True
    assert by_id["theorem"].concept_boost_applied is True
    assert by_id["exercise"].chunk_type == "exercise"
    assert by_id["exercise"].exercise_demoted_for_concept_lookup is True
    assert by_id["exercise"].full_term_boost_applied is False
    assert scored[0].fragment_id == "theorem"


def test_history_concept_query_keeps_exact_modifier_above_wrong_alexander():
    query = "правление Александра I"
    exact = _row(
        "exact-i",
        0.52,
        "Правление Александра I: первые реформы, Негласный комитет и внутренняя политика.",
        "history.pdf",
    )
    exact.meta = {"subject": "history"}
    wrong = _row(
        "wrong-ii",
        0.96,
        "Правление Александра II связано с реформами второй половины XIX века.",
        "history.pdf",
    )
    wrong.meta = {"subject": "history"}
    partial = _row(
        "partial-name",
        0.93,
        "Александр проводил государственные преобразования и укреплял власть.",
        "history.pdf",
    )
    partial.meta = {"subject": "history"}

    scored, rejected = score_retrieval_candidates(query, [wrong, partial, exact], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}

    assert not rejected
    assert by_id["exact-i"].matched_phrases == ["александра i"]
    assert by_id["exact-i"].phrase_score >= 0.75
    assert by_id["wrong-ii"].wrong_entity_modifier is True
    assert by_id["wrong-ii"].score < by_id["exact-i"].score
    assert by_id["partial-name"].phrase_score <= 0.2
    assert scored[0].fragment_id == "exact-i"


def test_reference_intent_smoke_queries_rank_explanatory_chunks_before_exercises_and_toc():
    queries = [
        (
            "Александр I и его правление",
            _row("history-context", 0.55, "Александр I и его правление: реформы, Негласный комитет и внешняя политика.", "history.pdf"),
            _row("history-toc", 0.9, "Содержание Александр I и его правление 12 Отечественная война 1812 года 18", "history.pdf"),
        ),
        (
            "АСДНР при чрезвычайных ситуациях",
            _row("safety-context", 0.55, "АСДНР при чрезвычайных ситуациях включает спасательные и другие неотложные работы.", "safety.pdf"),
            _row("safety-task", 0.9, "12. Выполните задание. Назовите средства гражданской обороны и порядок действий.", "safety.pdf"),
        ),
        (
            "квадратное уравнение",
            _row("math-rule", 0.55, "Квадратное уравнение - это уравнение вида ax2 + bx + c = 0. Формула корней использует дискриминант.", "math.pdf"),
            _row("math-task", 0.9, "25. Решите квадратные уравнения и запишите ответы.", "math.pdf"),
        ),
    ]
    for query, reference, lower_priority in queries:
        reference.meta = {"subject": analyze_query(query)["primary_subject"]}
        lower_priority.meta = {"subject": analyze_query(query)["primary_subject"]}
        scored, _ = score_retrieval_candidates(query, [lower_priority, reference], apply_noise_filter=False)
        assert scored[0].fragment_id == reference.fragment_id
        assert scored[0].score > {hit.fragment_id: hit for hit in scored}[lower_priority.fragment_id].score


def test_table_of_contents_is_penalized_below_content_chunk():
    query = "Отечественная война 1812 года"
    analysis = analyze_query(query)
    toc = _row("toc", 0.8, "Оглавление\nОтечественная война 1812 года 34\nРеформы 40", "history.pdf")
    toc.meta = {"is_toc": True, "subject": "history"}
    toc.subject_score = 0.9
    content = _row("content", 0.55, "Отечественная война 1812 года: причины, ход войны и Бородинское сражение.", "history.pdf")
    content.subject_score = 0.9

    scored, _ = score_retrieval_candidates(query, [toc, content], query_analysis=analysis, apply_noise_filter=False)
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


def test_numbered_exposition_is_not_misclassified_as_table_of_contents():
    text = (
        "Факторы размещения металлургических предприятий. "
        "1. Материалоёмкость производства является главным фактором: предприятия "
        "размещают вблизи источников сырья, так как перевозка руды увеличивает затраты. "
        "2. Производство использует много топлива и воды, поэтому учитывается "
        "энергетический фактор. Например, комбинаты полного цикла тяготеют к районам "
        "добычи железной руды и коксующегося угля. В результате складываются крупные "
        "металлургические базы России. Данные в таблице 12 показывают долю сырья 40, "
        "топлива 25 и электроэнергии 18 процентов."
    )

    assert is_toc_text(text, page=40) is False
    assert detect_chunk_type(text, page=40) not in {"navigation_index", "toc"}
    assert text_quality_flags(text, page=40)["low_text_quality"] is False


def test_substantive_numbered_plan_is_not_navigation():
    text = (
        "План синтаксического разбора сложноподчинённого предложения. "
        "1. Определить вид предложения по цели высказывания и эмоциональной окраске. "
        "2. Выделить грамматические основы и указать средства связи частей. "
        "3. Назвать главное и придаточное предложения. "
        "4. Объяснить расположение придаточного и расстановку знаков препинания. "
        "5. Построить схему предложения. "
        "6. Разобрать каждую часть как простое предложение."
    )

    assert detect_chunk_type(text, page=280) == "schema_or_plan"
    assert text_quality_flags(text, page=280)["low_text_quality"] is False


def test_runtime_content_overrides_stale_hard_index_metadata():
    text = (
        "План синтаксического разбора сложноподчинённого предложения. "
        "1. Определить вид предложения. 2. Выделить грамматические основы. "
        "3. Указать главное и придаточное предложения. 4. Объяснить знаки препинания. "
        "5. Построить схему. 6. Разобрать каждую часть предложения. "
        "Этот порядок используется для полного письменного разбора и помогает "
        "последовательно установить строение сложного предложения."
    )
    row = _row("stale-plan", 0.7, text, "russian.pdf")
    row.page = 280
    row.meta = {
        "is_toc": True,
        "low_text_quality": True,
        "quality_score": 0.25,
        "chunk_type": "navigation_index",
        "is_navigation_index": True,
    }

    metadata = _metadata_with_inferred_section(row)
    quality = _candidate_quality(row)

    assert metadata["chunk_type"] == "schema_or_plan"
    assert metadata["is_toc"] is False
    assert metadata["is_navigation_index"] is False
    assert quality["chunk_type"] == "schema_or_plan"
    assert quality["is_toc"] is False
    assert quality["is_navigation_index"] is False
    assert quality["low_text_quality"] is False
    assert quality["quality_score"] >= 0.75


def test_low_quality_fragmented_chunk_is_filtered_from_final_hits():
    query = "Морфологический разбор"
    broken = _row("broken", 0.9, "пинания. Выполните морфологический разбор причастий.", "russian.pdf")
    broken.meta = {"subject": "russian_language"}
    good = _row(
        "good",
        0.55,
        "Морфологический разбор причастия включает определение грамматических признаков и синтаксической роли.",
        "russian.pdf",
    )
    good.meta = {"subject": "russian_language"}

    assert text_quality_flags(broken.text)["low_text_quality"] is True
    scored, _ = score_retrieval_candidates(query, [broken, good], apply_noise_filter=False)
    by_id = {hit.fragment_id: hit for hit in scored}
    selected = select_final_hits(scored, top_k=5, min_score=0.0)

    assert by_id["broken"].low_text_quality is True
    assert by_id["broken"].quality_score < by_id["good"].quality_score
    assert "broken" not in [hit.fragment_id for hit in selected]
    assert selected[0].fragment_id == "good"


def test_multi_query_fusion_uses_weighted_max_score():
    calls = {
        "q1": [_row("a", 1.0, "a text"), _row("b", 0.7, "b text")],
        "q2": [_row("a", 0.9, "a text"), _row("c", 0.8, "c text")],
    }
    merged = retrieve_multi_query(retrieve_fn=lambda q: calls[q], queries=["q1", "q2"])

    assert set(merged) == {"a", "b", "c"}
    assert merged["a"][0] == 1.0
    assert merged["c"][0] == pytest.approx(0.72)


def test_balanced_prerank_uses_expansion_terms_for_dense_only_candidate():
    generic = []
    for idx in range(8):
        row = _row(
            f"generic-{idx}",
            0.75,
            "Общие сведения о предложении и средствах связи.",
        )
        row.rrf_score = 0.95 - idx * 0.01
        row.lexical_score = 0.8
        generic.append(row)
    answer = _row(
        "comparison-answer",
        0.68,
        (
            "Сравнение выражается именем существительным, сравнительной степенью "
            "и сравнительным оборотом."
        ),
    )
    answer.rrf_score = 0.5
    answer.lexical_score = 0.0

    selected = _balanced_prerank(
        [*generic, answer],
        limit=6,
        query_terms=["сравнение", "выражается"],
        query_term_groups=[
            ["сравнение", "выражается"],
            ["сравнительный", "оборот", "степень"],
        ],
    )

    assert "comparison-answer" in {row.fragment_id for row in selected}


def test_keyword_rerank_penalizes_irrelevant_hits_without_query_hardcode():
    hits = [
        _row("good", 0.7, "Османская империя, султан и Стамбул"),
        _row("bad", 0.7, "Британская армия и реформы промышленности"),
    ]

    reranked = rerank_by_keyword_relevance("османская империя", hits)

    assert reranked[0].fragment_id == "good"
    assert reranked[0].score > reranked[1].score
    assert all(0.0 <= hit.score <= 1.0 for hit in reranked)


def test_goal_query_promotes_direct_answer_and_penalizes_missing_entity():
    query = "Какие цели преследовали реформы Танзимат в Османской империи?"
    direct = _row(
        "direct",
        0.78,
        (
            "Эпоха Танзимата началась в Османской империи. "
            "Целью реформ стало укрепление центральной власти, успокоение Балкан "
            "и ослабление зависимости от Европы."
        ),
    )
    overview = _row(
        "overview",
        0.84,
        (
            "Реформы Танзимата в Османской империи проходили в XIX веке. "
            + "Хроника событий и международных договоров. " * 40
            + "Цель другого государства заключалась в расширении торговли."
        ),
    )
    unrelated = _row(
        "unrelated",
        0.86,
        "Целью политики Бисмарка было объединение германских земель и укрепление Пруссии.",
    )

    scored, rejected = score_retrieval_candidates(
        query,
        [overview, unrelated, direct],
        rerank_raw_scores=[0.05, 0.0, 1.0],
        apply_noise_filter=False,
    )

    assert not rejected
    assert scored[0].fragment_id == "direct"
    assert scored[0].answer_focus == "goal"
    assert scored[0].answer_alignment_score > 0.0
    assert scored[0].answer_alignment_boost_value > 0.0
    assert scored[0].required_entity_score == 1.0
    unrelated_hit = next(hit for hit in scored if hit.fragment_id == "unrelated")
    assert unrelated_hit.entity_penalty_value > 0.0
    assert "missing_required_entity_penalty" in unrelated_hit.penalties_applied
    assert unrelated_hit.final_score < scored[0].final_score


def test_anti_noise_filter_drops_zero_overlap_low_dense_candidates():
    candidates = [
        _row("history", 0.45, "Александр I. Реформы Сперанского и Негласный комитет.", "history.pdf"),
        _row("noise", 0.5, "Одуванчики, техника безопасности и бытовые инструкции.", "biology.pdf"),
    ]

    scored, rejected = score_retrieval_candidates("Александр I и его правление", candidates)

    assert [row.fragment_id for row in scored] == ["history"]
    assert rejected[0]["fragment_id"] == "noise"
    assert rejected[0]["rejection_reason"] in {"low_lexical_overlap", "low_document_score"}


def test_unknown_subject_rejects_single_generic_word_despite_high_reranker_score():
    candidate = _row(
        "swiftui-transition",
        0.62,
        "Переход от одной общественной формации к другой происходит в результате революции.",
        "history.pdf",
    )

    scored, rejected = score_retrieval_candidates(
        "Как реализовать анимацию перехода экрана в SwiftUI?",
        [candidate],
        rerank_raw_scores=[10.0],
    )

    assert scored == []
    assert rejected[0]["rejection_reason"] == "out_of_domain_low_evidence"


def test_unknown_subject_keeps_well_supported_corpus_evidence():
    candidate = _row(
        "local-topic",
        0.62,
        (
            "Редкий локальный термин «кварцитизация» описывает кварцитизацию породы. "
            "Процесс кварцитизации подробно рассматривается в этом разделе учебника."
        ),
        "local.pdf",
    )

    scored, rejected = score_retrieval_candidates(
        "Что означает редкий локальный термин кварцитизация?",
        [candidate],
        rerank_raw_scores=[10.0],
    )

    assert rejected == []
    assert scored[0].fragment_id == "local-topic"


def test_explicit_source_scope_bypasses_unknown_subject_ood_gate():
    candidate = _row(
        "scoped-transition",
        0.62,
        "Переход от одной общественной формации к другой происходит в результате революции.",
        "history.pdf",
    )
    analysis = analyze_query("Как реализовать анимацию перехода экрана в SwiftUI?")
    analysis["subject_filter_overridden"] = True

    scored, rejected = score_retrieval_candidates(
        "Как реализовать анимацию перехода экрана в SwiftUI?",
        [candidate],
        rerank_raw_scores=[10.0],
        query_analysis=analysis,
    )

    assert all(item["rejection_reason"] != "out_of_domain_low_evidence" for item in rejected)


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


def test_adaptive_threshold_keeps_strong_lexical_answer_below_score_tail():
    head = _row("head", 0.9, "Общее объяснение темы")
    head.final_score = 0.9
    answer = _row(
        "answer",
        0.72,
        "Придаточные предложения могут быть синтаксическими синонимами членов простого предложения.",
    )
    answer.final_score = 0.38
    answer.lexical_overlap = 0.86
    answer.lexical_score = 0.84
    answer.phrase_score = 0.68
    answer.rerank_score = 0.48
    answer.chunk_type = "explanatory"
    tail = _row("tail", 0.3, "Слабое совпадение")
    tail.final_score = 0.3

    selected = apply_adaptive_threshold([head, answer, tail])

    assert [hit.fragment_id for hit in selected] == ["head", "answer"]


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
