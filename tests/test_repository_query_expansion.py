from app.config import settings
from app.repository import _expand_query, expand_query


def test_expand_query_adds_synonyms_when_enabled():
    original = settings.query_expansion_enabled
    settings.query_expansion_enabled = True
    try:
        expanded = _expand_query("инфляция")
        assert "рост цен" in expanded
        assert expanded.startswith("инфляция")
    finally:
        settings.query_expansion_enabled = original


def test_expand_query_noop_when_disabled():
    original = settings.query_expansion_enabled
    settings.query_expansion_enabled = False
    try:
        text = "инфляция"
        assert _expand_query(text) == text
    finally:
        settings.query_expansion_enabled = original


def test_expand_query_uses_collection_specific_dictionary():
    original = settings.query_synonyms_by_collection
    settings.query_synonyms_by_collection = {
        "edu": {
            "реформа": ["модернизация", "преобразование"],
        }
    }
    try:
        variants = expand_query("реформа", collection="edu", source_uris=None)
        assert any("модернизация" in item for item in variants)
    finally:
        settings.query_synonyms_by_collection = original


def test_expand_query_uses_domain_specific_dictionary():
    original = settings.query_synonyms_by_domain
    settings.query_synonyms_by_domain = {
        "example.org": {
            "налог": ["фискальный сбор"],
        }
    }
    try:
        variants = expand_query("налог", source_uris=["https://example.org/docs/tax"])
        assert any("фискальный сбор" in item for item in variants)
    finally:
        settings.query_synonyms_by_domain = original


def test_expand_query_adds_answer_focused_variant_for_long_causal_question():
    variants = expand_query(
        "каковы причины младотурецкой революции 1908 года",
        query_analysis={
            "answer_focus": "cause",
            "primary_subject": "history",
            "detected_subjects": [],
        },
    )

    assert len(variants) >= 2
    assert any(
        "младотурецкой" in variant
        and "кризис" in variant
        and "оппозиция" in variant
        and "1908" not in variant
        for variant in variants[1:]
    )


def test_expand_query_adds_consequence_cues_without_auxiliary_verb():
    variants = expand_query(
        "какие последствия имела младотурецкая революция",
        query_analysis={
            "answer_focus": "consequence",
            "primary_subject": "history",
            "detected_subjects": [],
        },
    )

    assert any(
        "вызвало" in variant
        and "ухудшение" in variant
        and "имела" not in variant
        for variant in variants[1:]
    )


def test_expand_query_recognizes_implicit_goal_wording():
    variants = expand_query("кто такие младотурки и чего они добивались")

    assert any(
        "младотурки" in variant
        and "цели" in variant
        and "добивались" not in variant
        for variant in variants[1:]
    )
