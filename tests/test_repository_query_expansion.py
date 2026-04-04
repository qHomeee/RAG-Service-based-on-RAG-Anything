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
