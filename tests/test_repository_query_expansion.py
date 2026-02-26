from app.config import settings
from app.repository import _expand_query


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
