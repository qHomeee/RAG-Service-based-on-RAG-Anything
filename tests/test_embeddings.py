import builtins

import pytest

from app.config import settings
from app.embeddings import EmbeddingProvider


def _mock_sentence_transformers_import_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise ImportError("cannot import name split_torch_state_dict_into_shards from huggingface_hub")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_embedding_provider_degraded_mode_when_failoff(monkeypatch, caplog):
    original = settings.fail_on_embedding_fallback
    settings.fail_on_embedding_fallback = False
    _mock_sentence_transformers_import_error(monkeypatch)

    try:
        with caplog.at_level("WARNING", logger="rag_service"):
            provider = EmbeddingProvider()
        assert provider.using_fallback is True
        assert any("embedding_provider_degraded" in rec.message for rec in caplog.records)
        assert any("embedding_dependency_mismatch" in rec.message for rec in caplog.records)
    finally:
        settings.fail_on_embedding_fallback = original


def test_embedding_provider_hard_fail_in_strict_mode(monkeypatch):
    original = settings.fail_on_embedding_fallback
    settings.fail_on_embedding_fallback = True
    _mock_sentence_transformers_import_error(monkeypatch)

    try:
        with pytest.raises(RuntimeError) as exc_info:
            EmbeddingProvider()
        text = str(exc_info.value)
        assert "Embedding provider failed to initialize in strict mode" in text
        assert "requirements-accelerate.txt" in text
    finally:
        settings.fail_on_embedding_fallback = original
