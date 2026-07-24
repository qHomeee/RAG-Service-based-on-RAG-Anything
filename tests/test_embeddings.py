import builtins
import sys
from types import ModuleType

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


def _mock_sentence_transformer_class(monkeypatch, fake_class):
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = fake_class
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_embedding_provider_degraded_mode_when_failoff(monkeypatch, caplog):
    monkeypatch.setattr(settings, "fail_on_embedding_fallback", False)
    monkeypatch.setattr(settings, "embed_offline", False)
    monkeypatch.setattr(settings, "embed_model", "all-MiniLM-L6-v2")
    _mock_sentence_transformers_import_error(monkeypatch)

    with caplog.at_level("WARNING", logger="rag_service"):
        provider = EmbeddingProvider()
    assert provider.using_fallback is True
    assert any("embedding_provider_degraded" in rec.message for rec in caplog.records)
    assert any("embedding_dependency_mismatch" in rec.message for rec in caplog.records)


def test_embedding_provider_hard_fail_in_strict_mode(monkeypatch):
    monkeypatch.setattr(settings, "fail_on_embedding_fallback", True)
    monkeypatch.setattr(settings, "embed_offline", False)
    monkeypatch.setattr(settings, "embed_model", "all-MiniLM-L6-v2")
    _mock_sentence_transformers_import_error(monkeypatch)

    with pytest.raises(RuntimeError) as exc_info:
        EmbeddingProvider()
    text = str(exc_info.value)
    assert "Embedding model is not available" in text
    assert "FAIL_ON_EMBEDDING_FALLBACK=true" in text
    assert "requirements-accelerate.txt" in text


def test_embedding_provider_missing_local_path_has_actionable_error(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing-model"
    monkeypatch.setattr(settings, "fail_on_embedding_fallback", True)
    monkeypatch.setattr(settings, "embed_model", str(missing_model))
    monkeypatch.setattr(settings, "embed_offline", True)

    with pytest.raises(RuntimeError) as exc_info:
        EmbeddingProvider()
    text = str(exc_info.value)
    assert "Embedding model is not available" in text
    assert f"Configured EMBED_MODEL={missing_model}" in text
    assert "Model source: local path" in text
    assert "EMBED_OFFLINE=true" in text
    assert "Local embedding model path does not exist" in text
    assert "SentenceTransformer" in text
    assert "storage/models/all-MiniLM-L6-v2" in text


def test_embedding_provider_invalid_local_folder_has_actionable_error(monkeypatch, tmp_path):
    model_dir = tmp_path / "invalid-model"
    model_dir.mkdir()
    monkeypatch.setattr(settings, "fail_on_embedding_fallback", True)
    monkeypatch.setattr(settings, "embed_model", str(model_dir))
    monkeypatch.setattr(settings, "embed_offline", True)

    with pytest.raises(RuntimeError) as exc_info:
        EmbeddingProvider()
    text = str(exc_info.value)
    assert "Embedding model is not available" in text
    assert "Model source: local path" in text
    assert "EMBED_OFFLINE=true" in text
    assert "missing modules.json" in text
    assert "SentenceTransformer(...).save(...)" in text


def test_embedding_provider_existing_local_path_uses_local_files_only(monkeypatch, tmp_path):
    model_dir = tmp_path / "all-MiniLM-L6-v2"
    model_dir.mkdir()
    (model_dir / "modules.json").write_text("[]", encoding="utf-8")
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            calls.append((model_name_or_path, kwargs))

    _mock_sentence_transformer_class(monkeypatch, FakeSentenceTransformer)
    monkeypatch.setattr(settings, "fail_on_embedding_fallback", True)
    monkeypatch.setattr(settings, "embed_model", str(model_dir))
    monkeypatch.setattr(settings, "embed_offline", False)

    provider = EmbeddingProvider()

    assert provider.using_fallback is False
    assert provider.model_fingerprint.startswith("sha256:")
    assert calls == [(str(model_dir.resolve()), {"local_files_only": True})]


def test_embedding_provider_caches_repeated_text(monkeypatch, tmp_path):
    model_dir = tmp_path / "cached-model"
    model_dir.mkdir()
    (model_dir / "modules.json").write_text("[]", encoding="utf-8")
    encode_calls = []

    class Encoded:
        def tolist(self):
            return [0.0] * settings.embed_dim

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            pass

        def encode(self, text, **kwargs):
            encode_calls.append(text)
            return Encoded()

    _mock_sentence_transformer_class(monkeypatch, FakeSentenceTransformer)
    monkeypatch.setattr(settings, "fail_on_embedding_fallback", True)
    monkeypatch.setattr(settings, "embed_model", str(model_dir))
    provider = EmbeddingProvider()

    assert provider.embed("одинаковый запрос") == provider.embed("одинаковый запрос")
    assert encode_calls == ["одинаковый запрос"]


def test_embedding_provider_encodes_batches(monkeypatch, tmp_path):
    model_dir = tmp_path / "batch-model"
    model_dir.mkdir()
    (model_dir / "modules.json").write_text("[]", encoding="utf-8")
    encode_calls = []

    class Encoded:
        def tolist(self):
            return [[0.1] * settings.embed_dim, [0.2] * settings.embed_dim]

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            encode_calls.append((texts, kwargs))
            return Encoded()

    _mock_sentence_transformer_class(monkeypatch, FakeSentenceTransformer)
    monkeypatch.setattr(settings, "fail_on_embedding_fallback", True)
    monkeypatch.setattr(settings, "embed_model", str(model_dir))
    provider = EmbeddingProvider()

    vectors = provider.embed_many(["первый", "второй"], batch_size=7)

    assert len(vectors) == 2
    assert len(vectors[0]) == settings.embed_dim
    assert encode_calls == [
        (
            ["первый", "второй"],
            {
                "batch_size": 7,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
        )
    ]


def test_embedding_provider_rejects_invalid_batch_dimension(monkeypatch, tmp_path):
    model_dir = tmp_path / "invalid-batch-model"
    model_dir.mkdir()
    (model_dir / "modules.json").write_text("[]", encoding="utf-8")

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            return [[0.0] * 12 for _ in texts]

    _mock_sentence_transformer_class(monkeypatch, FakeSentenceTransformer)
    monkeypatch.setattr(settings, "fail_on_embedding_fallback", True)
    monkeypatch.setattr(settings, "embed_model", str(model_dir))
    provider = EmbeddingProvider()

    with pytest.raises(RuntimeError, match="Embedding batch mismatch"):
        provider.embed_many(["текст"])


def test_embedding_provider_offline_huggingface_id_passes_local_files_only(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            calls.append((model_name_or_path, kwargs))
            raise OSError("local files only and model is not cached")

    _mock_sentence_transformer_class(monkeypatch, FakeSentenceTransformer)
    monkeypatch.setattr(settings, "fail_on_embedding_fallback", True)
    monkeypatch.setattr(settings, "embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setattr(settings, "embed_offline", True)

    with pytest.raises(RuntimeError) as exc_info:
        EmbeddingProvider()

    assert calls == [("sentence-transformers/all-MiniLM-L6-v2", {"local_files_only": True})]
    text = str(exc_info.value)
    assert "Model source: HuggingFace model id" in text
    assert "EMBED_OFFLINE=true" in text
    assert "local files only and model is not cached" in text
