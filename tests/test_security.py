from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import _validate_secure_settings, app, get_service


class FakeService:
    def ingest(self, input_path: str, collection: str, reindex: bool):
        return {"indexed_docs": 0, "indexed_fragments": 0, "indexed_vectors": 0}

    def retrieve(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris,
        return_text: bool,
        include_toc: bool = False,
        include_low_quality: bool = False,
        include_navigation: bool = False,
    ):
        return []

    def query(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris,
        return_sources: bool = True,
    ):
        return {"answer": "ok", "sources": []}

    def list_sources(self, collection: str):
        return []


class DummyLimiter:
    def __init__(self):
        self._calls = 0

    def check(self, key: str) -> None:
        self._calls += 1
        if self._calls > 1:
            from fastapi import HTTPException

            raise HTTPException(status_code=429, detail="Rate limit exceeded")


def override_get_service():
    return FakeService()


def _client() -> TestClient:
    app.dependency_overrides[get_service] = override_get_service
    return TestClient(app)


def test_ingest_path_must_exist(tmp_path: Path):
    client = _client()
    original = settings.ingest_path_must_be_under_storage_raw
    settings.ingest_path_must_be_under_storage_raw = False
    try:
        missing = tmp_path / "missing"
        response = client.post(
            "/ingest",
            headers={"X-Admin-API-Key": settings.admin_api_key},
            json={"input_path": str(missing), "collection": "default", "reindex": False},
        )
        assert response.status_code == 400
    finally:
        settings.ingest_path_must_be_under_storage_raw = original
        app.dependency_overrides.clear()


def test_rate_limit_exceeded(monkeypatch):
    client = _client()
    limiter = DummyLimiter()
    monkeypatch.setattr("app.security.rate_limiter", limiter)

    try:
        response1 = client.post(
            "/sources",
            headers={"X-API-Key": settings.api_key},
            json={"collection": "default"},
        )
        assert response1.status_code == 200

        response2 = client.post(
            "/sources",
            headers={"X-API-Key": settings.api_key},
            json={"collection": "default"},
        )
        assert response2.status_code == 429
    finally:
        app.dependency_overrides.clear()


def test_production_requires_embedding_compatibility_guard(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "api_key", "a" * 32)
    monkeypatch.setattr(settings, "admin_api_key", "b" * 32)
    monkeypatch.setattr(settings, "uvicorn_workers", 1)
    monkeypatch.setattr(settings, "embed_offline", True)
    monkeypatch.setattr(settings, "reranker_offline", True)
    monkeypatch.setattr(settings, "fail_on_embedding_fallback", True)
    monkeypatch.setattr(settings, "enforce_embedding_model_compatibility", False)
    monkeypatch.setattr(settings, "auto_create_schema", False)

    with pytest.raises(RuntimeError, match="ENFORCE_EMBEDDING_MODEL_COMPATIBILITY"):
        _validate_secure_settings()


def test_production_rejects_example_placeholder_secret(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "api_key", "replace-with-at-least-32-random-characters")

    with pytest.raises(RuntimeError, match="API_KEY"):
        _validate_secure_settings()


def test_production_rejects_wildcard_allowed_hosts(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "api_key", "a" * 32)
    monkeypatch.setattr(settings, "admin_api_key", "b" * 32)
    monkeypatch.setattr(settings, "database_url", "postgresql+psycopg://rag:secret@postgres/rag")
    monkeypatch.setattr(settings, "uvicorn_workers", 1)
    monkeypatch.setattr(settings, "embed_offline", True)
    monkeypatch.setattr(settings, "reranker_offline", True)
    monkeypatch.setattr(settings, "fail_on_embedding_fallback", True)
    monkeypatch.setattr(settings, "enforce_embedding_model_compatibility", True)
    monkeypatch.setattr(settings, "auto_create_schema", False)
    monkeypatch.setattr(settings, "allowed_hosts", ["*"])

    with pytest.raises(RuntimeError, match="ALLOWED_HOSTS"):
        _validate_secure_settings()
