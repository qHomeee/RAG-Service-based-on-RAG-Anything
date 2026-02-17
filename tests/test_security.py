from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app, get_service


class FakeService:
    def ingest(self, input_path: str, collection: str, reindex: bool):
        return {"indexed_docs": 0, "indexed_fragments": 0, "indexed_vectors": 0}

    def retrieve(self, query: str, top_k: int, min_score: float, collection: str, source_uris, return_text: bool):
        return []

    def query(self, query: str, top_k: int, min_score: float, collection: str, source_uris):
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
            headers={"X-Admin-API-Key": "change-me-admin"},
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
            headers={"X-API-Key": "change-me"},
            json={"collection": "default"},
        )
        assert response1.status_code == 200

        response2 = client.post(
            "/sources",
            headers={"X-API-Key": "change-me"},
            json={"collection": "default"},
        )
        assert response2.status_code == 429
    finally:
        app.dependency_overrides.clear()
