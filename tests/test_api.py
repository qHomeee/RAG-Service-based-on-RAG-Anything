from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app, get_service


class FakeService:
    def ingest(self, input_path: str, collection: str, reindex: bool):
        return {"indexed_docs": 1, "indexed_fragments": 2, "indexed_vectors": 3}

    def retrieve(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
        return_text: bool,
        include_toc: bool = False,
    ):
        return [
            {
                "fragment_id": "frag-1",
                "source_uri": source_uris[0] if source_uris else "textbooks/econ.pdf",
                "title": "econ.pdf",
                "type": "text",
                "page": None,
                "snippet": "Inflation is sustained rise in price level.",
                "score": 0.9,
                "text": "Inflation is sustained rise in price level." if return_text else None,
                "dense_score": 0.8,
                "lexical_score": 0.7,
                "phrase_score": 0.6,
                "subject_score": 0.9,
                "section_score": 0.5,
                "rerank_score": 0.0,
                "final_score": 0.9,
                "lexical_overlap": 0.5,
                "document_score": 0.6,
                "rrf_score": 0.4,
                "exact_phrases": ["inflation"],
                "matched_phrases": ["inflation"],
                "missing_required_modifiers": [],
                "wrong_entity_modifier": False,
                "phrase_score_before_penalty": 0.6,
                "phrase_score_after_penalty": 0.6,
                "is_toc": False,
                "toc_filtered": False,
                "toc_penalty_applied": False,
            }
        ]

    def retrieve_with_debug(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
        return_text: bool,
        include_toc: bool = False,
    ):
        hits = self.retrieve(query, top_k, min_score, collection, source_uris, return_text, include_toc)
        return hits, {
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
            "dense_candidates": 1,
            "lexical_candidates": 1,
            "candidates_after_fusion": 1,
            "candidates_after_threshold": 1,
            "rejected_results": [
                {
                    "fragment_id": "bad",
                    "source_uri": "noise.pdf",
                    "rejection_reason": "low_lexical_overlap",
                    "dense_score": 0.7,
                    "lexical_score": 0.0,
                    "phrase_score": 0.0,
                    "subject_score": 0.1,
                    "section_score": 0.0,
                    "rerank_score": 0.0,
                    "final_score": 0.1,
                    "lexical_overlap": 0.0,
                    "document_score": 0.0,
                    "rrf_score": 0.0,
                    "exact_phrases": ["inflation"],
                    "matched_phrases": [],
                    "missing_required_modifiers": ["inflation"],
                    "wrong_entity_modifier": False,
                    "phrase_score_before_penalty": 0.0,
                    "phrase_score_after_penalty": 0.0,
                    "is_toc": False,
                    "toc_filtered": False,
                    "toc_penalty_applied": False,
                }
            ],
        }

    def query(self, query: str, top_k: int, min_score: float, collection: str, source_uris: list[str] | None):
        raise NotImplementedError

    def list_sources(self, collection: str):
        return [{"source_uri": "textbooks/econ.pdf", "title": "econ.pdf"}]


def override_get_service():
    return FakeService()


def _client() -> TestClient:
    app.dependency_overrides[get_service] = override_get_service
    return TestClient(app)


def test_ingest_endpoint(tmp_path: Path):
    client = _client()
    original_setting = settings.ingest_path_must_be_under_storage_raw
    settings.ingest_path_must_be_under_storage_raw = False
    try:
        response = client.post(
            "/ingest",
            headers={"X-Admin-API-Key": "change-me-admin"},
            json={"input_path": str(tmp_path), "collection": "default", "reindex": False},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["indexed_docs"] == 1
        assert body["indexed_fragments"] == 2
    finally:
        settings.ingest_path_must_be_under_storage_raw = original_setting
        app.dependency_overrides.clear()


def test_retrieve_endpoint():
    client = _client()
    response = client.post(
        "/retrieve",
        headers={"X-API-Key": "change-me"},
        json={"query": "inflation", "top_k": 5, "min_score": 0.2, "collection": "default"},
    )
    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert len(body["hits"]) == 1
    assert body["hits"][0]["fragment_id"] == "frag-1"


def test_retrieve_endpoint_with_source_filter():
    client = _client()
    response = client.post(
        "/retrieve",
        headers={"X-API-Key": "change-me"},
        json={
            "query": "inflation",
            "top_k": 5,
            "min_score": 0.2,
            "collection": "default",
            "source_uris": ["textbooks/russian.pdf"],
        },
    )
    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["hits"][0]["source_uri"] == "textbooks/russian.pdf"


def test_retrieve_endpoint_debug_returns_component_scores_and_rejections():
    client = _client()
    response = client.post(
        "/retrieve",
        headers={"X-API-Key": "change-me"},
        json={"query": "inflation", "top_k": 5, "min_score": 0.35, "collection": "default", "debug": True},
    )
    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    hit = body["hits"][0]
    for field in [
        "dense_score",
            "lexical_score",
            "phrase_score",
            "subject_score",
            "section_score",
            "rerank_score",
        "final_score",
        "lexical_overlap",
        "document_score",
        "rrf_score",
        "exact_phrases",
        "matched_phrases",
        "missing_required_modifiers",
        "wrong_entity_modifier",
        "phrase_score_before_penalty",
        "phrase_score_after_penalty",
        "is_toc",
        "toc_filtered",
        "toc_penalty_applied",
    ]:
        assert hit[field] is not None
    assert hit["phrase_score"] == hit["phrase_score_after_penalty"]
    assert body["debug"]["rejected_results"][0]["rejection_reason"] == "low_lexical_overlap"


def test_sources_endpoint():
    client = _client()
    response = client.post(
        "/sources",
        headers={"X-API-Key": "change-me"},
        json={"collection": "default"},
    )
    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["source_uri"] == "textbooks/econ.pdf"


def test_metrics_endpoint():
    client = _client()
    response = client.get("/metrics", headers={"X-API-Key": "change-me"})
    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert "slo" in body
    assert "p95_latency_ms" in body["slo"]


def test_readyz_endpoint_shape():
    client = _client()
    response = client.get("/readyz", headers={"X-API-Key": "change-me"})
    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "checks" in body
    assert "reranker_loaded" in body["checks"]
    assert "reranker_model" in body["checks"]
    assert "reranker_error" in body["checks"]



def test_get_requires_api_key():
    client = _client()
    response = client.get("/healthz")
    app.dependency_overrides.clear()
    assert response.status_code == 401


def test_ingest_requires_admin_api_key(tmp_path: Path):
    client = _client()
    original_setting = settings.ingest_path_must_be_under_storage_raw
    settings.ingest_path_must_be_under_storage_raw = False
    try:
        response = client.post(
            "/ingest",
            headers={"X-API-Key": "change-me"},
            json={"input_path": str(tmp_path), "collection": "default", "reindex": False},
        )
        assert response.status_code == 401
    finally:
        settings.ingest_path_must_be_under_storage_raw = original_setting
        app.dependency_overrides.clear()
