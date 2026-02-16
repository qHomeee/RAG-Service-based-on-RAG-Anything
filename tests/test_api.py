from fastapi.testclient import TestClient

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
            }
        ]

    def query(self, query: str, top_k: int, min_score: float, collection: str, source_uris: list[str] | None):
        raise NotImplementedError

    def list_sources(self, collection: str):
        return [{"source_uri": "textbooks/econ.pdf", "title": "econ.pdf"}]


def override_get_service():
    return FakeService()


app.dependency_overrides[get_service] = override_get_service
client = TestClient(app)


def test_ingest_endpoint():
    response = client.post(
        "/ingest",
        headers={"X-API-Key": "change-me"},
        json={"input_path": "storage/raw", "collection": "default", "reindex": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["indexed_docs"] == 1
    assert body["indexed_fragments"] == 2


def test_retrieve_endpoint():
    response = client.post(
        "/retrieve",
        headers={"X-API-Key": "change-me"},
        json={"query": "inflation", "top_k": 5, "min_score": 0.2, "collection": "default"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["hits"]) == 1
    assert body["hits"][0]["fragment_id"] == "frag-1"


def test_retrieve_endpoint_with_source_filter():
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
    assert response.status_code == 200
    body = response.json()
    assert body["hits"][0]["source_uri"] == "textbooks/russian.pdf"


def test_sources_endpoint():
    response = client.post(
        "/sources",
        headers={"X-API-Key": "change-me"},
        json={"collection": "default"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["source_uri"] == "textbooks/econ.pdf"
