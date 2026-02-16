from fastapi.testclient import TestClient

from app.main import app, get_service


class FakeService:
    def retrieve(self, query: str, top_k: int, min_score: float, collection: str, return_text: bool):
        return [
            {
                "fragment_id": "frag-1",
                "source_uri": "textbooks/econ.pdf",
                "title": "econ.pdf",
                "type": "text",
                "page": None,
                "snippet": "Inflation is sustained rise in price level.",
                "score": 0.9,
                "text": "Inflation is sustained rise in price level." if return_text else None,
            }
        ]

    def query(self, query: str, top_k: int, min_score: float, collection: str):
        raise NotImplementedError


def override_get_service():
    return FakeService()


app.dependency_overrides[get_service] = override_get_service
client = TestClient(app)


def test_ingest_endpoint():
    response = client.post(
        "/ingest",
        headers={"X-API-Key": "change-me"},
    )
    assert response.status_code == 410
    body = response.json()
    assert "Ingest через API отключен" in body["error"]


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
