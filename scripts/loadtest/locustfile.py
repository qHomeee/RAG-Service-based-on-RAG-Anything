import os

from locust import HttpUser, between, task


class RAGUser(HttpUser):
    wait_time = between(0.2, 1.0)
    api_key = os.environ.get("API_KEY")

    def on_start(self):
        if not self.api_key:
            raise RuntimeError("API_KEY environment variable is required")

    @task(3)
    def retrieve(self):
        self.client.post(
            "/retrieve",
            headers={"X-API-Key": self.api_key},
            json={
                "query": "тема урока: стили речи",
                "top_k": 8,
                "min_score": 0.2,
                "collection": "default",
                "return_text": False,
            },
            name="/retrieve",
        )

    @task(1)
    def query(self):
        self.client.post(
            "/query",
            headers={"X-API-Key": self.api_key},
            json={
                "query": "что такое стили речи",
                "top_k": 6,
                "min_score": 0.2,
                "collection": "default",
            },
            name="/query",
        )
