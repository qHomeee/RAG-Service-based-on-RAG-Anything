from locust import HttpUser, between, task


class RAGUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(3)
    def retrieve(self):
        self.client.post(
            "/retrieve",
            headers={"X-API-Key": "change-me"},
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
            headers={"X-API-Key": "change-me"},
            json={
                "query": "что такое стили речи",
                "top_k": 6,
                "min_score": 0.2,
                "collection": "default",
            },
            name="/query",
        )
