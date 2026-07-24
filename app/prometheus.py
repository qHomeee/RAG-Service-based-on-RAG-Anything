import os

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, multiprocess


REQUESTS = Counter(
    "rag_http_requests_total",
    "HTTP requests handled by the RAG service.",
    ("method", "path", "status"),
)
LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ("method", "path"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
RETRIEVAL_RESULTS = Histogram(
    "rag_retrieval_results",
    "Number of fragments returned by retrieval.",
    buckets=(0, 1, 2, 3, 5, 8, 12, 20),
)


def record_http_request(method: str, path: str, status_code: int, latency_seconds: float) -> None:
    REQUESTS.labels(method=method, path=path, status=str(status_code)).inc()
    LATENCY.labels(method=method, path=path).observe(latency_seconds)


def render_metrics() -> bytes:
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest()
