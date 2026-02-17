from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Anything Service"
    app_env: str = "development"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_recycle_seconds: int = 1800

    embed_dim: int = 384
    embed_model: str = "all-MiniLM-L6-v2"
    fail_on_embedding_fallback: bool = True
    api_key: str = "change-me"
    admin_api_key: str = "change-me-admin"
    storage_raw: str = "storage/raw"
    redis_url: str | None = None

    max_file_size_mb: int = 50
    max_query_chars: int = 4000
    default_top_k: int = 12
    default_min_score: float = 0.2
    chunk_size: int = 1500
    chunk_overlap: int = 180
    adaptive_chunk_min_chars: int = 800
    adaptive_chunk_max_chars: int = 1200

    vector_recall_top_n: int = 50
    rerank_top_n: int = 10
    hybrid_vector_weight: float = 0.7

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    parser_fallback_alert_threshold: float = 0.3

    ingest_path_must_be_under_storage_raw: bool = True
    rate_limit_per_minute: int = 120
    uvicorn_workers: int = 2


settings = Settings()
