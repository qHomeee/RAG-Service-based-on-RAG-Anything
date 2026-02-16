from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Anything Service"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
    embed_dim: int = 384
    embed_model: str = "all-MiniLM-L6-v2"
    fail_on_embedding_fallback: bool = True
    api_key: str = "change-me"
    storage_raw: str = "storage/raw"
    redis_url: str | None = None

    max_file_size_mb: int = 50
    max_query_chars: int = 4000
    default_top_k: int = 12
    default_min_score: float = 0.2
    chunk_size: int = 1500
    chunk_overlap: int = 180


settings = Settings()
