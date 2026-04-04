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
    storage_parsed: str = "storage/parsed"
    redis_url: str | None = None

    max_file_size_mb: int = 50
    max_query_chars: int = 4000
    default_top_k: int = 12
    default_min_score: float = 0.2
    chunk_size: int = 1500
    chunk_overlap: int = 180
    adaptive_chunk_min_chars: int = 800
    adaptive_chunk_max_chars: int = 1200
    semantic_chunking_enabled: bool = True
    semantic_table_chunk_max_chars: int = 700
    semantic_faq_chunk_max_chars: int = 900

    vector_recall_top_n: int = 120
    rerank_top_n: int = 40
    rag_final_top_k: int = 5
    hybrid_vector_weight: float = 0.6
    query_expansion_enabled: bool = True
    query_synonyms_default: dict[str, list[str]] = {
        "инфляция": ["рост цен", "индекс потребительских цен", "обесценивание"],
        "ввп": ["валовой внутренний продукт", "gdp"],
        "стили": ["стиль", "жанр", "речь"],
        "налог": ["налогообложение", "сбор", "пошлина"],
    }
    topic_expansions_default: dict[str, list[str]] = {
        "османская империя": ["осман", "турция", "султан", "танзим", "стамбул", "порта", "19 век", "упадок"],
        "римская империя": ["рим", "цезарь", "сенат", "легион", "провинция", "античность"],
        "первая мировая": ["1914", "антанта", "центральные державы", "окопная война", "верден"],
    }
    query_synonyms_by_collection: dict[str, dict[str, list[str]]] = {}
    topic_expansions_by_collection: dict[str, dict[str, list[str]]] = {}
    query_synonyms_by_domain: dict[str, dict[str, list[str]]] = {}
    topic_expansions_by_domain: dict[str, dict[str, list[str]]] = {}

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    parser_fallback_alert_threshold: float = 0.3

    mineru_python: str | None = None
    mineru_timeout_seconds: int | None = None

    ingest_path_must_be_under_storage_raw: bool = True
    rate_limit_per_minute: int = 120
    uvicorn_workers: int = 2


settings = Settings()
