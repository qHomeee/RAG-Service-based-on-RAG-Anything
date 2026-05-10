from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG Anything Service"
    app_env: str = "development"
    log_level: str = "INFO"

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
    default_min_score: float = 0.35
    chunk_size: int = 1000
    chunk_overlap: int = 150
    adaptive_chunk_min_chars: int = 800
    adaptive_chunk_max_chars: int = 1200
    semantic_chunking_enabled: bool = True
    semantic_table_chunk_max_chars: int = 700
    semantic_faq_chunk_max_chars: int = 900

    vector_recall_top_n: int = 120
    rerank_top_n: int = 40
    rag_final_top_k: int = 5
    hybrid_vector_weight: float = 0.6
    retrieval_dense_weight: float = 0.25
    retrieval_lexical_weight: float = 0.25
    retrieval_phrase_weight: float = 0.15
    retrieval_document_weight: float = 0.15
    retrieval_subject_weight: float = 0.1
    retrieval_rerank_weight: float = 0.1
    retrieval_rrf_weight: float = 0.05
    retrieval_rrf_k: int = 60
    retrieval_noise_dense_floor: float = 0.72
    retrieval_noise_strict_dense_floor: float = 0.88
    retrieval_min_lexical_overlap: float = 0.01
    retrieval_document_gate_min_score: float = 0.45
    retrieval_extreme_dense_score: float = 0.9
    retrieval_extreme_rerank_score: float = 0.92
    retrieval_adaptive_relative_floor: float = 0.55
    retrieval_adaptive_gap: float = 0.18
    retrieval_mmr_lambda: float = 0.85
    retrieval_mmr_similarity_threshold: float = 0.82
    retrieval_subject_confidence_threshold: float = 0.65
    retrieval_subject_mismatch_score: float = 0.25
    retrieval_subject_mismatch_penalty: float = 0.25
    retrieval_toc_penalty: float = 0.55
    document_routing_min_score: float = 0.18
    document_prefilter_enabled: bool = True
    document_prefilter_top_n: int = 8
    context_expansion_neighbors: int = 1
    context_expansion_max_chars: int = 2400
    query_expansion_enabled: bool = True
    query_synonyms_default: dict[str, list[str]] = {
        "инфляция": ["рост цен", "индекс потребительских цен", "обесценивание"],
        "ввп": ["валовой внутренний продукт", "gdp"],
        "стили": ["стиль", "жанр", "речь"],
        "налог": ["налогообложение", "сбор", "пошлина"],
    }
    topic_expansions_default: dict[str, list[str]] = {}
    query_synonyms_by_collection: dict[str, dict[str, list[str]]] = {}
    topic_expansions_by_collection: dict[str, dict[str, list[str]]] = {}
    query_synonyms_by_domain: dict[str, dict[str, list[str]]] = {}
    topic_expansions_by_domain: dict[str, dict[str, list[str]]] = {}
    subject_hints_default: dict[str, list[str]] = {
        "history": [
            "история",
            "правление",
            "царствование",
            "реформы",
            "война",
            "революция",
            "император",
            "царь",
            "князь",
            "политика",
            "государство",
            "конгресс",
            "крепостное право",
            "отечественная война",
        ],
        "russian_language": [
            "русский язык",
            "правописание",
            "орфография",
            "пунктуация",
            "причастие",
            "причастиях",
            "деепричастие",
            "придаточное предложение",
            "синтаксис",
            "морфология",
            "нн",
        ],
        "safety": [
            "обж",
            "безопасность",
            "гражданская оборона",
            "чрезвычайная ситуация",
            "чрезвычайных ситуациях",
            "чс",
            "асднр",
            "эвакуация",
            "защита населения",
            "средства защиты",
            "пожар",
        ],
        "biology": [
            "биология",
            "фотосинтез",
            "клетка",
            "организм",
            "растение",
            "животное",
            "экосистема",
            "хлорофилл",
            "днк",
            "ген",
        ],
        "math": [
            "математика",
            "алгебра",
            "геометрия",
            "уравнение",
            "квадратное уравнение",
            "функция",
            "дробь",
            "теорема",
            "дискриминант",
            "решить",
            "корень",
        ],
        "literature": ["литература", "роман", "повесть", "стихотворение", "поэма", "герой", "сюжет", "жанр"],
        "geography": ["география", "климат", "материк", "океан", "рельеф", "карта", "страна", "население"],
        "physics": ["физика", "сила", "масса", "скорость", "энергия", "электричество", "закон ньютона"],
        "chemistry": ["химия", "вещество", "реакция", "молекула", "атом", "кислота", "основание", "элемент"],
        "social_studies": ["обществознание", "общество", "право", "экономика", "гражданин", "социальный", "государство"],
    }
    query_expansions_by_subject: dict[str, dict[str, list[str]]] = {
        "history": {
            "правление": ["царствование", "внутренняя политика", "внешняя политика", "реформы"],
            "война": ["сражение", "кампания", "армия"],
        },
        "russian_language": {
            "правописание": ["орфография", "правило"],
            "причастиях": ["причастие", "нн", "суффикс"],
        },
        "safety": {
            "чс": ["чрезвычайная ситуация", "чрезвычайных ситуациях"],
            "асднр": ["аварийно спасательные работы", "неотложные работы"],
        },
        "math": {
            "квадратное уравнение": ["дискриминант", "корни уравнения"],
        },
        "biology": {
            "фотосинтез": ["хлорофилл", "углекислый газ", "кислород"],
        },
    }

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    parser_fallback_alert_threshold: float = 0.3

    mineru_python: str | None = None
    mineru_timeout_seconds: int | None = None

    ingest_path_must_be_under_storage_raw: bool = True
    rate_limit_per_minute: int = 120
    uvicorn_workers: int = 2


settings = Settings()
