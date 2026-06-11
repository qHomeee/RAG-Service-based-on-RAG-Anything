from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_SUBJECT_HINTS: dict[str, list[str]] = {
    "history": [
        "история",
        "правление",
        "царствование",
        "реформы",
        "реформы сперанского",
        "сперанский",
        "война",
        "отечественная война",
        "отечественная война 1812",
        "венский конгресс",
        "революция",
        "император",
        "царь",
        "князь",
        "политика",
        "государство",
        "конгресс",
        "крепостное право",
    ],
    "russian_language": [
        "русский язык",
        "морфологический разбор",
        "синтаксический разбор",
        "разбор предложения",
        "правописание",
        "орфография",
        "пунктуация",
        "причастие",
        "причастиях",
        "деепричастие",
        "имя существительное",
        "глагол",
        "придаточное предложение",
        "синтаксис",
        "морфология",
        "н и нн",
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
        "аварийно спасательные работы",
        "эвакуация",
        "защита населения",
        "средства защиты",
        "пожар",
    ],
    "biology": [
        "биология",
        "фотосинтез",
        "клетка",
        "митоз",
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
        "производная",
        "площадь треугольника",
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


def _load_subject_hints(path: str = "config/subject_hints.yaml") -> dict[str, list[str]]:
    hints = {subject: list(values) for subject, values in DEFAULT_SUBJECT_HINTS.items()}
    config_path = Path(path)
    if not config_path.exists():
        return hints

    current: str | None = None
    parsed: dict[str, list[str]] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")) and line.endswith(":"):
            current = line[:-1].strip()
            parsed.setdefault(current, [])
            continue
        if current and line.lstrip().startswith("- "):
            value = line.lstrip()[2:].strip().strip("\"'")
            if value:
                parsed[current].append(value)

    if parsed:
        hints.update({subject: values for subject, values in parsed.items() if values})
    return hints


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
    embed_offline: bool = False
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
    retrieval_section_weight: float = 0.15
    retrieval_subject_weight: float = 0.1
    retrieval_document_weight: float = 0.1
    retrieval_quality_weight: float = 0.05
    retrieval_rerank_weight: float = 0.05
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
    retrieval_low_quality_penalty: float = 0.3
    retrieval_intent_reference_boost: float = 1.15
    retrieval_intent_schema_boost: float = 1.25
    retrieval_concept_full_phrase_boost_value: float = 0.25
    retrieval_concept_section_title_boost_value: float = 0.15
    retrieval_concept_schema_rule_boost_value: float = 0.1
    retrieval_concept_missing_term_penalty_value: float = 0.3
    retrieval_concept_quality_penalty_max: float = 0.12
    retrieval_concept_boundary_penalty_value: float = 0.08
    retrieval_exercise_penalty: float = 0.55
    retrieval_test_question_penalty: float = 0.35
    retrieval_concept_exercise_penalty: float = 0.25
    retrieval_concept_test_question_penalty: float = 0.35
    retrieval_exercise_query_boost: float = 1.35
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
    subject_hints_default: dict[str, list[str]] = _load_subject_hints()
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
