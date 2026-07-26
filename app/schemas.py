from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from app.config import settings


CollectionName = Annotated[str, Field(min_length=1, max_length=settings.max_collection_chars)]
SourceUri = Annotated[str, Field(min_length=1, max_length=settings.max_source_uri_chars)]


class IngestRequest(BaseModel):
    input_path: str = Field(min_length=1, max_length=4096)
    collection: CollectionName = "default"
    reindex: bool = False
    reparse: bool | None = None


class IngestResponse(BaseModel):
    indexed_docs: int
    indexed_fragments: int
    indexed_vectors: int


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=settings.max_query_chars)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)
    min_score: float = Field(default=settings.default_min_score, ge=0.0, le=1.0)
    collection: CollectionName = "default"
    source_uris: list[SourceUri] | None = Field(default=None, max_length=settings.max_source_uris)
    return_text: bool = False
    debug: bool = False
    include_toc: bool = False
    include_low_quality: bool = False
    include_navigation: bool = False


class Hit(BaseModel):
    fragment_id: str
    source_uri: str
    title: str | None = None
    type: str
    page: int | None = None
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    snippet: str
    score: float
    text: str | None = None
    dense_score: float = 0.0
    lexical_score: float = 0.0
    phrase_score: float = 0.0
    subject_score: float = 0.0
    section_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    lexical_overlap: float = 0.0
    document_score: float = 0.0
    rrf_score: float = 0.0
    exact_phrases: list[str] = Field(default_factory=list)
    matched_phrases: list[str] = Field(default_factory=list)
    missing_required_modifiers: list[str] = Field(default_factory=list)
    wrong_entity_modifier: bool = False
    phrase_score_before_penalty: float = 0.0
    phrase_score_after_penalty: float = 0.0
    is_toc: bool = False
    toc_filtered: bool = False
    toc_penalty_applied: bool = False
    quality_score: float = 1.0
    is_index: bool = False
    is_bibliography: bool = False
    is_caption: bool = False
    is_fragmented: bool = False
    is_too_short: bool = False
    starts_mid_word: bool = False
    low_text_quality: bool = False
    low_quality_filtered: bool = False
    low_text_quality_reason: str | None = None
    query_type: str = "unknown"
    chunk_type: str = "unknown"
    chunk_type_reason: str | None = None
    section_title_reason: str | None = None
    inferred_section_title: str | None = None
    required_terms: list[str] = Field(default_factory=list)
    required_term_score: float = 0.0
    required_term_match_type: str = "none"
    full_phrase_match: bool = False
    missing_required_terms: list[str] = Field(default_factory=list)
    is_navigation_index: bool = False
    navigation_filtered: bool = False
    intent_boost_applied: bool = False
    exercise_penalty_applied: bool = False
    test_question_penalty_applied: bool = False
    schema_boost_applied: bool = False
    term_penalty_applied: bool = False
    full_term_boost_applied: bool = False
    concept_boost_applied: bool = False
    exercise_demoted_for_concept_lookup: bool = False
    schema_or_rule_boost_applied: bool = False
    final_rank_reason: str | None = None
    score_before_boosts: float = 0.0
    concept_full_phrase_boost_value: float = 0.0
    section_title_boost_value: float = 0.0
    schema_or_rule_boost_value: float = 0.0
    quality_penalty_value: float = 0.0
    boundary_penalty_value: float = 0.0
    exercise_penalty_value: float = 0.0
    score_after_boosts_before_clamp: float = 0.0
    expanded_from_neighbors: bool = False
    penalties_applied: list[str] = Field(default_factory=list)


class RetrieveResponse(BaseModel):
    hits: list[Hit]
    debug: dict[str, Any] | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=settings.max_query_chars)
    top_k: int = Field(default=10, ge=1, le=settings.max_top_k)
    min_score: float = Field(default=settings.default_min_score, ge=0.0, le=1.0)
    mode: Literal["grounded", "extractive"] = "grounded"
    citation_style: Literal["fragments"] = "fragments"
    return_sources: bool = True
    collection: CollectionName = "default"
    source_uris: list[SourceUri] | None = Field(default=None, max_length=settings.max_source_uris)


class SourcesRequest(BaseModel):
    collection: CollectionName = "default"


class SourceInfo(BaseModel):
    source_uri: str
    title: str | None = None


class SourcesResponse(BaseModel):
    sources: list[SourceInfo]


class Source(BaseModel):
    n: int
    fragment_id: str
    source_uri: str
    snippet: str
    score: float
    page: int | None = None
    type: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class ParsedElement(BaseModel):
    element_index: int
    type: str
    content: str
    page: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class CanonicalFragment(BaseModel):
    fragment_id: str
    element_index: int
    source_uri: str
    type: str
    page: int | None = None
    text: str
    snippet: str
    meta: dict[str, Any] = Field(default_factory=dict)
