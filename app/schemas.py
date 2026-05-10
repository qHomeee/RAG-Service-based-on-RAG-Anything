from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    input_path: str
    collection: str = "default"
    reindex: bool = False


class IngestResponse(BaseModel):
    indexed_docs: int
    indexed_fragments: int
    indexed_vectors: int


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = 12
    min_score: float = 0.35
    collection: str = "default"
    source_uris: list[str] | None = None
    return_text: bool = False
    debug: bool = False
    include_toc: bool = False


class Hit(BaseModel):
    fragment_id: str
    source_uri: str
    title: str | None = None
    type: str
    page: int | None = None
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


class RetrieveResponse(BaseModel):
    hits: list[Hit]
    debug: dict[str, Any] | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = 10
    min_score: float = 0.35
    mode: str = "grounded"
    citation_style: str = "fragments"
    return_sources: bool = True
    collection: str = "default"
    source_uris: list[str] | None = None


class SourcesRequest(BaseModel):
    collection: str = "default"


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
