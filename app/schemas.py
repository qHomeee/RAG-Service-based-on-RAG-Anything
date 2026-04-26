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


class Hit(BaseModel):
    fragment_id: str
    source_uri: str
    title: str | None = None
    type: str
    page: int | None = None
    snippet: str
    score: float
    text: str | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None
    lexical_overlap: float | None = None
    document_score: float | None = None
    rrf_score: float | None = None


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
