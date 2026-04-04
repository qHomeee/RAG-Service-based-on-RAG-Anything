import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable
from urllib.parse import urlparse

from sqlalchemy import bindparam, delete, select, text
from sqlalchemy.orm import Session

from app.chunking import split_to_subchunks
from app.config import settings
from app.embeddings import EmbeddingProvider
from app.models import Document, Embedding, Fragment
from app.reranker import CrossEncoderReranker
from app.schemas import CanonicalFragment

logger = logging.getLogger("rag_service")


@dataclass
class RetrievalRow:
    fragment_id: str
    source_uri: str
    title: str | None
    type: str
    page: int | None
    snippet: str
    score: float
    text: str


@dataclass
class SourceRow:
    source_uri: str
    title: str | None


class RagRepository:
    def __init__(self, db: Session, embeddings: EmbeddingProvider, reranker: CrossEncoderReranker) -> None:
        self.db = db
        self.embeddings = embeddings
        self.reranker = reranker

    def upsert_document(self, source_uri: str, title: str | None, collection: str, meta: dict, reindex: bool) -> Document:
        doc = self.db.scalar(select(Document).where(Document.source_uri == source_uri))
        if doc and reindex:
            self.db.execute(delete(Embedding).where(Embedding.fragment_id.in_(select(Fragment.fragment_id).where(Fragment.doc_id == doc.doc_id))))
            self.db.execute(delete(Fragment).where(Fragment.doc_id == doc.doc_id))
            self.db.flush()
        if doc:
            doc.meta = {**(doc.meta or {}), **meta}
            self.db.flush()
            return doc
        doc = Document(source_uri=source_uri, title=title, collection=collection, meta=meta)
        self.db.add(doc)
        self.db.flush()
        return doc

    def insert_fragment_with_embeddings(self, doc: Document, fragment: CanonicalFragment) -> int:
        existing = self.db.get(Fragment, fragment.fragment_id)
        if existing:
            return 0
        row = Fragment(
            fragment_id=fragment.fragment_id,
            doc_id=doc.doc_id,
            source_uri=fragment.source_uri,
            type=fragment.type,
            page=fragment.page,
            element_index=fragment.element_index,
            text=fragment.text,
            snippet=fragment.snippet,
            meta=fragment.meta,
        )
        self.db.add(row)
        self.db.flush()

        count = 0
        for idx, subchunk in enumerate(split_to_subchunks(fragment.text)):
            emb = self.embeddings.embed(subchunk)
            self.db.add(
                Embedding(
                    fragment_id=fragment.fragment_id,
                    subchunk_index=idx,
                    text=subchunk,
                    embedding=emb,
                    meta={"offset": idx},
                )
            )
            count += 1
        return count

    def retrieve(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
    ) -> list[RetrievalRow]:
        normalized_query = normalize_query(query)
        expanded_queries = expand_query(normalized_query, collection=collection, source_uris=source_uris)

        logger.debug(
            "retrieval_query_preprocessed",
            extra={
                "original_query": query,
                "normalized_query": normalized_query,
                "expanded_queries": expanded_queries,
            },
        )

        raw_hits_count = 0

        def _wrapped_retrieve(single_query: str) -> list[RetrievalRow]:
            nonlocal raw_hits_count
            rows = self._retrieve_single_query(
                single_query,
                top_k=top_k,
                min_score=min_score,
                collection=collection,
                source_uris=source_uris,
            )
            raw_hits_count += len(rows)
            return rows

        merged_hits = retrieve_multi_query(retrieve_fn=_wrapped_retrieve, queries=expanded_queries)

        logger.debug(
            "retrieval_multi_query_counts",
            extra={
                "raw_hits_count": raw_hits_count,
                "hits_after_dedup": len(merged_hits),
            },
        )

        fused_hits = [
            RetrievalRow(
                fragment_id=row.fragment_id,
                source_uri=row.source_uri,
                title=row.title,
                type=row.type,
                page=row.page,
                snippet=row.snippet,
                score=score,
                text=row.text,
            )
            for score, row in merged_hits.values()
        ]
        if not fused_hits:
            return []

        reranked_hits = rerank_by_keyword_relevance(
            normalized_query,
            fused_hits,
            collection=collection,
            source_uris=source_uris,
        )
        logger.debug(
            "retrieval_keyword_rerank_counts",
            extra={
                "hits_after_keyword_rerank": len(reranked_hits),
                "final_selected_hits": min(top_k, len(reranked_hits)),
            },
        )
        return reranked_hits[:top_k]

    def _retrieve_single_query(
        self,
        query: str,
        *,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
    ) -> list[RetrievalRow]:
        qvec = self.embeddings.embed(query)
        recall_top_n = max(top_k, settings.vector_recall_top_n)

        source_clause = ""
        params: dict = {
            "qvec": _vector_literal(qvec),
            "collection": collection,
            "recall_top_n": recall_top_n,
        }
        if source_uris:
            source_clause = " AND d.source_uri = ANY(:source_uris)"
            params["source_uris"] = source_uris

        sql = text(
            f"""
            SELECT
                e.fragment_id,
                f.source_uri,
                d.title,
                f.type,
                f.page,
                f.snippet,
                f.text,
                MIN(e.embedding <=> CAST(:qvec AS vector)) AS distance
            FROM embeddings e
            JOIN fragments f ON f.fragment_id = e.fragment_id
            JOIN documents d ON d.doc_id = f.doc_id
            WHERE d.collection = :collection{source_clause}
            GROUP BY e.fragment_id, f.source_uri, d.title, f.type, f.page, f.snippet, f.text
            ORDER BY distance ASC
            LIMIT :recall_top_n
            """
        )
        if source_uris:
            sql = sql.bindparams(bindparam("source_uris", expanding=False))

        rows = self.db.execute(sql, params).mappings().all()
        if not rows:
            return []

        recall_candidates = [
            RetrievalRow(
                fragment_id=row["fragment_id"],
                source_uri=row["source_uri"],
                title=row["title"],
                type=row["type"],
                page=row["page"],
                snippet=row["snippet"],
                score=max(0.0, 1.0 - float(row["distance"])),
                text=row["text"],
            )
            for row in rows
        ]

        query_terms = _tokenize(query)
        if query_terms:
            bm25_scores = _bm25_scores(query_terms, recall_candidates)
        else:
            bm25_scores = {c.fragment_id: 0.0 for c in recall_candidates}

        hybrid_candidates: list[tuple[RetrievalRow, float]] = []
        for row in recall_candidates:
            keyword_score = bm25_scores.get(row.fragment_id, 0.0)
            hybrid = settings.hybrid_vector_weight * row.score + (1 - settings.hybrid_vector_weight) * keyword_score
            if row.score >= min_score or keyword_score > 0:
                hybrid_candidates.append((row, hybrid))

        if not hybrid_candidates:
            return []

        hybrid_candidates.sort(key=lambda x: x[1], reverse=True)
        rerank_candidates = hybrid_candidates[: max(top_k, settings.rerank_top_n)]

        if self.reranker.available:
            rerank_scores = self.reranker.score(query, [item[0].text for item in rerank_candidates])
        else:
            logger.warning("reranker_unavailable_fallback", extra={"reranker_model": self.reranker.model_name, "error": self.reranker.load_error})
            logger.error(
                "reranker_unavailable_fallback_alert",
                extra={
                    "alert": True,
                    "recommendation": "Monitor /readyz checks.reranker_loaded and reranker_error",
                    "reranker_model": self.reranker.model_name,
                },
            )
            rerank_scores = [score for _, score in rerank_candidates]

        rescored: list[RetrievalRow] = []
        for (row, _), rerank_score in zip(rerank_candidates, rerank_scores):
            rescored.append(
                RetrievalRow(
                    fragment_id=row.fragment_id,
                    source_uri=row.source_uri,
                    title=row.title,
                    type=row.type,
                    page=row.page,
                    snippet=row.snippet,
                    score=float(rerank_score),
                    text=row.text,
                )
            )

        return sorted(rescored, key=lambda r: r.score, reverse=True)[:top_k]

    def list_sources(self, collection: str) -> list[SourceRow]:
        rows = self.db.scalars(
            select(Document).where(Document.collection == collection).order_by(Document.source_uri.asc())
        ).all()
        return [SourceRow(source_uri=row.source_uri, title=row.title) for row in rows]



def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in values) + "]"

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w-]+", (text or "").lower())


QUERY_PREFIXES = (
    "тема урока:",
    "тема:",
    "урок:",
)


def normalize_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", (query or "").strip().lower())
    for prefix in QUERY_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    return normalized


def expand_query(query: str, *, collection: str | None = None, source_uris: list[str] | None = None) -> list[str]:
    if not query:
        return []

    queries = [query]
    if not settings.query_expansion_enabled:
        return queries

    terms = _tokenize(query)
    if len(terms) > 4:
        return queries

    query_synonyms, topic_expansions = _expansion_maps(collection=collection, source_uris=source_uris)

    extra_terms: list[str] = []
    for term in terms:
        extra_terms.extend(query_synonyms.get(term, []))

    for key, topic_terms in topic_expansions.items():
        if key in query:
            extra_terms.extend(topic_terms)

    if extra_terms:
        queries.append(f"{query} {' '.join(extra_terms)}")

    if len(terms) <= 3 and any(re.search(r"\d", item) for item in extra_terms):
        temporal_terms = [item for item in extra_terms if re.search(r"\d", item)]
        queries.append(f"{query} {' '.join(temporal_terms)}")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in queries:
        normalized_item = re.sub(r"\s+", " ", item).strip()
        key = normalized_item.lower()
        if not normalized_item or key in seen:
            continue
        deduped.append(normalized_item)
        seen.add(key)
    return deduped


def retrieve_multi_query(
    *,
    retrieve_fn: Callable[[str], list[RetrievalRow]],
    queries: Iterable[str],
) -> dict[str, tuple[float, RetrievalRow]]:
    merged: dict[str, tuple[float, RetrievalRow]] = {}
    for idx, q in enumerate(queries):
        rows = retrieve_fn(q)
        weight = 1.0 if idx == 0 else 0.9
        for row in rows:
            weighted_score = float(row.score) * weight
            current = merged.get(row.fragment_id)
            if current is None or weighted_score > current[0]:
                merged[row.fragment_id] = (weighted_score, row)
    return merged


def rerank_by_keyword_relevance(
    query: str,
    hits: list[RetrievalRow],
    *,
    collection: str | None = None,
    source_uris: list[str] | None = None,
) -> list[RetrievalRow]:
    markers = _topic_markers(query, collection=collection, source_uris=source_uris)
    if not markers:
        return sorted(hits, key=lambda h: h.score, reverse=True)

    rescored: list[RetrievalRow] = []
    for hit in hits:
        haystack = (hit.text or "").lower()
        matches = sum(1 for marker in markers if marker in haystack)
        adjusted_score = hit.score
        if matches == 0:
            adjusted_score *= 0.55
        elif matches == 1:
            adjusted_score += 0.15
        else:
            adjusted_score += 0.35 + min(0.25, (matches - 2) * 0.05)

        rescored.append(
            RetrievalRow(
                fragment_id=hit.fragment_id,
                source_uri=hit.source_uri,
                title=hit.title,
                type=hit.type,
                page=hit.page,
                snippet=hit.snippet,
                score=adjusted_score,
                text=hit.text,
            )
        )

    return sorted(rescored, key=lambda h: h.score, reverse=True)


def _topic_markers(query: str, *, collection: str | None = None, source_uris: list[str] | None = None) -> list[str]:
    markers: set[str] = set()
    query_lc = query.lower()
    _, topic_expansions = _expansion_maps(collection=collection, source_uris=source_uris)
    for token in _tokenize(query_lc):
        if len(token) >= 4:
            markers.add(token[:5])
        if len(token) >= 6:
            markers.add(token)

    for topic, topic_markers in topic_expansions.items():
        if topic in query_lc:
            markers.update(topic_markers)

    return sorted(m for m in markers if len(m) >= 4)


def _expand_query(query: str) -> str:
    expanded = expand_query(normalize_query(query))
    if not expanded:
        return query
    return expanded[1] if len(expanded) > 1 else expanded[0]


def _expansion_maps(
    *,
    collection: str | None,
    source_uris: list[str] | None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    query_synonyms = _clone_map(settings.query_synonyms_default)
    topic_expansions = _clone_map(settings.topic_expansions_default)

    if collection:
        _merge_map(query_synonyms, settings.query_synonyms_by_collection.get(collection, {}))
        _merge_map(topic_expansions, settings.topic_expansions_by_collection.get(collection, {}))

    for domain in _extract_domains(source_uris):
        _merge_map(query_synonyms, settings.query_synonyms_by_domain.get(domain, {}))
        _merge_map(topic_expansions, settings.topic_expansions_by_domain.get(domain, {}))

    return query_synonyms, topic_expansions


def _clone_map(source: dict[str, list[str]]) -> dict[str, list[str]]:
    return {key.lower(): [item.strip() for item in values if item and item.strip()] for key, values in source.items()}


def _merge_map(target: dict[str, list[str]], incoming: dict[str, list[str]]) -> None:
    for key, values in incoming.items():
        normalized_key = (key or "").strip().lower()
        if not normalized_key:
            continue
        existing = target.setdefault(normalized_key, [])
        seen = {item.lower() for item in existing}
        for raw_value in values:
            value = (raw_value or "").strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            existing.append(value)
            seen.add(lowered)


def _extract_domains(source_uris: list[str] | None) -> set[str]:
    if not source_uris:
        return set()
    domains: set[str] = set()
    for source_uri in source_uris:
        uri = (source_uri or "").strip().lower()
        if not uri:
            continue
        parsed = urlparse(uri if "://" in uri else f"https://{uri}")
        host = parsed.netloc or parsed.path.split("/")[0]
        host = host.split("@")[-1].split(":")[0].strip(".")
        if host:
            domains.add(host)
    return domains


def _bm25_scores(query_terms: list[str], rows: list[RetrievalRow], *, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    tokenized = {row.fragment_id: _tokenize(row.text) for row in rows}
    doc_count = len(rows)
    avg_len = sum(len(tokens) for tokens in tokenized.values()) / doc_count if doc_count else 0.0

    df: Counter[str] = Counter()
    for tokens in tokenized.values():
        for term in set(tokens):
            df[term] += 1

    scores: dict[str, float] = {}
    for row in rows:
        tokens = tokenized[row.fragment_id]
        tf = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (doc_count - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * doc_len / avg_len) if avg_len else 1.0
            score += idf * ((tf[term] * (k1 + 1)) / denom)
        scores[row.fragment_id] = score

    max_score = max(scores.values()) if scores else 0.0
    if max_score <= 0:
        return {row.fragment_id: 0.0 for row in rows}
    return {key: value / max_score for key, value in scores.items()}
