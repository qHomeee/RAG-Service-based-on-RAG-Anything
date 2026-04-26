import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable
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
    doc_id: str | None = None
    element_index: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    dense_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0
    lexical_overlap: float = 0.0
    document_score: float = 0.0
    rrf_score: float = 0.0
    rejection_reason: str | None = None


@dataclass
class RetrievalResult:
    hits: list[RetrievalRow]
    debug: dict[str, Any] | None = None


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
        return self.retrieve_with_debug(query, top_k, min_score, collection, source_uris, debug=False).hits

    def retrieve_with_debug(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
        *,
        debug: bool = False,
    ) -> RetrievalResult:
        normalized_query = normalize_query(query)
        query_terms = _query_terms_for_scoring(normalized_query)
        expanded_queries = expand_query(normalized_query, collection=collection, source_uris=source_uris)
        min_score = _clamp_score(min_score)
        recall_top_n = max(top_k, settings.vector_recall_top_n)

        diagnostics: dict[str, Any] = {
            "query": normalized_query,
            "top_k": top_k,
            "min_score": min_score,
            "collection": collection,
            "expanded_queries": expanded_queries,
            "dense_candidates": 0,
            "lexical_candidates": 0,
            "document_candidates": 0,
            "candidates_after_fusion": 0,
            "candidates_after_noise_filter": 0,
            "candidates_after_threshold": 0,
            "candidates_after_adaptive_threshold": 0,
            "candidates_after_mmr": 0,
            "rejected_results": [],
            "source_uri_first_results": [],
        }

        candidate_groups: list[list[RetrievalRow]] = []
        doc_scores: dict[str, float] = {}
        for idx, single_query in enumerate(expanded_queries):
            single_terms = _query_terms_for_scoring(single_query)
            doc_filter_uris, single_doc_scores = self._document_prefilter(
                single_query,
                single_terms,
                collection=collection,
                source_uris=source_uris,
                limit=settings.document_prefilter_top_n,
            )
            for source_uri, score in single_doc_scores.items():
                doc_scores[source_uri] = max(doc_scores.get(source_uri, 0.0), score)

            dense_candidates = self._dense_recall(
                single_query,
                collection=collection,
                source_uris=doc_filter_uris,
                limit=recall_top_n,
            )
            lexical_candidates = self._keyword_recall(
                single_terms,
                collection=collection,
                source_uris=doc_filter_uris,
                limit=recall_top_n,
            )
            fused = reciprocal_rank_fusion(dense_candidates, lexical_candidates, document_scores=single_doc_scores)
            if idx > 0:
                fused = [replace(row, rrf_score=row.rrf_score * 0.9) for row in fused]
            candidate_groups.append(fused)
            diagnostics["dense_candidates"] += len(dense_candidates)
            diagnostics["lexical_candidates"] += len(lexical_candidates)
            diagnostics["document_candidates"] += len(single_doc_scores)

        fused_candidates = _merge_candidates(*candidate_groups)
        for row in fused_candidates:
            row.document_score = max(row.document_score, doc_scores.get(row.source_uri, 0.0))
        diagnostics["candidates_after_fusion"] = len(fused_candidates)

        if not fused_candidates:
            logger.debug("retrieval_empty_after_fusion", extra=diagnostics)
            return RetrievalResult(hits=[], debug=diagnostics if debug else None)

        preliminary_scored, rejected = score_retrieval_candidates(
            normalized_query,
            fused_candidates,
            apply_noise_filter=True,
        )
        diagnostics["rejected_results"].extend(rejected)
        diagnostics["candidates_after_noise_filter"] = len(preliminary_scored)
        if not preliminary_scored:
            logger.debug("retrieval_empty_after_noise_filter", extra=diagnostics)
            return RetrievalResult(hits=[], debug=diagnostics if debug else None)

        rerank_limit = max(top_k, settings.rerank_top_n)
        rerank_candidates = preliminary_scored[:rerank_limit]

        if self.reranker.available:
            raw_rerank_scores = self.reranker.score(normalized_query, [item.text for item in rerank_candidates])
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
            raw_rerank_scores = None

        scored_candidates, rerank_rejected = score_retrieval_candidates(
            normalized_query,
            rerank_candidates,
            rerank_raw_scores=raw_rerank_scores,
            apply_noise_filter=True,
        )
        diagnostics["rejected_results"].extend(rerank_rejected)

        thresholded = [hit for hit in scored_candidates if hit.final_score >= min_score]
        diagnostics["candidates_after_threshold"] = len(thresholded)
        adaptive_hits = apply_adaptive_threshold(thresholded)
        diagnostics["candidates_after_adaptive_threshold"] = len(adaptive_hits)
        final_hits = mmr_select(adaptive_hits, top_k=top_k, query_terms=query_terms)
        diagnostics["candidates_after_mmr"] = len(final_hits)
        final_hits = self._expand_context(final_hits)

        diagnostics["source_uri_first_results"] = [hit.source_uri for hit in final_hits[:5]]
        diagnostics["top_scores"] = _top_scores(final_hits)
        diagnostics["hits"] = [_debug_hit(hit) for hit in final_hits[:top_k]]

        logger.debug(
            "retrieval_diagnostics",
            extra={**diagnostics, "rejected_results": diagnostics["rejected_results"][:20]},
        )
        return RetrievalResult(hits=final_hits, debug=diagnostics if debug else None)

    def _document_prefilter(
        self,
        query: str,
        query_terms: list[str],
        *,
        collection: str,
        source_uris: list[str] | None,
        limit: int,
    ) -> tuple[list[str] | None, dict[str, float]]:
        if source_uris or not settings.document_prefilter_enabled:
            return source_uris, {uri: 1.0 for uri in source_uris or []}

        dense_docs = self._dense_document_recall(query, collection=collection, limit=limit)
        lexical_docs = self._keyword_document_recall(query_terms, collection=collection, limit=limit)
        doc_scores = reciprocal_rank_scores(dense_docs, lexical_docs)
        if not doc_scores:
            return None, {}
        return list(doc_scores.keys())[:limit], doc_scores

    def _dense_document_recall(self, query: str, *, collection: str, limit: int) -> list[str]:
        qvec = self.embeddings.embed(query)
        sql = text(
            """
            SELECT d.source_uri, MIN(e.embedding <=> CAST(:qvec AS vector)) AS distance
            FROM documents d
            JOIN fragments f ON f.doc_id = d.doc_id
            JOIN embeddings e ON e.fragment_id = f.fragment_id
            WHERE d.collection = :collection
            GROUP BY d.source_uri
            ORDER BY distance ASC
            LIMIT :limit
            """
        )
        rows = self.db.execute(
            sql,
            {"qvec": _vector_literal(qvec), "collection": collection, "limit": limit},
        ).mappings().all()
        return [row["source_uri"] for row in rows]

    def _keyword_document_recall(self, query_terms: list[str], *, collection: str, limit: int) -> list[str]:
        terms = _keyword_recall_terms(query_terms)
        if not terms:
            return []

        params: dict[str, Any] = {"collection": collection, "limit": limit}
        conditions: list[str] = []
        rank_parts: list[str] = []
        for idx, term_value in enumerate(terms):
            param_name = f"doc_kw{idx}"
            params[param_name] = f"%{term_value}%"
            conditions.append(
                f"(f.text ILIKE :{param_name} OR d.title ILIKE :{param_name} OR d.source_uri ILIKE :{param_name})"
            )
            rank_parts.append(f"(CASE WHEN f.text ILIKE :{param_name} THEN 1 ELSE 0 END)")

        sql = text(
            f"""
            SELECT d.source_uri
            FROM documents d
            JOIN fragments f ON f.doc_id = d.doc_id
            WHERE d.collection = :collection
              AND ({" OR ".join(conditions)})
            GROUP BY d.source_uri
            ORDER BY SUM({" + ".join(rank_parts)}) DESC, d.source_uri ASC
            LIMIT :limit
            """
        )
        rows = self.db.execute(sql, params).mappings().all()
        return [row["source_uri"] for row in rows]

    def _expand_context(self, hits: list[RetrievalRow]) -> list[RetrievalRow]:
        neighbors = max(0, settings.context_expansion_neighbors)
        if neighbors <= 0:
            return hits

        expanded: list[RetrievalRow] = []
        for hit in hits:
            if not hit.doc_id or hit.element_index is None:
                expanded.append(hit)
                continue

            rows = self.db.execute(
                text(
                    """
                    SELECT text
                    FROM fragments
                    WHERE doc_id = CAST(:doc_id AS uuid)
                      AND element_index BETWEEN :start_idx AND :end_idx
                    ORDER BY element_index ASC
                    """
                ),
                {
                    "doc_id": hit.doc_id,
                    "start_idx": hit.element_index - neighbors,
                    "end_idx": hit.element_index + neighbors,
                },
            ).mappings().all()
            context = normalize_context(" ".join(row["text"] for row in rows), settings.context_expansion_max_chars)
            expanded.append(replace(hit, text=context or hit.text))
        return expanded

    def _dense_recall(
        self,
        query: str,
        *,
        collection: str,
        source_uris: list[str] | None,
        limit: int,
    ) -> list[RetrievalRow]:
        qvec = self.embeddings.embed(query)

        source_clause = ""
        params: dict = {
            "qvec": _vector_literal(qvec),
            "collection": collection,
            "recall_top_n": limit,
        }
        if source_uris:
            source_clause = " AND d.source_uri = ANY(:source_uris)"
            params["source_uris"] = source_uris

        sql = text(
            f"""
            SELECT
                e.fragment_id,
                f.doc_id,
                f.source_uri,
                d.title,
                f.type,
                f.page,
                f.element_index,
                f.snippet,
                f.text,
                MIN(e.embedding <=> CAST(:qvec AS vector)) AS distance
            FROM embeddings e
            JOIN fragments f ON f.fragment_id = e.fragment_id
            JOIN documents d ON d.doc_id = f.doc_id
            WHERE d.collection = :collection{source_clause}
            GROUP BY e.fragment_id, f.doc_id, f.source_uri, d.title, f.type, f.page, f.element_index, f.snippet, f.text
            ORDER BY distance ASC
            LIMIT :recall_top_n
            """
        )
        if source_uris:
            sql = sql.bindparams(bindparam("source_uris", expanding=False))

        rows = self.db.execute(sql, params).mappings().all()
        if not rows:
            return []

        return [
            RetrievalRow(
                fragment_id=row["fragment_id"],
                doc_id=str(row["doc_id"]) if row["doc_id"] is not None else None,
                source_uri=row["source_uri"],
                title=row["title"],
                type=row["type"],
                page=row["page"],
                element_index=row["element_index"],
                snippet=row["snippet"],
                score=_dense_similarity(row["distance"]),
                text=row["text"],
                dense_score=_dense_similarity(row["distance"]),
            )
            for row in rows
        ]

    def _keyword_recall(
        self,
        query_terms: list[str],
        *,
        collection: str,
        source_uris: list[str] | None,
        limit: int,
    ) -> list[RetrievalRow]:
        terms = _keyword_recall_terms(query_terms)
        if not terms:
            return []

        params: dict = {"collection": collection, "recall_top_n": limit}
        conditions: list[str] = []
        rank_parts: list[str] = []
        for idx, term_value in enumerate(terms):
            param_name = f"kw{idx}"
            params[param_name] = f"%{term_value}%"
            conditions.append(
                f"(f.text ILIKE :{param_name} OR f.snippet ILIKE :{param_name} "
                f"OR d.title ILIKE :{param_name} OR f.source_uri ILIKE :{param_name})"
            )
            rank_parts.append(f"(CASE WHEN f.text ILIKE :{param_name} THEN 1 ELSE 0 END)")

        source_clause = ""
        if source_uris:
            source_clause = " AND d.source_uri = ANY(:source_uris)"
            params["source_uris"] = source_uris

        sql = text(
            f"""
            SELECT
                f.fragment_id,
                f.doc_id,
                f.source_uri,
                d.title,
                f.type,
                f.page,
                f.element_index,
                f.snippet,
                f.text,
                f.meta
            FROM fragments f
            JOIN documents d ON d.doc_id = f.doc_id
            WHERE d.collection = :collection{source_clause}
              AND ({" OR ".join(conditions)})
            ORDER BY ({" + ".join(rank_parts)}) DESC, f.fragment_id ASC
            LIMIT :recall_top_n
            """
        )
        if source_uris:
            sql = sql.bindparams(bindparam("source_uris", expanding=False))

        rows = self.db.execute(sql, params).mappings().all()
        return [
            RetrievalRow(
                fragment_id=row["fragment_id"],
                doc_id=str(row["doc_id"]) if row["doc_id"] is not None else None,
                source_uri=row["source_uri"],
                title=row["title"],
                type=row["type"],
                page=row["page"],
                element_index=row["element_index"],
                snippet=row["snippet"],
                score=0.0,
                text=row["text"],
                meta=row["meta"] or {},
            )
            for row in rows
        ]

    def list_sources(self, collection: str) -> list[SourceRow]:
        rows = self.db.scalars(
            select(Document).where(Document.collection == collection).order_by(Document.source_uri.asc())
        ).all()
        return [SourceRow(source_uri=row.source_uri, title=row.title) for row in rows]



def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in values) + "]"

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w-]+", (text or "").lower().replace("ё", "е"))


QUERY_STOPWORDS = {
    "а",
    "без",
    "в",
    "во",
    "для",
    "до",
    "его",
    "ее",
    "её",
    "и",
    "или",
    "их",
    "к",
    "на",
    "над",
    "о",
    "об",
    "от",
    "по",
    "под",
    "при",
    "про",
    "с",
    "со",
    "у",
    "and",
    "of",
    "the",
}


def _query_terms_for_scoring(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _tokenize(text):
        if token in QUERY_STOPWORDS:
            continue
        if len(token) < 2 and not any(char.isdigit() for char in token) and not _is_roman_numeral(token):
            continue
        if token in seen:
            continue
        terms.append(token)
        seen.add(token)
    return terms


def _keyword_recall_terms(query_terms: list[str], *, limit: int = 8) -> list[str]:
    terms: list[str] = []
    for term in query_terms:
        if len(term) >= 3 or any(char.isdigit() for char in term):
            terms.append(term)
        elif len(term) >= 2 and not _is_roman_numeral(term):
            terms.append(term)
    return terms[:limit]


def _merge_candidates(*candidate_groups: list[RetrievalRow]) -> list[RetrievalRow]:
    merged: dict[str, RetrievalRow] = {}
    for candidates in candidate_groups:
        for row in candidates:
            current = merged.get(row.fragment_id)
            if current is None:
                merged[row.fragment_id] = row
                continue
            merged[row.fragment_id] = replace(
                current,
                dense_score=max(current.dense_score, row.dense_score),
                lexical_score=max(current.lexical_score, row.lexical_score),
                rrf_score=max(current.rrf_score, row.rrf_score),
                document_score=max(current.document_score, row.document_score),
                score=max(current.score, row.score),
            )
    return list(merged.values())


def _top_scores(rows: list[RetrievalRow], *, limit: int = 5) -> list[float]:
    return _top_values([row.score for row in rows], limit=limit)


def _top_tuple_scores(rows: list[tuple[RetrievalRow, float]], *, limit: int = 5) -> list[float]:
    return _top_values([score for _, score in rows], limit=limit)


def _top_values(values: Iterable[float], *, limit: int = 5) -> list[float]:
    return [round(float(score), 4) for score in list(values)[:limit]]


def _clamp_score(value: float | int | None) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _dense_similarity(distance: Any) -> float:
    try:
        return _clamp_score(1.0 - float(distance))
    except (TypeError, ValueError):
        return 0.0


def _is_roman_numeral(token: str) -> bool:
    return bool(re.fullmatch(r"[ivxlcdm]+", token or ""))


def _lexical_forms(token: str) -> set[str]:
    token = (token or "").lower().replace("ё", "е")
    if not token:
        return set()

    forms = {token}
    if any("а" <= char <= "я" for char in token) and len(token) >= 5:
        for suffix in (
            "иями",
            "ями",
            "ами",
            "ого",
            "ему",
            "ыми",
            "ими",
            "ая",
            "яя",
            "ое",
            "ее",
            "ые",
            "ие",
            "ый",
            "ий",
            "ой",
            "ах",
            "ях",
            "ам",
            "ям",
            "ом",
            "ем",
            "ов",
            "ев",
            "ия",
            "ие",
            "ии",
            "ей",
            "ой",
            "ы",
            "и",
            "а",
            "я",
            "е",
            "у",
            "ю",
        ):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                forms.add(token[: -len(suffix)])
                break
    return forms


def _matching_tokens(text: str) -> list[str]:
    forms: list[str] = []
    for token in _tokenize(text):
        forms.extend(_lexical_forms(token))
    return forms


def lexical_overlap(query_terms: list[str], text_value: str) -> float:
    if not query_terms:
        return 0.0
    haystack_forms = set(_matching_tokens(text_value))
    if not haystack_forms:
        return 0.0
    matched = 0
    for term in query_terms:
        if _lexical_forms(term) & haystack_forms:
            matched += 1
    return matched / len(query_terms)


def reciprocal_rank_scores(*ranked_lists: list[str], k: int | None = None) -> dict[str, float]:
    rrf_k = k or settings.retrieval_rrf_k
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    max_score = max(scores.values()) if scores else 0.0
    if max_score <= 0:
        return {}
    return dict(sorted(((key, value / max_score) for key, value in scores.items()), key=lambda item: item[1], reverse=True))


def reciprocal_rank_fusion(
    dense_candidates: list[RetrievalRow],
    lexical_candidates: list[RetrievalRow],
    *,
    document_scores: dict[str, float] | None = None,
) -> list[RetrievalRow]:
    dense_rank = [row.fragment_id for row in dense_candidates]
    lexical_rank = [row.fragment_id for row in lexical_candidates]
    rrf_scores = reciprocal_rank_scores(dense_rank, lexical_rank)
    merged = _merge_candidates(dense_candidates, lexical_candidates)
    document_scores = document_scores or {}
    fused = [
        replace(
            row,
            rrf_score=rrf_scores.get(row.fragment_id, 0.0),
            document_score=document_scores.get(row.source_uri, row.document_score),
        )
        for row in merged
    ]
    return sorted(fused, key=lambda row: (row.rrf_score, row.dense_score), reverse=True)


def _normalize_rerank_scores(raw_scores: list[float] | None, count: int) -> list[float | None]:
    if not raw_scores:
        return [None] * count
    sigmoid_scores = [1.0 / (1.0 + math.exp(-float(score))) for score in raw_scores]
    if len(sigmoid_scores) < count:
        sigmoid_scores.extend([None] * (count - len(sigmoid_scores)))  # type: ignore[arg-type]
    return sigmoid_scores[:count]


def _final_score(
    *,
    dense_score: float,
    lexical_score: float,
    lexical_overlap_score: float,
    rerank_score: float | None,
    rrf_score: float,
    document_score: float,
) -> float:
    lexical_component = max(lexical_score, lexical_overlap_score)
    components: list[tuple[float, float]] = [
        (settings.retrieval_dense_weight, dense_score),
        (settings.retrieval_lexical_weight, lexical_component),
        (settings.retrieval_rrf_weight, rrf_score),
        (settings.retrieval_document_weight, document_score),
    ]
    if rerank_score is not None:
        components.append((settings.retrieval_rerank_weight, rerank_score))
    total_weight = sum(weight for weight, _ in components if weight > 0)
    if total_weight <= 0:
        return 0.0
    return _clamp_score(sum(weight * score for weight, score in components) / total_weight)


def score_retrieval_candidates(
    query: str,
    candidates: list[RetrievalRow],
    *,
    rerank_raw_scores: list[float] | None = None,
    apply_noise_filter: bool = True,
) -> tuple[list[RetrievalRow], list[dict[str, Any]]]:
    query_terms = _query_terms_for_scoring(query)
    lexical_scores = _bm25_scores(query_terms, candidates) if query_terms else {}
    rerank_scores = _normalize_rerank_scores(rerank_raw_scores, len(candidates))
    scored: list[RetrievalRow] = []
    rejected: list[dict[str, Any]] = []

    for idx, row in enumerate(candidates):
        dense_score = _clamp_score(row.dense_score or row.score)
        lexical_score = _clamp_score(max(row.lexical_score, lexical_scores.get(row.fragment_id, 0.0)))
        overlap = lexical_overlap(query_terms, f"{row.title or ''} {row.source_uri or ''} {row.text or ''}")
        rerank_score = rerank_scores[idx]
        final_score = _final_score(
            dense_score=dense_score,
            lexical_score=lexical_score,
            lexical_overlap_score=overlap,
            rerank_score=rerank_score,
            rrf_score=_clamp_score(row.rrf_score),
            document_score=_clamp_score(row.document_score),
        )
        reason = _noise_rejection_reason(
            query_terms=query_terms,
            dense_score=dense_score,
            lexical_score=lexical_score,
            lexical_overlap_score=overlap,
            final_score=final_score,
        )
        updated = replace(
            row,
            score=final_score,
            dense_score=dense_score,
            lexical_score=lexical_score,
            lexical_overlap=overlap,
            rerank_score=rerank_score,
            final_score=final_score,
            rejection_reason=reason,
        )
        if apply_noise_filter and reason:
            rejected.append(_debug_hit(updated))
            continue
        scored.append(updated)

    return sorted(scored, key=lambda row: row.final_score, reverse=True), rejected


def _noise_rejection_reason(
    *,
    query_terms: list[str],
    dense_score: float,
    lexical_score: float,
    lexical_overlap_score: float,
    final_score: float,
) -> str | None:
    if not query_terms:
        return None
    if (
        lexical_overlap_score <= settings.retrieval_min_lexical_overlap
        and lexical_score <= 0.02
        and dense_score < settings.retrieval_noise_dense_floor
    ):
        return "low_dense_no_lexical_overlap"
    if lexical_overlap_score == 0 and lexical_score == 0 and dense_score < settings.retrieval_noise_strict_dense_floor:
        return "no_query_terms_in_chunk"
    if final_score <= 0:
        return "zero_final_score"
    return None


def apply_adaptive_threshold(hits: list[RetrievalRow]) -> list[RetrievalRow]:
    ordered = sorted(hits, key=lambda row: row.final_score, reverse=True)
    if len(ordered) <= 2:
        return ordered

    top_score = ordered[0].final_score
    kept = [ordered[0]]
    for prev, current in zip(ordered, ordered[1:]):
        if top_score > 0 and current.final_score < top_score * settings.retrieval_adaptive_relative_floor:
            break
        if (
            prev.final_score - current.final_score >= settings.retrieval_adaptive_gap
            and current.final_score < top_score * 0.75
        ):
            break
        kept.append(current)
    return kept


def mmr_select(hits: list[RetrievalRow], *, top_k: int, query_terms: list[str]) -> list[RetrievalRow]:
    candidates = sorted(hits, key=lambda row: row.final_score, reverse=True)
    selected: list[RetrievalRow] = []
    while candidates and len(selected) < top_k:
        if not selected:
            selected.append(candidates.pop(0))
            continue

        best_idx = 0
        best_score = -1.0
        for idx, candidate in enumerate(candidates):
            max_similarity = max(_text_similarity(candidate.text, chosen.text) for chosen in selected)
            if max_similarity >= settings.retrieval_mmr_similarity_threshold:
                adjusted = candidate.final_score * 0.5
            else:
                adjusted = (
                    settings.retrieval_mmr_lambda * candidate.final_score
                    - (1 - settings.retrieval_mmr_lambda) * max_similarity
                )
            adjusted += 0.02 * lexical_overlap(query_terms, candidate.text)
            if adjusted > best_score:
                best_score = adjusted
                best_idx = idx
        selected.append(candidates.pop(best_idx))
    return selected


def _text_similarity(left: str, right: str) -> float:
    left_terms = set(_matching_tokens(left))
    right_terms = set(_matching_tokens(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def normalize_context(text_value: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", (text_value or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0].strip()


def _debug_hit(hit: RetrievalRow) -> dict[str, Any]:
    return {
        "fragment_id": hit.fragment_id,
        "source_uri": hit.source_uri,
        "page": hit.page,
        "dense_score": round(hit.dense_score, 4),
        "lexical_score": round(hit.lexical_score, 4),
        "rerank_score": round(hit.rerank_score, 4) if hit.rerank_score is not None else None,
        "final_score": round(hit.final_score, 4),
        "score": round(hit.score, 4),
        "lexical_overlap": round(hit.lexical_overlap, 4),
        "document_score": round(hit.document_score, 4),
        "rrf_score": round(hit.rrf_score, 4),
        "rejection_reason": hit.rejection_reason,
    }


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

    terms = _query_terms_for_scoring(query)
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


def select_final_hits(hits: list[RetrievalRow], *, top_k: int, min_score: float) -> list[RetrievalRow]:
    normalized_hits = [
        hit if hit.final_score else replace(hit, final_score=hit.score, score=hit.score)
        for hit in hits
    ]
    eligible = [hit for hit in normalized_hits if hit.final_score >= _clamp_score(min_score)]
    return mmr_select(apply_adaptive_threshold(eligible), top_k=top_k, query_terms=[])


def rerank_by_keyword_relevance(
    query: str,
    hits: list[RetrievalRow],
    *,
    collection: str | None = None,
    source_uris: list[str] | None = None,
) -> list[RetrievalRow]:
    scored, _ = score_retrieval_candidates(query, hits, apply_noise_filter=False)
    return scored


def _topic_markers(query: str, *, collection: str | None = None, source_uris: list[str] | None = None) -> list[str]:
    markers: set[str] = set()
    query_lc = query.lower()
    _, topic_expansions = _expansion_maps(collection=collection, source_uris=source_uris)
    for token in _query_terms_for_scoring(query_lc):
        if len(token) >= 4:
            markers.add(token[:5])
        if len(token) >= 6:
            markers.add(token)

    for topic, topic_markers in topic_expansions.items():
        if topic in query_lc:
            markers.update(marker.lower() for marker in topic_markers)

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
    query_forms = []
    for term in query_terms:
        query_forms.extend(_lexical_forms(term))
    query_forms = list(dict.fromkeys(query_forms))
    tokenized = {row.fragment_id: _matching_tokens(f"{row.title or ''} {row.source_uri or ''} {row.text}") for row in rows}
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
        for term in query_forms:
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
