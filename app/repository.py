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
from app.document_intelligence import (
    analyze_query,
    cosine_similarity,
    detect_chunk_type,
    detect_chunk_type_details,
    infer_section_title_details,
    infer_section_title,
    is_toc_text,
    profile_lexical_score,
    profile_subject_score,
    profile_text,
    subject_expansions_for_query,
    text_quality_flags,
)
from app.embeddings import EmbeddingProvider
from app.models import Document, Embedding, Fragment
from app.reranker import CrossEncoderReranker
from app.schemas import CanonicalFragment

logger = logging.getLogger("rag_service")
MAX_DB_SNIPPET_CHARS = 450


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
    phrase_score: float = 0.0
    subject_score: float = 0.5
    section_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0
    lexical_overlap: float = 0.0
    document_score: float = 0.0
    rrf_score: float = 0.0
    rejection_reason: str | None = None
    exact_phrases: list[str] = field(default_factory=list)
    matched_phrases: list[str] = field(default_factory=list)
    missing_required_modifiers: list[str] = field(default_factory=list)
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
    required_terms: list[str] = field(default_factory=list)
    required_term_score: float = 0.0
    required_term_match_type: str = "none"
    is_navigation_index: bool = False
    navigation_filtered: bool = False
    intent_boost_applied: bool = False
    exercise_penalty_applied: bool = False
    test_question_penalty_applied: bool = False
    schema_boost_applied: bool = False
    term_penalty_applied: bool = False
    full_term_boost_applied: bool = False
    full_phrase_match: bool = False
    missing_required_terms: list[str] = field(default_factory=list)
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
    penalties_applied: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    hits: list[RetrievalRow]
    debug: dict[str, Any] | None = None


@dataclass
class SourceRow:
    source_uri: str
    title: str | None


@dataclass(frozen=True)
class RequiredExactPhrase:
    tokens: tuple[str, ...]
    lead: str
    modifier: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class PhraseModifierMatch:
    exact_phrases: list[str]
    matched_phrases: list[str]
    missing_required_modifiers: list[str]
    wrong_entity_modifier: bool
    phrase_score_before_penalty: float
    phrase_score_after_penalty: float


@dataclass(frozen=True)
class RequiredTermMatch:
    required_terms: list[str]
    score: float
    match_type: str


@dataclass(frozen=True)
class RequiredTermSignals:
    full_phrase_match: bool
    missing_required_terms: list[str]


class EmbeddingModelMismatchError(RuntimeError):
    pass


class RagRepository:
    def __init__(self, db: Session, embeddings: EmbeddingProvider, reranker: CrossEncoderReranker) -> None:
        self.db = db
        self.embeddings = embeddings
        self.reranker = reranker

    def upsert_document(self, source_uri: str, title: str | None, collection: str, meta: dict, reindex: bool) -> Document:
        doc = self.db.scalar(
            select(Document).where(
                Document.collection == collection,
                Document.source_uri == source_uri,
            )
        )
        incoming_fingerprint = meta.get("embedding_fingerprint")
        stored_fingerprint = (doc.meta or {}).get("embedding_fingerprint") if doc else None
        if doc and not reindex and incoming_fingerprint and stored_fingerprint != incoming_fingerprint:
            has_vectors = self.db.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM fragments f
                        JOIN embeddings e ON e.fragment_id = f.fragment_id
                        WHERE f.doc_id = CAST(:doc_id AS uuid)
                    )
                    """
                ),
                {"doc_id": str(doc.doc_id)},
            ).scalar()
            if has_vectors:
                raise EmbeddingModelMismatchError(
                    f"{source_uri} was indexed by another embedding model; run ingest with reindex=true"
                )
        if doc and reindex:
            self.db.execute(delete(Embedding).where(Embedding.fragment_id.in_(select(Fragment.fragment_id).where(Fragment.doc_id == doc.doc_id))))
            self.db.execute(delete(Fragment).where(Fragment.doc_id == doc.doc_id))
            self.db.flush()
        if doc:
            doc.title = title
            doc.collection = collection
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
            snippet=normalize_context(fragment.snippet or fragment.text, MAX_DB_SNIPPET_CHARS),
            meta=fragment.meta,
        )
        self.db.add(row)
        self.db.flush()

        count = 0
        embedding_text = str((fragment.meta or {}).get("search_text") or fragment.text)
        for idx, subchunk in enumerate(split_to_subchunks(embedding_text)):
            emb = self.embeddings.embed(subchunk)
            self.db.add(
                Embedding(
                    fragment_id=fragment.fragment_id,
                    subchunk_index=idx,
                    text=subchunk,
                    embedding=emb,
                    meta={
                        "offset": idx,
                        "search_text": True,
                        "embedding_fingerprint": getattr(self.embeddings, "model_fingerprint", None),
                    },
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
        include_toc: bool = False,
        include_low_quality: bool = False,
        include_navigation: bool = False,
    ) -> list[RetrievalRow]:
        return self.retrieve_with_debug(
            query,
            top_k,
            min_score,
            collection,
            source_uris,
            include_toc=include_toc,
            include_low_quality=include_low_quality,
            include_navigation=include_navigation,
            debug=False,
        ).hits

    def retrieve_with_debug(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
        include_toc: bool = False,
        include_low_quality: bool = False,
        include_navigation: bool = False,
        *,
        debug: bool = False,
    ) -> RetrievalResult:
        self.validate_embedding_compatibility(collection)
        normalized_query = normalize_query(query)
        query_terms = _query_terms_for_scoring(normalized_query)
        query_analysis = analyze_query(normalized_query)
        required_exact_phrases = _required_exact_phrases_for_query(normalized_query)
        expanded_queries = expand_query(
            normalized_query,
            collection=collection,
            source_uris=source_uris,
            query_analysis=query_analysis,
        )
        min_score = _clamp_score(min_score)
        recall_top_n = max(top_k, settings.vector_recall_top_n)

        diagnostics: dict[str, Any] = {
            "query": normalized_query,
            "top_k": top_k,
            "min_score": min_score,
            "collection": collection,
            "include_toc": include_toc,
            "include_low_quality": include_low_quality,
            "include_navigation": include_navigation,
            "query_analysis": query_analysis,
            "query_type": query_analysis.get("query_type", "unknown"),
            "detected_subjects": query_analysis.get("detected_subjects", []),
            "required_terms": query_analysis.get("required_terms", []),
            "exact_phrases": query_analysis.get("exact_phrases") or [_format_phrase(phrase.tokens) for phrase in required_exact_phrases],
            "expanded_queries": expanded_queries,
            "reranker_used": False,
            "score_component_notes": {},
            "dense_candidates": 0,
            "lexical_candidates": 0,
            "document_candidates": 0,
            "selected_documents": [],
            "rejected_documents": [],
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
        subject_scores: dict[str, float] = {}
        for idx, single_query in enumerate(expanded_queries):
            single_terms = _query_terms_for_scoring(single_query)
            doc_filter_uris, single_doc_scores, single_subject_scores, selected_docs, rejected_docs = self._document_prefilter(
                single_query,
                single_terms,
                query_analysis=query_analysis,
                collection=collection,
                source_uris=source_uris,
                limit=settings.document_prefilter_top_n,
            )
            for source_uri, score in single_doc_scores.items():
                doc_scores[source_uri] = max(doc_scores.get(source_uri, 0.0), score)
            for source_uri, score in single_subject_scores.items():
                subject_scores[source_uri] = max(subject_scores.get(source_uri, 0.0), score)
            diagnostics["selected_documents"].extend(selected_docs)
            diagnostics["rejected_documents"].extend(rejected_docs)
            if doc_filter_uris == []:
                candidate_groups.append([])
                continue

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
            row.subject_score = max(row.subject_score, subject_scores.get(row.source_uri, 0.5))
        diagnostics["candidates_after_fusion"] = len(fused_candidates)

        if not fused_candidates:
            logger.debug("retrieval_empty_after_fusion", extra=diagnostics)
            return RetrievalResult(hits=[], debug=diagnostics if debug else None)

        rerank_limit = max(top_k, settings.rerank_top_n)
        rerank_candidates = _balanced_prerank(
            fused_candidates,
            limit=rerank_limit,
            query_terms=query_terms,
        )
        self._hydrate_fragment_metadata(rerank_candidates)
        rerank_candidates = self._expand_context(rerank_candidates)
        diagnostics["candidates_pruned_before_rerank"] = max(0, len(fused_candidates) - len(rerank_candidates))
        diagnostics["candidates_after_noise_filter"] = len(rerank_candidates)

        if self.reranker.available:
            raw_rerank_scores = self.reranker.score(normalized_query, [item.text for item in rerank_candidates])
            diagnostics["reranker_used"] = True
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
            diagnostics["score_component_notes"]["rerank_score"] = "0.0 because reranker is unavailable"

        scored_candidates, rerank_rejected = score_retrieval_candidates(
            normalized_query,
            rerank_candidates,
            rerank_raw_scores=raw_rerank_scores,
            query_analysis=query_analysis,
            apply_noise_filter=True,
        )
        diagnostics["rejected_results"].extend(rerank_rejected)

        thresholded: list[RetrievalRow] = []
        for hit in scored_candidates:
            if hit.is_toc and not include_toc:
                diagnostics["rejected_results"].append(_debug_hit(replace(hit, rejection_reason="toc_filtered", toc_filtered=True)))
                continue
            if hit.low_text_quality and not include_low_quality:
                diagnostics["rejected_results"].append(_debug_hit(replace(hit, rejection_reason="low_text_quality_filtered", low_quality_filtered=True)))
                continue
            if hit.is_navigation_index and not include_navigation:
                diagnostics["rejected_results"].append(_debug_hit(replace(hit, rejection_reason="navigation_filtered", navigation_filtered=True)))
                continue
            if hit.final_score >= min_score:
                thresholded.append(hit)
            else:
                diagnostics["rejected_results"].append(_debug_hit(replace(hit, rejection_reason="below_min_score")))
        diagnostics["candidates_after_threshold"] = len(thresholded)
        adaptive_hits = apply_adaptive_threshold(thresholded)
        adaptive_ids = {hit.fragment_id for hit in adaptive_hits}
        for hit in thresholded:
            if hit.fragment_id not in adaptive_ids:
                diagnostics["rejected_results"].append(_debug_hit(replace(hit, rejection_reason="adaptive_tail_cut")))
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

    def embedding_compatibility(self, collection: str | None = None) -> dict[str, Any]:
        expected = getattr(self.embeddings, "model_fingerprint", None)
        if not expected or not settings.enforce_embedding_model_compatibility:
            return {"compatible": True, "expected": expected, "indexed_documents": 0, "incompatible_sources": []}

        collection_clause = "AND d.collection = :collection" if collection is not None else ""
        rows = self.db.execute(
            text(
                f"""
                SELECT d.source_uri, d.meta->>'embedding_fingerprint' AS fingerprint
                FROM documents d
                WHERE EXISTS (
                    SELECT 1
                    FROM fragments f
                    JOIN embeddings e ON e.fragment_id = f.fragment_id
                    WHERE f.doc_id = d.doc_id
                )
                {collection_clause}
                ORDER BY d.source_uri
                """
            ),
            {"collection": collection} if collection is not None else {},
        ).mappings().all()
        incompatible = [row["source_uri"] for row in rows if row["fingerprint"] != expected]
        return {
            "compatible": not incompatible,
            "expected": expected,
            "indexed_documents": len(rows),
            "incompatible_sources": incompatible[:20],
        }

    def validate_embedding_compatibility(self, collection: str | None = None) -> None:
        status = self.embedding_compatibility(collection)
        if not status["compatible"]:
            sample = ", ".join(status["incompatible_sources"][:3])
            raise EmbeddingModelMismatchError(
                "Embedding index is incompatible with the configured model"
                f"{f': {sample}' if sample else ''}; reindex the affected collection"
            )

    def _document_prefilter(
        self,
        query: str,
        query_terms: list[str],
        *,
        query_analysis: dict[str, Any],
        collection: str,
        source_uris: list[str] | None,
        limit: int,
    ) -> tuple[list[str] | None, dict[str, float], dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
        profiles = self._document_profiles(collection=collection, source_uris=source_uris)
        subject_scores = {
            source_uri: profile_subject_score(query_analysis, profile)
            for source_uri, profile in profiles.items()
        }
        if source_uris or not settings.document_prefilter_enabled:
            selected = [
                _debug_document_profile(uri, profiles.get(uri, {}), 1.0, subject_scores.get(uri, 0.5), "explicit_source_filter")
                for uri in source_uris or []
            ]
            return source_uris, {uri: 1.0 for uri in source_uris or []}, subject_scores, selected, []

        query_vector = self.embeddings.embed(query)
        routed: list[tuple[str, float, float, float, float, dict[str, Any]]] = []
        rejected: list[dict[str, Any]] = []
        high_subject_confidence = float(query_analysis.get("subject_confidence") or 0.0) >= settings.retrieval_subject_confidence_threshold

        for source_uri, profile in profiles.items():
            lexical_profile_score = profile_lexical_score(query_analysis, profile)
            subject_score = subject_scores.get(source_uri, 0.5)
            embedding_score = max(
                0.0,
                2.0 * cosine_similarity(query_vector, profile.get("summary_embedding")) - 1.0,
            )
            rank_score = _clamp_score(0.55 * lexical_profile_score + 0.45 * embedding_score)
            score = _clamp_score(
                0.35 * rank_score
                + 0.30 * lexical_profile_score
                + 0.20 * subject_score
                + 0.15 * embedding_score
            )
            reason = None
            if high_subject_confidence and subject_score <= settings.retrieval_subject_mismatch_score and lexical_profile_score < 0.35:
                reason = "subject_mismatch"
                score *= settings.retrieval_subject_mismatch_penalty
            elif score < settings.document_routing_min_score and lexical_profile_score <= 0.0:
                reason = "low_document_score"

            if reason:
                rejected.append(_debug_document_profile(source_uri, profile, score, subject_score, reason))
                continue
            routed.append((source_uri, score, rank_score, lexical_profile_score, subject_score, profile))

        routed.sort(key=lambda item: item[1], reverse=True)
        selected_routed = routed[:limit]
        selected_uris = [item[0] for item in selected_routed]
        doc_scores = {source_uri: score for source_uri, score, *_ in selected_routed}
        selected = [
            _debug_document_profile(source_uri, profile, score, subject_score, "selected")
            | {"rank_score": round(rank_score, 4), "profile_lexical_score": round(lexical_profile_score, 4)}
            for source_uri, score, rank_score, lexical_profile_score, subject_score, profile in selected_routed
        ]
        if not selected_uris:
            return [], {}, subject_scores, selected, rejected
        return selected_uris, doc_scores, subject_scores, selected, rejected

    def _document_profiles(self, *, collection: str, source_uris: list[str] | None) -> dict[str, dict[str, Any]]:
        stmt = select(Document).where(Document.collection == collection)
        if source_uris:
            stmt = stmt.where(Document.source_uri.in_(source_uris))
        rows = self.db.scalars(stmt).all()
        profiles: dict[str, dict[str, Any]] = {}
        for doc in rows:
            meta = doc.meta or {}
            profile = dict(meta.get("document_profile") or {})
            if not profile:
                profile = {
                    "source_uri": doc.source_uri,
                    "title": doc.title,
                    "subject": meta.get("subject") or "unknown",
                    "grade": meta.get("grade"),
                    "doc_type": meta.get("doc_type") or "unknown",
                    "language": meta.get("language") or "unknown",
                    "keywords": meta.get("keywords") or [],
                    "section_titles": meta.get("section_titles") or [],
                }
            profile.setdefault("source_uri", doc.source_uri)
            profile.setdefault("title", doc.title)
            profiles[doc.source_uri] = profile
        return profiles

    def _hydrate_fragment_metadata(self, rows: list[RetrievalRow]) -> None:
        fragment_ids = [row.fragment_id for row in rows if not row.meta]
        if not fragment_ids:
            return
        db_rows = self.db.execute(
            select(Fragment.fragment_id, Fragment.meta).where(Fragment.fragment_id.in_(fragment_ids))
        ).mappings().all()
        meta_by_id = {row["fragment_id"]: row["meta"] or {} for row in db_rows}
        for row in rows:
            if not row.meta:
                row.meta = meta_by_id.get(row.fragment_id, {})

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
            params[param_name] = _sql_token_pattern(term_value)
            text_expr = _sql_normalized_text("f.text")
            title_expr = _sql_normalized_text("d.title")
            source_expr = _sql_normalized_text("d.source_uri")
            doc_meta_expr = _sql_normalized_text("d.meta::text")
            fragment_meta_expr = _sql_normalized_text("f.meta::text")
            conditions.append(
                f"({text_expr} ~ :{param_name} "
                f"OR {title_expr} ~ :{param_name} "
                f"OR {source_expr} ~ :{param_name} "
                f"OR {doc_meta_expr} ~ :{param_name} "
                f"OR {fragment_meta_expr} ~ :{param_name})"
            )
            rank_parts.append(
                f"(CASE WHEN {text_expr} ~ :{param_name} THEN 1 ELSE 0 END)"
                f" + (CASE WHEN {doc_meta_expr} ~ :{param_name} THEN 1 ELSE 0 END)"
            )

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
            if hit.expanded_from_neighbors or (hit.meta or {}).get("expanded_from_neighbors") is True:
                expanded.append(hit)
                continue
            if not hit.doc_id or hit.element_index is None:
                expanded.append(hit)
                continue

            rows = self.db.execute(
                text(
                    """
                    WITH ordered AS (
                        SELECT
                            text,
                            meta,
                            page,
                            element_index,
                            row_number() OVER (ORDER BY element_index ASC) AS rn
                        FROM fragments
                        WHERE doc_id = CAST(:doc_id AS uuid)
                    ),
                    target AS (
                        SELECT rn, page, meta
                        FROM ordered
                        WHERE element_index = :element_index
                        LIMIT 1
                    )
                    SELECT o.text, o.meta, o.page, o.element_index
                    FROM ordered o, target t
                    WHERE o.rn BETWEEN t.rn - :neighbors AND t.rn + :neighbors
                    ORDER BY o.rn ASC
                    """
                ),
                {
                    "doc_id": hit.doc_id,
                    "element_index": hit.element_index,
                    "neighbors": neighbors,
                },
            ).mappings().all()
            if not rows:
                expanded.append(hit)
                continue

            same_scope = _same_context_scope(hit, rows)
            context_rows = same_scope or rows
            context = normalize_context(" ".join(str(row["text"]) for row in context_rows if row["text"]), settings.context_expansion_max_chars)
            if not context or context == hit.text:
                expanded.append(hit)
                continue

            meta = dict(hit.meta or {})
            meta.setdefault("search_text", hit.text)
            meta.setdefault("search_snippet", hit.snippet)
            if hit.chunk_type and hit.chunk_type != "unknown":
                meta.setdefault("chunk_type", hit.chunk_type)
            if hit.chunk_type_reason:
                meta.setdefault("chunk_type_reason", hit.chunk_type_reason)
            meta["expanded_text"] = context
            meta["expanded_from_neighbors"] = True
            quality = text_quality_flags(context, page=hit.page, meta=meta)
            refreshed_meta = {
                key: value
                for key, value in meta.items()
                if key not in {"chunk_type", "chunk_type_reason", "is_navigation_index"}
            }
            chunk_details = detect_chunk_type_details(context, meta=refreshed_meta, page=hit.page)
            meta["chunk_type"] = chunk_details.get("chunk_type", "unknown")
            meta["chunk_type_reason"] = chunk_details.get("chunk_type_reason")
            meta["is_navigation_index"] = meta["chunk_type"] == "navigation_index"
            for key in (
                "is_toc",
                "is_index",
                "is_bibliography",
                "is_caption",
                "is_fragmented",
                "is_too_short",
                "starts_mid_word",
                "low_text_quality",
            ):
                meta[key] = bool(quality.get(key))
            meta["quality_score"] = float(quality.get("quality_score") or hit.quality_score)
            meta["low_text_quality_reason"] = quality.get("low_text_quality_reason")
            expanded.append(
                replace(
                    hit,
                    text=context,
                    meta=meta,
                    expanded_from_neighbors=True,
                    is_fragmented=bool(quality.get("is_fragmented")),
                    is_too_short=bool(quality.get("is_too_short")),
                    starts_mid_word=bool(quality.get("starts_mid_word")),
                    low_text_quality=bool(quality.get("low_text_quality")),
                    quality_score=max(hit.quality_score, float(quality.get("quality_score") or hit.quality_score)),
                    low_text_quality_reason=quality.get("low_text_quality_reason"),
                )
            )
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
        subchunk_limit = max(limit, limit * max(1, settings.vector_subchunk_oversample))
        params: dict = {
            "qvec": _vector_literal(qvec),
            "collection": collection,
            "recall_top_n": limit,
            "subchunk_recall_top_n": subchunk_limit,
        }
        if source_uris:
            source_clause = " AND d.source_uri = ANY(:source_uris)"
            params["source_uris"] = source_uris

        sql = text(
            f"""
            WITH nearest_subchunks AS MATERIALIZED (
                SELECT
                    e.fragment_id,
                    e.embedding <=> CAST(:qvec AS vector) AS distance
                FROM embeddings e
                JOIN fragments nf ON nf.fragment_id = e.fragment_id
                JOIN documents nd ON nd.doc_id = nf.doc_id
                WHERE nd.collection = :collection{source_clause.replace("d.source_uri", "nd.source_uri")}
                ORDER BY e.embedding <=> CAST(:qvec AS vector)
                LIMIT :subchunk_recall_top_n
            )
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
                MIN(e.distance) AS distance
            FROM nearest_subchunks e
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

        params: dict = {
            "collection": collection,
            "recall_top_n": limit,
            "keyword_query": " OR ".join(terms),
        }

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
                f.meta,
                ts_rank_cd(
                    f.search_tsv,
                    websearch_to_tsquery('russian', :keyword_query)
                ) AS lexical_rank
            FROM fragments f
            JOIN documents d ON d.doc_id = f.doc_id
            WHERE d.collection = :collection{source_clause}
              AND f.search_tsv @@ websearch_to_tsquery('russian', :keyword_query)
            ORDER BY
                ts_rank_cd(f.search_tsv, websearch_to_tsquery('russian', :keyword_query)) DESC,
                f.fragment_id ASC
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
                score=_normalize_fts_rank(row["lexical_rank"]),
                text=row["text"],
                meta=row["meta"] or {},
                lexical_score=_normalize_fts_rank(row["lexical_rank"]),
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


def _normalize_fts_rank(value: Any) -> float:
    try:
        rank = max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0
    return _clamp_score(rank / (rank + 0.5)) if rank else 0.0


def _normalize_text_for_matching(text: str) -> str:
    normalized = (text or "").lower().replace("ё", "е")
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = re.sub(r"([a-zа-я]+)-\s+([a-zа-я]+)", r"\1\2", normalized)
    return normalized


def _sql_normalized_text(column_expr: str) -> str:
    return (
        f"regexp_replace("
        f"regexp_replace(lower(coalesce({column_expr}, '')), 'ё', 'е', 'g'), "
        f"'([a-zа-я]+)-[[:space:]]+([a-zа-я]+)', '\\1\\2', 'g'"
        f")"
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9a-zа-я]+", _normalize_text_for_matching(text))


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
    "как",
    "что",
    "кто",
    "какой",
    "какая",
    "какое",
    "какие",
    "когда",
    "почему",
    "где",
    "чем",
    "сколько",
    "такое",
    "был",
    "была",
    "было",
    "были",
    "бывает",
    "бывают",
    "изучает",
    "изучают",
    "отвечает",
    "отвечают",
    "отличить",
    "расставлять",
    "ставится",
    "ставятся",
    "устроено",
    "с",
    "со",
    "у",
    "and",
    "of",
    "the",
}


TOPICAL_EXPANSIONS: dict[str, tuple[tuple[str, ...], ...]] = {
    "правление": (
        ("царствование",),
        ("реформы",),
        ("внутренняя", "политика"),
        ("внешняя", "политика"),
        ("война",),
        ("конгресс",),
        ("крепостное", "право"),
    ),
    "царствование": (
        ("правление",),
        ("реформы",),
        ("внутренняя", "политика"),
        ("внешняя", "политика"),
        ("война",),
        ("конгресс",),
        ("крепостное", "право"),
    ),
}


def _query_terms_for_scoring(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _tokenize(text):
        if token in QUERY_STOPWORDS:
            continue
        if _is_weak_standalone_token(token):
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
        if _is_weak_standalone_token(term):
            continue
        if len(term) >= 3 or any(char.isdigit() for char in term):
            terms.append(term)
        elif len(term) >= 2 and not _is_roman_numeral(term):
            terms.append(term)
    return terms[:limit]


def _sql_token_pattern(term: str) -> str:
    forms = sorted(
        {
            form
            for form in _lexical_forms(term)
            if form and not _is_weak_standalone_token(form) and (len(form) >= 2 or any(char.isdigit() for char in form))
        },
        key=len,
        reverse=True,
    )
    if not forms:
        forms = [re.escape(term)]
    alternatives = "|".join(re.escape(form) + ("[[:alnum:]_]*" if not form.isdigit() else "") for form in forms)
    return f"(^|[^[:alnum:]_])({alternatives})([^[:alnum:]_]|$)"


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
                phrase_score=max(current.phrase_score, row.phrase_score),
                subject_score=max(current.subject_score, row.subject_score),
                section_score=max(current.section_score, row.section_score),
                rrf_score=max(current.rrf_score, row.rrf_score),
                document_score=max(current.document_score, row.document_score),
                score=max(current.score, row.score),
            )
    return list(merged.values())


def _cheap_prerank(candidates: list[RetrievalRow]) -> list[RetrievalRow]:
    """Select the small cross-encoder pool using already-computed recall signals."""
    return sorted(
        candidates,
        key=lambda row: (
            0.45 * _clamp_score(row.rrf_score)
            + 0.25 * _clamp_score(row.dense_score or row.score)
            + 0.15 * _clamp_score(row.lexical_score)
            + 0.10 * _clamp_score(row.document_score)
            + 0.05 * _clamp_score(row.subject_score)
        ),
        reverse=True,
    )


def _balanced_prerank(
    candidates: list[RetrievalRow],
    *,
    limit: int,
    query_terms: list[str],
) -> list[RetrievalRow]:
    """Keep both semantic and lexical recall paths represented in the CPU rerank pool."""
    if limit <= 0:
        return []
    combined = _cheap_prerank(candidates)
    lexical = sorted(
        (row for row in candidates if row.lexical_score > 0.0),
        key=lambda row: (
            lexical_overlap(
                query_terms,
                " ".join(
                    str(value)
                    for value in (
                        row.text,
                        (row.meta or {}).get("section_title"),
                        " ".join(str(item) for item in ((row.meta or {}).get("section_path") or [])),
                    )
                    if value
                ),
            ),
            row.lexical_score,
            row.rrf_score,
            row.dense_score,
        ),
        reverse=True,
    )
    selected: dict[str, RetrievalRow] = {}

    def take(rows: list[RetrievalRow], count: int) -> None:
        for row in rows:
            if len(selected) >= count:
                break
            selected.setdefault(row.fragment_id, row)

    combined_quota = max(1, (limit + 1) // 2)
    take(combined, combined_quota)
    take(lexical, limit)
    take(combined, limit)
    return list(selected.values())[:limit]


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


def _is_weak_standalone_token(token: str) -> bool:
    token = (token or "").lower()
    if len(token) == 1 and (token.isalpha() or _is_roman_numeral(token)):
        return True
    return False


def _format_phrase(phrase: tuple[str, ...]) -> str:
    return " ".join(phrase)


def _required_exact_phrases_for_query(text: str) -> list[RequiredExactPhrase]:
    tokens = _tokenize(text)
    phrases: list[RequiredExactPhrase] = []
    seen: set[tuple[str, ...]] = set()

    def add(tokens_value: tuple[str, ...], lead: str, modifier: tuple[str, ...], kind: str) -> None:
        if not tokens_value or tokens_value in seen:
            return
        phrases.append(RequiredExactPhrase(tokens=tokens_value, lead=lead, modifier=modifier, kind=kind))
        seen.add(tokens_value)

    for idx, token in enumerate(tokens[:-1]):
        nxt = tokens[idx + 1]
        if token in QUERY_STOPWORDS or _is_weak_standalone_token(token):
            continue
        if not _is_required_modifier_start(nxt):
            continue
        phrase_tokens = [token, nxt]
        if idx + 2 < len(tokens) and _is_modifier_continuation(nxt, tokens[idx + 2]):
            phrase_tokens.append(tokens[idx + 2])
        modifier = tuple(phrase_tokens[1:])
        kind = "name_roman" if _is_roman_numeral(nxt) else "term_modifier"
        add(tuple(phrase_tokens), token, modifier, kind)

    for phrase in _letter_rule_phrases(text):
        phrase_tokens = tuple(_tokenize(phrase))
        if phrase_tokens:
            add(phrase_tokens, phrase_tokens[0], phrase_tokens[1:], "abbreviation_phrase")

    for acronym in _strong_abbreviation_tokens(text):
        if any(len(phrase.tokens) > 1 and acronym in phrase.tokens for phrase in phrases):
            continue
        add((acronym,), acronym, tuple(), "abbreviation")

    return phrases


def _is_required_modifier_start(token: str) -> bool:
    token = (token or "").lower()
    if not token or token in QUERY_STOPWORDS:
        return False
    if token.isdigit() or _is_roman_numeral(token):
        return True
    if any(char.isdigit() for char in token):
        return True
    return len(token) == 1 and bool(re.fullmatch(r"[a-z]", token))


def _is_modifier_continuation(previous: str, token: str) -> bool:
    token = (token or "").lower()
    if not token or token in QUERY_STOPWORDS:
        return False
    if previous.isdigit() and token.isalpha() and 2 <= len(token) <= 4:
        return True
    return any(char.isdigit() for char in previous) and token.isalpha() and 1 <= len(token) <= 4


def _strong_abbreviation_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"(?<![0-9A-Za-zА-Яа-яЁё])(?:[A-ZА-ЯЁ]{2,}[0-9]*|[A-ZА-ЯЁ]+[0-9]{2,})(?![0-9A-Za-zА-Яа-яЁё])")
    for match in pattern.finditer(text or ""):
        if match.start() >= 2 and (text or "")[match.start() - 1] == "-" and (text or "")[match.start() - 2].isdigit():
            continue
        normalized = _tokenize(match.group(0))
        if len(normalized) != 1:
            continue
        token = normalized[0]
        if _is_roman_numeral(token) or token in seen:
            continue
        tokens.append(token)
        seen.add(token)
    return tokens


def _letter_rule_phrases(text: str) -> list[str]:
    pattern = re.compile(r"(?<![0-9A-Za-zА-Яа-яЁё])[A-ZА-ЯЁ]\s+(?:и|and)\s+[A-ZА-ЯЁ]{2,}(?![0-9A-Za-zА-Яа-яЁё])")
    return [match.group(0) for match in pattern.finditer(text or "")]


def _phrase_modifier_match(
    required_phrases: list[RequiredExactPhrase],
    text_value: str,
    phrase_score_before_penalty: float,
) -> PhraseModifierMatch:
    exact_phrases = [_format_phrase(phrase.tokens) for phrase in required_phrases]
    if not required_phrases:
        score = _clamp_score(phrase_score_before_penalty)
        return PhraseModifierMatch(
            exact_phrases=[],
            matched_phrases=[],
            missing_required_modifiers=[],
            wrong_entity_modifier=False,
            phrase_score_before_penalty=score,
            phrase_score_after_penalty=score,
        )

    text_tokens = _tokenize(text_value)
    matched: list[str] = []
    missing: list[str] = []
    wrong_modifier = False
    for phrase in required_phrases:
        phrase_label = _format_phrase(phrase.tokens)
        if _unit_matches(phrase.tokens, text_tokens):
            matched.append(phrase_label)
            continue
        missing.append(phrase_label)
        wrong_modifier = wrong_modifier or _has_conflicting_entity_modifier(phrase, text_tokens)

    score = _clamp_score(phrase_score_before_penalty)
    if not missing and matched:
        score = max(score, 0.75)
    elif missing:
        matched_ratio = len(matched) / len(required_phrases)
        max_allowed = 0.2 if matched_ratio <= 0 else max(0.2, 0.75 * matched_ratio)
        score = min(score, max_allowed)
    if wrong_modifier:
        score = min(score, 0.2)

    return PhraseModifierMatch(
        exact_phrases=exact_phrases,
        matched_phrases=matched,
        missing_required_modifiers=missing,
        wrong_entity_modifier=wrong_modifier,
        phrase_score_before_penalty=_clamp_score(phrase_score_before_penalty),
        phrase_score_after_penalty=_clamp_score(score),
    )


def _has_conflicting_entity_modifier(required: RequiredExactPhrase, text_tokens: list[str]) -> bool:
    if not required.modifier or not text_tokens:
        return False
    phrase_len = len(required.tokens)
    for idx, token in enumerate(text_tokens):
        if not _phrase_window_matches((required.lead,), [token]):
            continue
        exact_window = text_tokens[idx : idx + phrase_len]
        if len(exact_window) == phrase_len and _phrase_window_matches(required.tokens, exact_window):
            continue
        if idx + 1 >= len(text_tokens):
            continue
        next_token = text_tokens[idx + 1]
        candidate_modifier = tuple(text_tokens[idx + 1 : idx + 1 + len(required.modifier)])
        if len(candidate_modifier) == len(required.modifier) and _looks_like_modifier_sequence(candidate_modifier):
            if not _phrase_window_matches(required.modifier, list(candidate_modifier)):
                return True
        if _looks_like_name_family_conflict(required, next_token):
            return True
    return False


def _looks_like_modifier_sequence(tokens: tuple[str, ...]) -> bool:
    if not tokens:
        return False
    first = tokens[0]
    if _is_required_modifier_start(first):
        return True
    return len(tokens) > 1 and tokens[0].isdigit() and all(len(token) <= 4 for token in tokens[1:])


def _looks_like_name_family_conflict(required: RequiredExactPhrase, token: str) -> bool:
    if not required.modifier or not _is_roman_numeral(required.modifier[0]):
        return False
    lead_forms = [form for form in _lexical_forms(required.lead) if len(form) >= 5]
    token_forms = [form for form in _lexical_forms(token) if len(form) >= 5]
    for lead_form in lead_forms:
        prefix = lead_form[: min(8, len(lead_form))]
        if any(form.startswith(prefix) and form != lead_form for form in token_forms):
            return True
    return False


def _query_phrases_for_scoring(text: str) -> list[tuple[str, ...]]:
    tokens = _tokenize(text)
    phrases: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    for idx, token in enumerate(tokens[:-1]):
        nxt = tokens[idx + 1]
        if token in QUERY_STOPWORDS:
            continue
        if (_is_roman_numeral(nxt) or nxt.isdigit()) and not _is_weak_standalone_token(token):
            phrase = (token, nxt)
            if phrase not in seen:
                phrases.append(phrase)
                seen.add(phrase)

    significant = [token for token in tokens if token not in QUERY_STOPWORDS and not _is_weak_standalone_token(token)]
    for left, right in zip(significant, significant[1:]):
        if len(left) >= 4 and len(right) >= 4:
            phrase = (left, right)
            if phrase not in seen:
                phrases.append(phrase)
                seen.add(phrase)
    return phrases


def _anchor_phrases_for_query(text: str) -> list[tuple[str, ...]]:
    tokens = _tokenize(text)
    anchors: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for idx, token in enumerate(tokens[:-1]):
        nxt = tokens[idx + 1]
        if token in QUERY_STOPWORDS or _is_weak_standalone_token(token):
            continue
        if _is_roman_numeral(nxt) or nxt.isdigit():
            phrase = (token, nxt)
            if phrase not in seen:
                anchors.append(phrase)
                seen.add(phrase)
    return anchors


def _topic_units_for_query(text: str) -> list[tuple[tuple[str, ...], float, bool]]:
    query_terms = _query_terms_for_scoring(text)
    anchor_leads = {phrase[0] for phrase in _anchor_phrases_for_query(text)}
    units: list[tuple[tuple[str, ...], float, bool]] = []
    seen: set[tuple[str, ...]] = set()

    for term in query_terms:
        if term in anchor_leads:
            continue
        unit = (term,)
        if unit not in seen:
            units.append((unit, 1.0, False))
            seen.add(unit)
        for seed, expansions in TOPICAL_EXPANSIONS.items():
            if not (_lexical_forms(seed) & _lexical_forms(term)):
                continue
            for expansion in expansions:
                if expansion in seen:
                    continue
                units.append((expansion, 0.7, True))
                seen.add(expansion)

    for phrase in _query_phrases_for_scoring(text):
        if len(phrase) < 2 or any(part in anchor_leads for part in phrase):
            continue
        if phrase not in seen:
            units.append((phrase, 1.15, False))
            seen.add(phrase)
    return units


def phrase_match_score(query: str, text_value: str) -> float:
    phrases = _query_phrases_for_scoring(query)
    if not phrases:
        return 0.0
    text_tokens = _tokenize(text_value)
    if not text_tokens:
        return 0.0

    matches = 0
    for phrase in phrases:
        phrase_len = len(phrase)
        for idx in range(0, len(text_tokens) - phrase_len + 1):
            window = text_tokens[idx : idx + phrase_len]
            if _phrase_window_matches(phrase, window):
                matches += 1
                break
    return matches / len(phrases)


def anchor_phrase_score(query: str, text_value: str) -> float:
    anchors = _anchor_phrases_for_query(query)
    if not anchors:
        return 0.0
    return _weighted_unit_score([(phrase, 1.0, False) for phrase in anchors], text_value, use_expansion_discount=False)


def topical_match_score(query: str, text_value: str) -> float:
    return _weighted_unit_score(_topic_units_for_query(query), text_value, use_expansion_discount=True)


def weighted_phrase_term_score(query: str, text_value: str) -> float:
    anchor_score = anchor_phrase_score(query, text_value)
    topic_score = topical_match_score(query, text_value)
    has_anchor = bool(_anchor_phrases_for_query(query))
    has_topic = bool(_topic_units_for_query(query))

    if has_anchor and has_topic:
        if anchor_score > 0 and topic_score > 0:
            return _clamp_score(0.45 * anchor_score + 0.55 * topic_score)
        if anchor_score > 0:
            return min(anchor_score, 0.45)
        return min(topic_score, 0.55)
    if has_anchor:
        return anchor_score
    return topic_score


def _weighted_unit_score(
    units: list[tuple[tuple[str, ...], float, bool]],
    text_value: str,
    *,
    use_expansion_discount: bool,
) -> float:
    if not units:
        return 0.0
    text_tokens = _tokenize(text_value)
    if not text_tokens:
        return 0.0

    exact_total = sum(weight for _, weight, is_expansion in units if not is_expansion)
    exact_matched = 0.0
    expansion_matched = 0.0
    for unit, weight, is_expansion in units:
        if not _unit_matches(unit, text_tokens):
            continue
        if is_expansion:
            expansion_matched += weight
        else:
            exact_matched += weight

    exact_score = exact_matched / exact_total if exact_total > 0 else 0.0
    if not use_expansion_discount:
        total = sum(weight for _, weight, _ in units)
        return _clamp_score((exact_matched + expansion_matched) / total if total else 0.0)
    expansion_score = min(1.0, expansion_matched)
    return _clamp_score(max(exact_score, 0.8 * expansion_score))


def _unit_matches(unit: tuple[str, ...], text_tokens: list[str]) -> bool:
    unit_len = len(unit)
    for idx in range(0, len(text_tokens) - unit_len + 1):
        window = text_tokens[idx : idx + unit_len]
        if _phrase_window_matches(unit, window):
            return True
    return False


def _phrase_window_matches(phrase: tuple[str, ...], window: list[str]) -> bool:
    for expected, actual in zip(phrase, window):
        if _is_roman_numeral(expected) or expected.isdigit():
            if expected != actual:
                return False
            continue
        if expected in {"г", "год", "года", "году", "годы"} and actual in {
            "г",
            "год",
            "года",
            "году",
            "годы",
        }:
            continue
        if not (_lexical_forms(expected) & _lexical_forms(actual)):
            return False
    return True


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
            "о",
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


def _required_terms_for_scoring(query: str, query_analysis: dict[str, Any]) -> list[str]:
    analysis_terms = query_analysis.get("required_terms")
    if isinstance(analysis_terms, list):
        terms = [str(term).strip().lower().replace("ё", "е") for term in analysis_terms if str(term).strip()]
        if terms:
            return terms
    if query_analysis.get("query_type") not in {"concept_lookup", "definition", "rule_lookup"}:
        return []
    if _required_exact_phrases_for_query(query):
        return []
    terms = _query_terms_for_scoring(query)
    if 2 <= len(terms) <= 5:
        return [" ".join(terms)]
    return []


def required_term_match_score(required_terms: list[str], text_value: str) -> RequiredTermMatch:
    normalized_terms = [normalize_for_required_term(term) for term in required_terms if _tokenize(term)]
    if not normalized_terms:
        return RequiredTermMatch(required_terms=[], score=0.0, match_type="none")
    normalized_text = normalize_for_required_term(text_value)
    text_tokens = tokenize_required_text(normalized_text)
    if not text_tokens:
        return RequiredTermMatch(required_terms=normalized_terms, score=0.0, match_type="none")

    best_score = 0.0
    best_type = "none"
    type_rank = {"none": 0, "partial": 1, "all_terms": 2, "full_phrase": 3, "full_phrase_prefix": 4}
    text_forms = set(_matching_tokens(text_value))

    for term in normalized_terms:
        term_tokens = tuple(tokenize_required_text(term))
        if not term_tokens:
            continue
        exact_prefix = bool(term and (normalized_text == term or normalized_text.startswith(f"{term} ")))
        exact_anywhere = bool(term and f" {term} " in f" {normalized_text} ")
        ordered_phrase = has_ordered_phrase(term_tokens, text_tokens)
        if exact_prefix:
            score, match_type = 1.0, "full_phrase_prefix"
        elif exact_anywhere or ordered_phrase:
            score, match_type = 1.0, "full_phrase"
        elif _all_terms_match(term_tokens, text_forms):
            score, match_type = 0.8, "all_terms"
        else:
            matched = sum(1 for token in term_tokens if _lexical_forms(token) & text_forms)
            if matched:
                score = min(0.25, 0.25 * matched / len(term_tokens))
                match_type = "partial"
            else:
                score, match_type = 0.0, "none"
        if score > best_score or (math.isclose(score, best_score) and type_rank[match_type] > type_rank[best_type]):
            best_score = score
            best_type = match_type

    return RequiredTermMatch(required_terms=normalized_terms, score=_clamp_score(best_score), match_type=best_type)


def normalize_for_required_term(text_value: str) -> str:
    normalized = (text_value or "").lower().replace("ё", "е").replace("\u00ad", "")
    normalized = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", normalized)
    normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "", normalized)
    normalized = re.sub(r"[\r\n\t]+", " ", normalized)
    normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize_required_text(text_value: str) -> list[str]:
    return re.findall(r"[a-zа-я0-9]+", normalize_for_required_term(text_value), flags=re.IGNORECASE)


def has_ordered_phrase(query_terms: tuple[str, ...] | list[str], text_tokens: list[str]) -> bool:
    term_len = len(query_terms)
    if term_len <= 0 or len(text_tokens) < term_len:
        return False
    phrase = tuple(query_terms)
    return any(
        _phrase_window_matches(phrase, text_tokens[idx : idx + term_len])
        for idx in range(0, len(text_tokens) - term_len + 1)
    )


def _all_terms_match(query_terms: tuple[str, ...] | list[str], text_forms: set[str]) -> bool:
    return bool(query_terms) and all(_lexical_forms(token) & text_forms for token in query_terms)


def _normalize_required_term_text(text_value: str) -> str:
    return normalize_for_required_term(text_value)


def _candidate_required_term_text(row: RetrievalRow, meta: dict[str, Any]) -> str:
    parts: list[str] = []
    seen_parts: set[str] = set()
    def add_part(value: Any) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        normalized = normalize_for_required_term(value)
        if normalized in seen_parts:
            return
        seen_parts.add(normalized)
        parts.append(value)

    for value in (row.text, row.snippet, meta.get("expanded_text")):
        add_part(value)
    for key in ("section_title", "inferred_section_title", "parent_heading", "heading", "nearest_heading"):
        add_part(meta.get(key))
    for key in ("section_path", "heading_path", "nearest_headings"):
        value = meta.get(key)
        if isinstance(value, list):
            for item in value:
                add_part(str(item))
    return " ".join(parts)


def required_term_signals(required_terms: list[str], text_value: str, match: RequiredTermMatch) -> RequiredTermSignals:
    full_phrase_match = bool(required_terms) and match.match_type in {"full_phrase_prefix", "full_phrase"} and match.score >= 0.8
    if not required_terms:
        return RequiredTermSignals(full_phrase_match=False, missing_required_terms=[])

    text_forms = set(_matching_tokens(text_value))
    missing: list[str] = []
    seen: set[str] = set()
    for term in required_terms:
        for token in _tokenize(term):
            if token in QUERY_STOPWORDS or _is_weak_standalone_token(token):
                continue
            if _lexical_forms(token) & text_forms:
                continue
            if token not in seen:
                missing.append(token)
                seen.add(token)
    return RequiredTermSignals(full_phrase_match=full_phrase_match, missing_required_terms=missing)


def section_title_startswith_required_term(required_terms: list[str], meta: dict[str, Any]) -> bool:
    section_values: list[str] = []
    for key in ("section_title", "inferred_section_title", "parent_heading"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            section_values.append(value)
    for section in section_values:
        section_norm = normalize_for_required_term(section)
        section_tokens = tokenize_required_text(section)
        for term in required_terms:
            term_norm = normalize_for_required_term(term)
            term_tokens = tokenize_required_text(term)
            if term_norm and (section_norm == term_norm or section_norm.startswith(f"{term_norm} ")):
                return True
            if section_tokens[: len(term_tokens)] and _phrase_window_matches(tuple(term_tokens), section_tokens[: len(term_tokens)]):
                return True
    return False


def concept_boundary_penalty(required_terms: list[str], text_value: str, *, chunk_type: str) -> float:
    if chunk_type != "schema_or_plan" or not required_terms:
        return 0.0
    text_tokens = tokenize_required_text(text_value)
    if not text_tokens:
        return 0.0

    repeated_phrase = False
    for term in required_terms:
        term_tokens = tuple(tokenize_required_text(term))
        if not term_tokens:
            continue
        positions = [
            idx
            for idx in range(0, len(text_tokens) - len(term_tokens) + 1)
            if _phrase_window_matches(term_tokens, text_tokens[idx : idx + len(term_tokens)])
        ]
        if len(positions) >= 2 and positions[-1] - positions[0] >= len(term_tokens) + 4:
            repeated_phrase = True
            break

    marker_count = len(re.findall(r"(?<![A-Za-zА-Яа-яЁё])(?:[IVXLCDM]+|[1-9]\d*)\s*[.)]", text_value or ""))
    incomplete_schema = marker_count < 3
    penalty = 0.0
    if repeated_phrase:
        penalty += settings.retrieval_concept_boundary_penalty_value
    if incomplete_schema:
        penalty += min(0.06, settings.retrieval_concept_boundary_penalty_value)
    return min(settings.retrieval_concept_boundary_penalty_value + 0.06, penalty)


def section_match_score(query: str, query_terms: list[str], meta: dict[str, Any] | None) -> float:
    meta = meta or {}
    section_parts: list[str] = []
    for key in ("section_title", "inferred_section_title", "parent_heading", "heading", "nearest_heading"):
        value = meta.get(key)
        if isinstance(value, str) and _is_valid_section_component(value):
            section_parts.append(_normalize_section_component(value))
    for key in ("section_path", "heading_path", "nearest_headings"):
        value = meta.get(key)
        if isinstance(value, list):
            section_parts.extend(_normalize_section_component(str(item)) for item in value if item and _is_valid_section_component(str(item)))
    section_text = " ".join(section_parts)
    if not section_text:
        return 0.0
    score = max(
        lexical_overlap(query_terms, section_text),
        phrase_match_score(query, section_text),
        weighted_phrase_term_score(query, section_text),
    )
    if query_terms and section_text:
        section_forms = set(_matching_tokens(section_text))
        matched_terms = sum(1 for term in query_terms if _lexical_forms(term) & section_forms)
        if matched_terms:
            score = max(score, min(1.0, 0.35 + 0.2 * matched_terms))
    return _clamp_score(score)


def _normalize_section_component(value: str) -> str:
    details = infer_section_title_details(value)
    title = details.get("section_title")
    return str(title or re.sub(r"\s+", " ", value.strip()))


def _is_valid_section_component(value: str) -> bool:
    details = infer_section_title_details(value)
    return bool(details.get("section_title"))


def _metadata_with_inferred_section(row: RetrievalRow) -> dict[str, Any]:
    meta = dict(row.meta or {})
    section_details = infer_section_title_details(row.text or row.snippet or "", meta=meta)
    existing_section_title = meta.get("section_title")
    if not isinstance(existing_section_title, str) or not _is_valid_section_component(existing_section_title):
        existing_section_title = None
    inferred_section_title = section_details.get("section_title")
    section_title = existing_section_title or inferred_section_title
    section_path = section_details.get("section_path") or []
    meta["section_title"] = section_title
    meta["inferred_section_title"] = inferred_section_title
    meta["parent_heading"] = section_title
    meta["section_title_reason"] = (
        "meta:section_title"
        if existing_section_title
        else section_details.get("section_title_reason")
    )
    meta["section_path"] = section_path
    chunk_source = str(meta.get("search_text") or row.text or row.snippet or "")
    chunk_details = detect_chunk_type_details(chunk_source, meta=meta, page=row.page)
    meta["chunk_type"] = meta.get("chunk_type") or chunk_details.get("chunk_type", "unknown")
    meta["chunk_type_reason"] = meta.get("chunk_type_reason") or chunk_details.get("chunk_type_reason")
    meta["is_navigation_index"] = meta["chunk_type"] == "navigation_index"
    return meta


def _candidate_subject_score(query_analysis: dict[str, Any], row: RetrievalRow) -> float:
    if not math.isclose(row.subject_score, 0.5, abs_tol=1e-9):
        return _clamp_score(row.subject_score)
    meta = row.meta or {}
    subject = meta.get("subject")
    if not subject:
        return 0.5
    return profile_subject_score(query_analysis, {"subject": subject})


def _candidate_quality(row: RetrievalRow) -> dict[str, Any]:
    flags = text_quality_flags(row.text or row.snippet or "", page=row.page, meta=row.meta)
    meta = row.meta or {}
    for key in (
        "is_toc",
        "is_index",
        "is_bibliography",
        "is_caption",
        "is_fragmented",
        "is_too_short",
        "starts_mid_word",
        "low_text_quality",
    ):
        if meta.get(key) is True:
            flags[key] = True
    if "quality_score" in meta:
        try:
            flags["quality_score"] = min(float(flags.get("quality_score", 1.0)), float(meta["quality_score"]))
        except (TypeError, ValueError):
            pass
    chunk_details = detect_chunk_type_details(row.text or row.snippet or "", meta=meta, page=row.page)
    flags["chunk_type"] = meta.get("chunk_type") or chunk_details.get("chunk_type") or "unknown"
    flags["chunk_type_reason"] = meta.get("chunk_type_reason") or chunk_details.get("chunk_type_reason")
    flags["is_navigation_index"] = bool(meta.get("is_navigation_index") is True or flags["chunk_type"] == "navigation_index")
    if flags.get("low_text_quality"):
        flags["quality_score"] = min(float(flags.get("quality_score", 1.0)), 0.25)
    return flags


def _penalties_applied(
    *,
    is_toc: bool,
    low_text_quality: bool,
    wrong_entity_modifier: bool,
    subject_confidence: float,
    subject_score: float,
    required_exact_phrase_missing: bool,
    exercise_penalty_applied: bool = False,
    test_question_penalty_applied: bool = False,
    term_penalty_applied: bool = False,
) -> list[str]:
    penalties: list[str] = []
    if is_toc:
        penalties.append("toc_penalty")
    if low_text_quality:
        penalties.append("low_text_quality_penalty")
    if wrong_entity_modifier:
        penalties.append("wrong_entity_modifier_penalty")
    if required_exact_phrase_missing:
        penalties.append("missing_required_modifier_penalty")
    if exercise_penalty_applied:
        penalties.append("exercise_intent_penalty")
    if test_question_penalty_applied:
        penalties.append("test_question_intent_penalty")
    if term_penalty_applied:
        penalties.append("required_term_penalty")
    if subject_confidence >= settings.retrieval_subject_confidence_threshold and subject_score <= settings.retrieval_subject_mismatch_score:
        penalties.append("subject_mismatch_penalty")
    return penalties


def _is_toc_fragment(row: RetrievalRow) -> bool:
    if (row.meta or {}).get("is_toc") is True:
        return True
    return is_toc_text(row.text or row.snippet or "", page=row.page)


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


def document_level_scores(dense_docs: list[str], lexical_docs: list[str]) -> dict[str, float]:
    dense_scores = reciprocal_rank_scores(dense_docs)
    lexical_scores = reciprocal_rank_scores(lexical_docs)
    doc_ids = set(dense_scores) | set(lexical_scores)
    combined: dict[str, float] = {}
    for doc_id in doc_ids:
        dense_part = dense_scores.get(doc_id, 0.0)
        lexical_part = lexical_scores.get(doc_id, 0.0)
        score = 0.3 * dense_part + 0.7 * lexical_part
        if lexical_part <= 0.0:
            score = min(score, 0.15)
        combined[doc_id] = _clamp_score(score)
    return dict(sorted(combined.items(), key=lambda item: item[1], reverse=True))


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
    values = [float(score) for score in raw_scores[:count]]
    if not values:
        return [None] * count
    if len(values) == 1:
        normalized: list[float | None] = [0.5]
    else:
        min_score = min(values)
        max_score = max(values)
        if math.isclose(max_score, min_score, rel_tol=1e-9, abs_tol=1e-9):
            normalized = [0.5 for _ in values]
        else:
            normalized = [(value - min_score) / (max_score - min_score) for value in values]
    if len(normalized) < count:
        normalized.extend([None] * (count - len(normalized)))
    return normalized[:count]


def _final_score(
    *,
    dense_score: float,
    lexical_score: float,
    lexical_overlap_score: float,
    phrase_score: float,
    section_score: float,
    anchor_score: float,
    topic_score: float,
    subject_score: float,
    subject_confidence: float,
    has_anchor_phrases: bool,
    has_topic_units: bool,
    required_exact_phrase_missing: bool,
    wrong_entity_modifier: bool,
    is_toc: bool,
    low_text_quality: bool,
    quality_score: float,
    rerank_score: float | None,
    rrf_score: float,
    document_score: float,
) -> float:
    lexical_component = max(lexical_score, lexical_overlap_score, phrase_score, topic_score * 0.8)
    phrase_component = _clamp_score(phrase_score)
    components: list[tuple[float, float]] = [
        (settings.retrieval_dense_weight, dense_score),
        (settings.retrieval_lexical_weight, lexical_component),
        (settings.retrieval_phrase_weight, phrase_component),
        (settings.retrieval_section_weight, section_score),
        (settings.retrieval_rrf_weight, rrf_score),
        (settings.retrieval_document_weight, document_score),
        (settings.retrieval_subject_weight, subject_score),
        (settings.retrieval_quality_weight, quality_score),
    ]
    if rerank_score is not None:
        components.append((settings.retrieval_rerank_weight, rerank_score))
    total_weight = sum(weight for weight, _ in components if weight > 0)
    if total_weight <= 0:
        return 0.0
    score = _clamp_score(sum(weight * score for weight, score in components) / total_weight)
    rerank_component = rerank_score or 0.0

    if lexical_component <= 0.0 and phrase_component <= 0.0:
        if dense_score < settings.retrieval_extreme_dense_score and rerank_component < settings.retrieval_extreme_rerank_score:
            score = min(score * 0.25, 0.24)
        else:
            score = min(score * 0.6, 0.5)

    if lexical_component <= 0.0 and phrase_component <= 0.0 and document_score < settings.retrieval_document_gate_min_score:
        score = min(score, 0.18)

    if has_anchor_phrases and has_topic_units and anchor_score > 0 and topic_score <= 0.0:
        score = min(score, 0.18)

    if required_exact_phrase_missing:
        score *= 0.5

    if wrong_entity_modifier:
        score *= 0.25

    if subject_confidence >= settings.retrieval_subject_confidence_threshold and subject_score <= settings.retrieval_subject_mismatch_score:
        score *= settings.retrieval_subject_mismatch_penalty

    if lexical_overlap_score <= 0.0 and phrase_component <= 0.0:
        score *= 0.5

    if is_toc:
        score = min(score * settings.retrieval_toc_penalty, 0.45)

    if low_text_quality:
        score *= settings.retrieval_low_quality_penalty

    return _clamp_score(score)


def _apply_intent_scoring(
    score: float,
    *,
    query_type: str,
    chunk_type: str,
    schema_has_query_phrase: bool,
    has_required_terms: bool = False,
    required_term_score: float = 0.0,
    missing_required_terms: list[str] | None = None,
    full_phrase_match: bool = False,
    section_title_startswith_required_term: bool = False,
    quality_score: float = 1.0,
    is_too_short: bool = False,
    low_text_quality: bool = False,
    boundary_penalty_value: float = 0.0,
) -> tuple[float, bool, bool, bool, bool, bool, bool, bool, bool, bool, str, dict[str, float]]:
    normalized_query_type = query_type or "unknown"
    normalized_chunk_type = chunk_type or "unknown"
    intent_boost = False
    exercise_penalty = False
    test_question_penalty = False
    schema_boost = False
    term_penalty = False
    full_term_boost = False
    concept_boost = False
    exercise_demoted_for_concept = False
    schema_or_rule_boost = False
    reasons: list[str] = []
    score_before_boosts = _clamp_score(score)
    concept_full_phrase_boost_value = 0.0
    section_title_boost_value = 0.0
    schema_or_rule_boost_value = 0.0
    quality_penalty_value = 0.0
    exercise_penalty_value = 0.0

    reference_chunk_types = {"schema_or_plan", "definition", "rule", "explanatory"}
    required_term_ok = (not has_required_terms) or required_term_score >= 0.8

    if normalized_query_type == "concept_lookup":
        boost_scale = max(0.0, 1.0 - score_before_boosts) * 1.2
        if full_phrase_match:
            concept_full_phrase_boost_value = settings.retrieval_concept_full_phrase_boost_value * boost_scale
            reasons.append("concept_full_phrase_boost")
        if full_phrase_match and section_title_startswith_required_term:
            section_title_boost_value = settings.retrieval_concept_section_title_boost_value * boost_scale
            reasons.append("section_title_prefix_boost")
        if full_phrase_match and normalized_chunk_type in reference_chunk_types:
            schema_or_rule_boost_value = settings.retrieval_concept_schema_rule_boost_value * boost_scale
            intent_boost = True
            concept_boost = True
            full_term_boost = True
            schema_or_rule_boost = True
            reasons.append("concept_full_phrase_reference_boost")
            if normalized_chunk_type == "schema_or_plan":
                schema_boost = True
        elif required_term_score < 0.8 and has_required_terms:
            term_penalty = True
            reasons.append(f"missing_required_terms:{','.join(missing_required_terms or [])}" if missing_required_terms else "missing_required_terms")

        if normalized_chunk_type == "exercise":
            exercise_penalty_value = settings.retrieval_concept_exercise_penalty
            exercise_penalty = True
            exercise_demoted_for_concept = True
            reasons.append("exercise_demoted_for_concept_lookup")
        elif normalized_chunk_type == "test_question":
            exercise_penalty_value = settings.retrieval_concept_test_question_penalty
            test_question_penalty = True
            exercise_demoted_for_concept = True
            reasons.append("test_question_demoted_for_concept_lookup")

        if quality_score < 1.0:
            quality_penalty_value += max(0.0, 1.0 - _clamp_score(quality_score)) * 0.07
        if is_too_short:
            quality_penalty_value += 0.03
        if low_text_quality:
            quality_penalty_value += 0.02
        quality_penalty_value = min(settings.retrieval_concept_quality_penalty_max, quality_penalty_value)
        term_penalty_value = (
            settings.retrieval_concept_missing_term_penalty_value
            if term_penalty
            else 0.0
        )

        score_after_boosts_before_clamp = (
            score_before_boosts
            + concept_full_phrase_boost_value
            + section_title_boost_value
            + schema_or_rule_boost_value
            - quality_penalty_value
            - term_penalty_value
            - boundary_penalty_value
            - exercise_penalty_value
        )
        final_score = max(0.0, min(0.99, score_after_boosts_before_clamp))
        if full_phrase_match and normalized_chunk_type in reference_chunk_types:
            concept_boost = True
            schema_or_rule_boost = True
            intent_boost = True
        score_debug = {
            "score_before_boosts": score_before_boosts,
            "concept_full_phrase_boost_value": concept_full_phrase_boost_value,
            "section_title_boost_value": section_title_boost_value,
            "schema_or_rule_boost_value": schema_or_rule_boost_value,
            "quality_penalty_value": quality_penalty_value,
            "boundary_penalty_value": boundary_penalty_value,
            "exercise_penalty_value": exercise_penalty_value,
            "score_after_boosts_before_clamp": score_after_boosts_before_clamp,
        }
        return (
            _clamp_score(final_score),
            intent_boost,
            exercise_penalty,
            test_question_penalty,
            schema_boost,
            term_penalty,
            full_term_boost,
            concept_boost,
            exercise_demoted_for_concept,
            schema_or_rule_boost,
            ";".join(reasons) if reasons else "base_score",
            score_debug,
        )

    required_reference_boosted = False
    if normalized_query_type == "concept_lookup" and has_required_terms:
        if full_phrase_match and normalized_chunk_type in reference_chunk_types:
            score *= settings.retrieval_intent_schema_boost
            intent_boost = True
            concept_boost = True
            full_term_boost = True
            schema_or_rule_boost = True
            required_reference_boosted = True
            reasons.append("concept_full_phrase_reference_boost")
            if normalized_chunk_type == "schema_or_plan":
                schema_boost = True
        elif required_term_score < 0.8:
            score *= 0.45
            term_penalty = True
            missing_label = ",".join(missing_required_terms or [])
            reasons.append(f"missing_required_terms:{missing_label}" if missing_label else "missing_required_terms")

    if normalized_query_type in {"concept_lookup", "rule_lookup", "explanation", "definition"}:
        if normalized_chunk_type == "schema_or_plan":
            if not required_reference_boosted and required_term_ok and schema_has_query_phrase:
                score *= settings.retrieval_intent_schema_boost
                intent_boost = True
                schema_boost = True
                if normalized_query_type == "concept_lookup":
                    concept_boost = True
                    schema_or_rule_boost = True
                    reasons.append("concept_schema_boost")
        elif normalized_chunk_type in {"definition", "rule", "explanatory"}:
            if not required_reference_boosted and required_term_ok:
                score *= settings.retrieval_intent_reference_boost
                intent_boost = True
                if normalized_query_type == "concept_lookup":
                    concept_boost = True
                    schema_or_rule_boost = True
                    reasons.append("concept_reference_boost")
        elif normalized_chunk_type == "exercise":
            if normalized_query_type == "concept_lookup":
                score *= settings.retrieval_concept_exercise_penalty
                exercise_demoted_for_concept = True
                reasons.append("exercise_demoted_for_concept_lookup")
            else:
                score *= settings.retrieval_exercise_penalty
            exercise_penalty = True
        elif normalized_chunk_type == "test_question":
            if normalized_query_type == "concept_lookup":
                score *= settings.retrieval_concept_test_question_penalty
                exercise_demoted_for_concept = True
                reasons.append("test_question_demoted_for_concept_lookup")
            else:
                score *= settings.retrieval_test_question_penalty
            test_question_penalty = True
    elif normalized_query_type in {"exercise_lookup", "problem_solving"} and normalized_chunk_type in {"exercise", "test_question"}:
        score *= settings.retrieval_exercise_query_boost
        intent_boost = True
        reasons.append("exercise_lookup_boost")

    return (
        _clamp_score(score),
        intent_boost,
        exercise_penalty,
        test_question_penalty,
        schema_boost,
        term_penalty,
        full_term_boost,
        concept_boost,
        exercise_demoted_for_concept,
        schema_or_rule_boost,
        ";".join(reasons) if reasons else "base_score",
        {
            "score_before_boosts": score_before_boosts,
            "concept_full_phrase_boost_value": 0.0,
            "section_title_boost_value": 0.0,
            "schema_or_rule_boost_value": 0.0,
            "quality_penalty_value": 0.0,
            "boundary_penalty_value": 0.0,
            "exercise_penalty_value": 0.0,
            "score_after_boosts_before_clamp": score,
        },
    )


def score_retrieval_candidates(
    query: str,
    candidates: list[RetrievalRow],
    *,
    rerank_raw_scores: list[float] | None = None,
    query_analysis: dict[str, Any] | None = None,
    apply_noise_filter: bool = True,
) -> tuple[list[RetrievalRow], list[dict[str, Any]]]:
    query_analysis = query_analysis or analyze_query(query)
    query_terms = _query_terms_for_scoring(query)
    query_phrases = _query_phrases_for_scoring(query)
    anchor_phrases = _anchor_phrases_for_query(query)
    required_exact_phrases = _required_exact_phrases_for_query(query)
    required_terms = _required_terms_for_scoring(query, query_analysis)
    topic_units = _topic_units_for_query(query)
    lexical_scores = _bm25_scores(query_terms, candidates) if query_terms else {}
    rerank_scores = _normalize_rerank_scores(rerank_raw_scores, len(candidates))
    scored: list[RetrievalRow] = []
    rejected: list[dict[str, Any]] = []

    for idx, row in enumerate(candidates):
        meta = _metadata_with_inferred_section(row)
        row_for_scoring = replace(row, meta=meta)
        dense_score = _clamp_score(row.dense_score or row.score)
        lexical_score = _clamp_score(max(row.lexical_score, lexical_scores.get(row.fragment_id, 0.0)))
        section_context = " ".join(
            str(value)
            for value in (
                meta.get("section_title"),
                meta.get("inferred_section_title"),
                " ".join(str(item) for item in meta.get("section_path") or [] if item),
            )
            if value
        )
        # Enriched search_text is useful for recall, but it contains document-wide
        # keywords and must not be treated as evidence during final scoring.
        lexical_text = f"{row.title or ''} {row.source_uri or ''} {section_context} {row.text or ''}"
        overlap = lexical_overlap(query_terms, lexical_text)
        raw_phrase_score = phrase_match_score(query, lexical_text)
        anchor_score = anchor_phrase_score(query, lexical_text)
        topic_score = topical_match_score(query, lexical_text)
        weighted_match_score = weighted_phrase_term_score(query, lexical_text)
        section_score = section_match_score(query, query_terms, meta)
        required_term_text = _candidate_required_term_text(row, meta)
        required_term_match = required_term_match_score(required_terms, required_term_text)
        term_signals = required_term_signals(required_terms, required_term_text, required_term_match)
        phrase_score_before_penalty = _clamp_score(max(raw_phrase_score, weighted_match_score, section_score * 0.85))
        phrase_match = _phrase_modifier_match(required_exact_phrases, lexical_text, phrase_score_before_penalty)
        phrase_score = phrase_match.phrase_score_after_penalty
        subject_score = _candidate_subject_score(query_analysis, row_for_scoring)
        quality = _candidate_quality(row_for_scoring)
        chunk_type = str(quality.get("chunk_type") or meta.get("chunk_type") or "unknown")
        is_navigation_index = bool(quality.get("is_navigation_index") or chunk_type == "navigation_index")
        if chunk_type in {"exercise", "test_question"} and quality.get("low_text_quality_reason") == "toc":
            quality["is_toc"] = False
            quality["low_text_quality"] = False
            quality["low_text_quality_reason"] = None
            quality["quality_score"] = max(float(quality.get("quality_score") or 0.0), 0.75)
        is_toc = bool(quality.get("is_toc") or _is_toc_fragment(row_for_scoring))
        if chunk_type in {"exercise", "test_question"}:
            is_toc = False
        document_score = _effective_document_score(
            row.document_score,
            lexical_overlap_score=overlap,
            phrase_score=phrase_score,
            anchor_score=anchor_score,
            topic_score=topic_score,
            has_query_phrases=bool(query_phrases),
            has_anchor_phrases=bool(anchor_phrases),
            has_topic_units=bool(topic_units),
        )
        rerank_score = rerank_scores[idx]
        final_score = _final_score(
            dense_score=dense_score,
            lexical_score=lexical_score,
            lexical_overlap_score=overlap,
            phrase_score=phrase_score,
            section_score=section_score,
            anchor_score=anchor_score,
            topic_score=topic_score,
            subject_score=subject_score,
            subject_confidence=float(query_analysis.get("subject_confidence") or 0.0),
            has_anchor_phrases=bool(anchor_phrases),
            has_topic_units=bool(topic_units),
            required_exact_phrase_missing=bool(phrase_match.missing_required_modifiers),
            wrong_entity_modifier=phrase_match.wrong_entity_modifier,
            is_toc=is_toc,
            low_text_quality=bool(quality.get("low_text_quality")),
            quality_score=float(quality.get("quality_score") or 1.0),
            rerank_score=rerank_score,
            rrf_score=_clamp_score(row.rrf_score),
            document_score=document_score,
        )
        schema_has_query_phrase = chunk_type == "schema_or_plan" and (raw_phrase_score > 0 or section_score >= 0.8)
        (
            final_score,
            intent_boost_applied,
            exercise_penalty_applied,
            test_question_penalty_applied,
            schema_boost_applied,
            term_penalty_applied,
            full_term_boost_applied,
            concept_boost_applied,
            exercise_demoted_for_concept_lookup,
            schema_or_rule_boost_applied,
            final_rank_reason,
            score_adjustments,
        ) = _apply_intent_scoring(
            final_score,
            query_type=str(query_analysis.get("query_type") or "unknown"),
            chunk_type=chunk_type,
            schema_has_query_phrase=schema_has_query_phrase or (chunk_type == "schema_or_plan" and required_term_match.score >= 0.8),
            has_required_terms=bool(required_terms),
            required_term_score=required_term_match.score,
            missing_required_terms=term_signals.missing_required_terms,
            full_phrase_match=term_signals.full_phrase_match,
            section_title_startswith_required_term=section_title_startswith_required_term(required_terms, meta),
            quality_score=float(quality.get("quality_score") or 1.0),
            is_too_short=bool(quality.get("is_too_short")),
            low_text_quality=bool(quality.get("low_text_quality")),
            boundary_penalty_value=concept_boundary_penalty(required_terms, row.text or row.snippet or "", chunk_type=chunk_type),
        )
        reason = _noise_rejection_reason(
            query_terms=query_terms,
            dense_score=dense_score,
            lexical_score=lexical_score,
            lexical_overlap_score=overlap,
            phrase_score=phrase_score,
            anchor_score=anchor_score,
            topic_score=topic_score,
            subject_score=subject_score,
            subject_confidence=float(query_analysis.get("subject_confidence") or 0.0),
            has_query_phrases=bool(query_phrases),
            has_anchor_phrases=bool(anchor_phrases),
            has_topic_units=bool(topic_units),
            required_exact_phrase_missing=bool(phrase_match.missing_required_modifiers),
            hard_required_phrase_missing=bool(phrase_match.missing_required_modifiers)
            and any(
                phrase.kind == "abbreviation_phrase"
                or any(any(char.isdigit() for char in token) for token in phrase.tokens)
                for phrase in required_exact_phrases
            ),
            wrong_entity_modifier=phrase_match.wrong_entity_modifier,
            is_toc=is_toc,
            document_score=document_score,
            rerank_score=rerank_score,
            final_score=final_score,
        )
        updated = replace(
            row,
            meta=meta,
            score=final_score,
            dense_score=dense_score,
            lexical_score=lexical_score,
            phrase_score=phrase_score,
            subject_score=subject_score,
            section_score=section_score,
            lexical_overlap=overlap,
            rerank_score=rerank_score,
            final_score=final_score,
            document_score=document_score,
            rejection_reason=reason,
            exact_phrases=phrase_match.exact_phrases,
            matched_phrases=phrase_match.matched_phrases,
            missing_required_modifiers=phrase_match.missing_required_modifiers,
            wrong_entity_modifier=phrase_match.wrong_entity_modifier,
            phrase_score_before_penalty=phrase_match.phrase_score_before_penalty,
            phrase_score_after_penalty=phrase_match.phrase_score_after_penalty,
            is_toc=is_toc,
            toc_penalty_applied=is_toc,
            quality_score=float(quality.get("quality_score") or 1.0),
            is_index=bool(quality.get("is_index")),
            is_bibliography=bool(quality.get("is_bibliography")),
            is_caption=bool(quality.get("is_caption")),
            is_fragmented=bool(quality.get("is_fragmented")),
            is_too_short=bool(quality.get("is_too_short")),
            starts_mid_word=bool(quality.get("starts_mid_word")),
            low_text_quality=bool(quality.get("low_text_quality")),
            low_text_quality_reason=quality.get("low_text_quality_reason"),
            query_type=str(query_analysis.get("query_type") or "unknown"),
            chunk_type=chunk_type,
            chunk_type_reason=quality.get("chunk_type_reason") or meta.get("chunk_type_reason"),
            section_title_reason=meta.get("section_title_reason"),
            inferred_section_title=meta.get("inferred_section_title"),
            required_terms=required_term_match.required_terms,
            required_term_score=required_term_match.score,
            required_term_match_type=required_term_match.match_type,
            full_phrase_match=term_signals.full_phrase_match,
            missing_required_terms=term_signals.missing_required_terms,
            is_navigation_index=is_navigation_index,
            intent_boost_applied=intent_boost_applied,
            exercise_penalty_applied=exercise_penalty_applied,
            test_question_penalty_applied=test_question_penalty_applied,
            schema_boost_applied=schema_boost_applied,
            term_penalty_applied=term_penalty_applied,
            full_term_boost_applied=full_term_boost_applied,
            concept_boost_applied=concept_boost_applied,
            exercise_demoted_for_concept_lookup=exercise_demoted_for_concept_lookup,
            schema_or_rule_boost_applied=schema_or_rule_boost_applied,
            final_rank_reason=final_rank_reason,
            score_before_boosts=float(score_adjustments["score_before_boosts"]),
            concept_full_phrase_boost_value=float(score_adjustments["concept_full_phrase_boost_value"]),
            section_title_boost_value=float(score_adjustments["section_title_boost_value"]),
            schema_or_rule_boost_value=float(score_adjustments["schema_or_rule_boost_value"]),
            quality_penalty_value=float(score_adjustments["quality_penalty_value"]),
            boundary_penalty_value=float(score_adjustments["boundary_penalty_value"]),
            exercise_penalty_value=float(score_adjustments["exercise_penalty_value"]),
            score_after_boosts_before_clamp=float(score_adjustments["score_after_boosts_before_clamp"]),
            penalties_applied=_penalties_applied(
                is_toc=is_toc,
                low_text_quality=bool(quality.get("low_text_quality")),
                wrong_entity_modifier=phrase_match.wrong_entity_modifier,
                subject_confidence=float(query_analysis.get("subject_confidence") or 0.0),
                subject_score=subject_score,
                required_exact_phrase_missing=bool(phrase_match.missing_required_modifiers),
                exercise_penalty_applied=exercise_penalty_applied,
                test_question_penalty_applied=test_question_penalty_applied,
                term_penalty_applied=term_penalty_applied,
            ),
        )
        if apply_noise_filter and reason:
            rejected.append(_debug_hit(updated))
            continue
        scored.append(updated)

    return sorted(scored, key=lambda row: row.final_score, reverse=True), rejected


def _effective_document_score(
    document_score: float,
    *,
    lexical_overlap_score: float,
    phrase_score: float,
    anchor_score: float,
    topic_score: float,
    has_query_phrases: bool,
    has_anchor_phrases: bool,
    has_topic_units: bool,
) -> float:
    score = _clamp_score(document_score)
    if lexical_overlap_score <= 0.0 and phrase_score <= 0.0:
        return min(score, 0.15)
    if has_anchor_phrases and has_topic_units and anchor_score > 0 and topic_score <= 0.0:
        return min(score, 0.35)
    if (
        has_query_phrases
        and phrase_score <= 0.0
        and lexical_overlap_score < 0.67
        and not (has_anchor_phrases and has_topic_units and topic_score > 0)
    ):
        return min(score, 0.15)
    if lexical_overlap_score < 0.5 and phrase_score <= 0.0:
        return min(score, 0.35)
    return score


def _noise_rejection_reason(
    *,
    query_terms: list[str],
    dense_score: float,
    lexical_score: float,
    lexical_overlap_score: float,
    phrase_score: float,
    anchor_score: float,
    topic_score: float,
    subject_score: float,
    subject_confidence: float,
    has_query_phrases: bool,
    has_anchor_phrases: bool,
    has_topic_units: bool,
    required_exact_phrase_missing: bool,
    hard_required_phrase_missing: bool,
    wrong_entity_modifier: bool,
    is_toc: bool,
    document_score: float,
    rerank_score: float | None,
    final_score: float,
) -> str | None:
    if not query_terms:
        return None
    lexical_component = max(lexical_score, lexical_overlap_score, phrase_score)
    rerank_component = rerank_score or 0.0
    if subject_confidence >= settings.retrieval_subject_confidence_threshold and subject_score <= settings.retrieval_subject_mismatch_score:
        return "subject_mismatch"
    if wrong_entity_modifier and final_score < settings.default_min_score:
        return "wrong_entity_modifier"
    if is_toc and final_score < settings.default_min_score:
        return "table_of_contents"
    if hard_required_phrase_missing and phrase_score <= 0.2:
        return "missing_required_modifier"
    if (
        document_score < settings.retrieval_document_gate_min_score
        and lexical_component <= 0.0
        and rerank_component < settings.retrieval_extreme_rerank_score
    ):
        return "low_document_score"
    if (
        required_exact_phrase_missing
        and phrase_score <= 0.2
        and topic_score <= 0.0
        and dense_score < settings.retrieval_extreme_dense_score
        and rerank_component < settings.retrieval_extreme_rerank_score
    ):
        return "missing_required_modifier"
    if (
        has_query_phrases
        and phrase_score <= 0.0
        and lexical_overlap_score < 0.67
        and not (has_anchor_phrases and has_topic_units and topic_score > 0)
        and dense_score < settings.retrieval_extreme_dense_score
        and rerank_component < settings.retrieval_extreme_rerank_score
    ):
        return "low_phrase_match"
    if (
        len(query_terms) >= 3
        and lexical_overlap_score < 0.34
        and document_score < settings.retrieval_document_gate_min_score
        and dense_score < settings.retrieval_extreme_dense_score
        and rerank_component < settings.retrieval_extreme_rerank_score
    ):
        return "low_lexical_overlap"
    if (
        lexical_component <= 0.0
        and dense_score < settings.retrieval_extreme_dense_score
    ):
        return "low_lexical_overlap"
    if (
        lexical_overlap_score <= settings.retrieval_min_lexical_overlap
        and lexical_score <= 0.02
        and phrase_score <= 0.02
        and dense_score < settings.retrieval_noise_dense_floor
    ):
        return "low_lexical_overlap"
    if lexical_overlap_score == 0 and lexical_score == 0 and phrase_score == 0 and dense_score < settings.retrieval_noise_strict_dense_floor:
        return "low_lexical_overlap"
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
        anchor_signal = bool(
            current.full_phrase_match
            or current.required_term_score >= 0.8
            or (current.phrase_score >= 0.9 and current.lexical_overlap >= 0.67)
        )
        semantic_anchor = anchor_signal and (
            current.chunk_type not in {"exercise", "test_question", "navigation_index", "toc"}
            or current.expanded_from_neighbors
        )
        if semantic_anchor:
            kept.append(current)
            continue
        if top_score > 0 and current.final_score < top_score * settings.retrieval_adaptive_relative_floor:
            continue
        if (
            prev.final_score - current.final_score >= settings.retrieval_adaptive_gap
            and current.final_score < top_score * 0.75
        ):
            continue
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


def _same_context_scope(hit: RetrievalRow, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    hit_meta = hit.meta or {}
    hit_section = hit_meta.get("section_path") or hit_meta.get("heading_path") or []
    if isinstance(hit_section, list) and hit_section:
        scoped = [
            row
            for row in rows
            if isinstance(row.get("meta"), dict)
            and ((row["meta"].get("section_path") or row["meta"].get("heading_path") or []) == hit_section)
        ]
        if len(scoped) >= 2:
            return scoped
    if hit.page is not None:
        scoped = [row for row in rows if row.get("page") == hit.page]
        if len(scoped) >= 2:
            return scoped
    return []


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
        "section_title": (hit.meta or {}).get("section_title"),
        "inferred_section_title": hit.inferred_section_title or (hit.meta or {}).get("inferred_section_title"),
        "section_path": (hit.meta or {}).get("section_path") or (hit.meta or {}).get("heading_path") or [],
        "dense_score": round(hit.dense_score, 4),
        "lexical_score": round(hit.lexical_score, 4),
        "phrase_score": round(hit.phrase_score, 4),
        "subject_score": round(hit.subject_score, 4),
        "section_score": round(hit.section_score, 4),
        "rerank_score": round(hit.rerank_score, 4) if hit.rerank_score is not None else 0.0,
        "final_score": round(hit.final_score, 4),
        "score": round(hit.score, 4),
        "lexical_overlap": round(hit.lexical_overlap, 4),
        "document_score": round(hit.document_score, 4),
        "rrf_score": round(hit.rrf_score, 4),
        "exact_phrases": hit.exact_phrases,
        "matched_phrases": hit.matched_phrases,
        "missing_required_modifiers": hit.missing_required_modifiers,
        "wrong_entity_modifier": hit.wrong_entity_modifier,
        "phrase_score_before_penalty": round(hit.phrase_score_before_penalty, 4),
        "phrase_score_after_penalty": round(hit.phrase_score_after_penalty, 4),
        "is_toc": hit.is_toc,
        "toc_filtered": hit.toc_filtered,
        "toc_penalty_applied": hit.toc_penalty_applied,
        "quality_score": round(hit.quality_score, 4),
        "is_index": hit.is_index,
        "is_bibliography": hit.is_bibliography,
        "is_caption": hit.is_caption,
        "is_fragmented": hit.is_fragmented,
        "is_too_short": hit.is_too_short,
        "starts_mid_word": hit.starts_mid_word,
        "low_text_quality": hit.low_text_quality,
        "low_quality_filtered": hit.low_quality_filtered,
        "low_text_quality_reason": hit.low_text_quality_reason,
        "query_type": hit.query_type,
        "chunk_type": hit.chunk_type,
        "chunk_type_reason": hit.chunk_type_reason,
        "section_title_reason": hit.section_title_reason,
        "required_terms": hit.required_terms,
        "required_term_score": round(hit.required_term_score, 4),
        "required_term_match_type": hit.required_term_match_type,
        "full_phrase_match": hit.full_phrase_match,
        "missing_required_terms": hit.missing_required_terms,
        "is_navigation_index": hit.is_navigation_index,
        "navigation_filtered": hit.navigation_filtered,
        "intent_boost_applied": hit.intent_boost_applied,
        "exercise_penalty_applied": hit.exercise_penalty_applied,
        "test_question_penalty_applied": hit.test_question_penalty_applied,
        "schema_boost_applied": hit.schema_boost_applied,
        "term_penalty_applied": hit.term_penalty_applied,
        "full_term_boost_applied": hit.full_term_boost_applied,
        "concept_boost_applied": hit.concept_boost_applied,
        "exercise_demoted_for_concept_lookup": hit.exercise_demoted_for_concept_lookup,
        "schema_or_rule_boost_applied": hit.schema_or_rule_boost_applied,
        "final_rank_reason": hit.final_rank_reason,
        "score_before_boosts": round(hit.score_before_boosts, 4),
        "concept_full_phrase_boost_value": round(hit.concept_full_phrase_boost_value, 4),
        "section_title_boost_value": round(hit.section_title_boost_value, 4),
        "schema_or_rule_boost_value": round(hit.schema_or_rule_boost_value, 4),
        "quality_penalty_value": round(hit.quality_penalty_value, 4),
        "boundary_penalty_value": round(hit.boundary_penalty_value, 4),
        "exercise_penalty_value": round(hit.exercise_penalty_value, 4),
        "score_after_boosts_before_clamp": round(hit.score_after_boosts_before_clamp, 4),
        "expanded_from_neighbors": hit.expanded_from_neighbors,
        "penalties_applied": hit.penalties_applied,
        "rejection_reason": hit.rejection_reason,
    }


def _debug_document_profile(
    source_uri: str,
    profile: dict[str, Any],
    score: float,
    subject_score: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "source_uri": source_uri,
        "title": profile.get("title"),
        "subject": profile.get("subject", "unknown"),
        "grade": profile.get("grade"),
        "doc_type": profile.get("doc_type", "unknown"),
        "document_score": round(_clamp_score(score), 4),
        "subject_score": round(_clamp_score(subject_score), 4),
        "reason": reason,
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


def expand_query(
    query: str,
    *,
    collection: str | None = None,
    source_uris: list[str] | None = None,
    query_analysis: dict[str, Any] | None = None,
) -> list[str]:
    if not query:
        return []

    queries = [query]
    if not settings.query_expansion_enabled:
        return queries

    query_analysis = query_analysis or analyze_query(query)
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
    extra_terms.extend(subject_expansions_for_query(query_analysis))

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


def select_final_hits(
    hits: list[RetrievalRow],
    *,
    top_k: int,
    min_score: float,
    include_toc: bool = False,
    include_low_quality: bool = False,
    include_navigation: bool = False,
) -> list[RetrievalRow]:
    normalized_hits = [
        hit if hit.final_score else replace(hit, final_score=hit.score, score=hit.score)
        for hit in hits
    ]
    eligible = [
        hit
        for hit in normalized_hits
        if hit.final_score >= _clamp_score(min_score)
        and (include_toc or not hit.is_toc)
        and (include_low_quality or not hit.low_text_quality)
        and (include_navigation or not hit.is_navigation_index)
    ]
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
    if not query_terms or not rows:
        return {row.fragment_id: 0.0 for row in rows}

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
    normalized: dict[str, float] = {}
    for row in rows:
        coverage = _token_coverage(query_terms, tokenized[row.fragment_id])
        normalized[row.fragment_id] = _clamp_score((scores[row.fragment_id] / max_score) * coverage)
    return normalized


def _token_coverage(query_terms: list[str], tokens: list[str]) -> float:
    if not query_terms:
        return 0.0
    haystack_forms = set(tokens)
    if not haystack_forms:
        return 0.0
    matched = 0
    for term in query_terms:
        if _lexical_forms(term) & haystack_forms:
            matched += 1
    return matched / len(query_terms)
