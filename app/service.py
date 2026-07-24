import logging
from pathlib import Path
import re

from app.chunking import split_structured_chunks
from app.config import settings
from app.document_intelligence import build_document_profile, detect_chunk_type_details, infer_section_title_details, is_toc_text, text_quality_flags
from app.parser import RAGAnythingParser
from app.repository import RagRepository
from app.schemas import CanonicalFragment, ParsedElement, QueryResponse, Source, SourceInfo
from app.utils import snippet_from_text, stable_fragment_id

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".txt", ".md", ".png", ".jpg", ".jpeg"}
logger = logging.getLogger("rag_service")


class IngestLimitError(ValueError):
    pass


class RagService:
    def __init__(self, parser: RAGAnythingParser, repository: RagRepository) -> None:
        self.parser = parser
        self.repository = repository

    def ingest(self, input_path: str, collection: str, reindex: bool) -> dict[str, int]:
        root = Path(input_path)
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if len(files) > settings.max_ingest_files:
            raise IngestLimitError(
                f"Too many files: {len(files)}; maximum is {settings.max_ingest_files}"
            )
        max_file_bytes = settings.max_file_size_mb * 1024 * 1024
        oversized = [path for path in files if path.stat().st_size > max_file_bytes]
        if oversized:
            raise IngestLimitError(
                f"File exceeds MAX_FILE_SIZE_MB={settings.max_file_size_mb}: {oversized[0].name}"
            )
        total_bytes = sum(path.stat().st_size for path in files)
        if total_bytes > settings.max_ingest_batch_mb * 1024 * 1024:
            raise IngestLimitError(
                f"Ingest batch exceeds MAX_INGEST_BATCH_MB={settings.max_ingest_batch_mb}"
            )

        indexed_docs = indexed_fragments = indexed_vectors = 0
        fallback_docs = 0
        for file_path in files:
            source_uri = str(file_path.relative_to(root)).replace("\\", "/")
            parsed, parse_mode = self.parser.parse_file_with_mode(source_uri=source_uri, path=file_path, reindex=reindex)
            if settings.coalesce_parsed_elements_enabled:
                parsed = coalesce_parsed_elements(parsed)
            if parse_mode == "fallback":
                fallback_docs += 1

            document_profile = build_document_profile(
                source_uri=source_uri,
                title=file_path.name,
                file_path=file_path,
                parsed_elements=parsed,
                collection=collection,
            )
            embedding_provider = getattr(self.repository, "embeddings", None)
            embedding_fingerprint = getattr(embedding_provider, "model_fingerprint", None)
            if embedding_provider is not None:
                document_profile["summary_embedding"] = embedding_provider.embed(document_profile["profile_text"])
            else:
                document_profile["summary_embedding"] = []
            doc = self.repository.upsert_document(
                source_uri,
                file_path.name,
                collection,
                {
                    **_infer_document_metadata(file_path, root, collection),
                    "path": str(file_path),
                    "parse_mode": parse_mode,
                    "embedding_fingerprint": embedding_fingerprint,
                    "document_profile": document_profile,
                    "subject": document_profile["subject"],
                    "grade": document_profile["grade"],
                    "doc_type": document_profile["doc_type"],
                    "language": document_profile["language"],
                    "keywords": document_profile["keywords"],
                    "section_titles": document_profile["section_titles"],
                },
                reindex,
            )
            document_profile["doc_id"] = str(doc.doc_id)
            doc.meta = {
                **(getattr(doc, "meta", None) or {}),
                "document_profile": document_profile,
                "subject": document_profile["subject"],
                "grade": document_profile["grade"],
                "doc_type": document_profile["doc_type"],
                "language": document_profile["language"],
            }
            if hasattr(self.repository.db, "flush"):
                self.repository.db.flush()
            indexed_docs += 1

            for elem in parsed:
                structured_chunks = split_structured_chunks(elem.content)
                if not structured_chunks:
                    continue

                for chunk_idx, chunk in enumerate(structured_chunks):
                    element_sort_index = elem.element_index * 10_000 + chunk_idx
                    fragment_source_key = source_uri if collection == "default" else f"{collection}:{source_uri}"
                    fragment_id = stable_fragment_id(fragment_source_key, element_sort_index, chunk.text)
                    meta = dict(elem.meta)
                    inferred_heading_path = chunk.heading_path or _heading_path_from_meta(meta)
                    section_details = infer_section_title_details(chunk.text, meta=meta)
                    inferred_section_title = section_details.get("section_title")
                    if not inferred_heading_path and inferred_section_title:
                        inferred_heading_path = [str(inferred_section_title)]
                    meta["heading_path"] = inferred_heading_path
                    meta["source_uri"] = source_uri
                    meta["title"] = getattr(doc, "title", file_path.name)
                    meta["collection"] = collection
                    meta["page"] = elem.page
                    meta["original_element_index"] = elem.element_index
                    meta["chunk_index"] = chunk_idx
                    meta["section_path"] = section_details.get("section_path") or inferred_heading_path
                    meta["section_title"] = inferred_heading_path[-1] if inferred_heading_path else inferred_section_title
                    meta["inferred_section_title"] = meta["section_title"]
                    meta["parent_heading"] = inferred_heading_path[-1] if inferred_heading_path else inferred_section_title
                    meta["section_title_reason"] = section_details.get("section_title_reason")
                    meta["subject"] = document_profile["subject"]
                    meta["grade"] = document_profile["grade"]
                    meta["doc_type"] = document_profile["doc_type"]
                    meta["language"] = document_profile["language"]
                    meta["is_toc"] = is_toc_text(chunk.text, page=getattr(chunk, "page", None) or elem.page)
                    chunk_type_details = detect_chunk_type_details(chunk.text, meta=meta, page=elem.page)
                    meta["chunk_type"] = chunk_type_details.get("chunk_type")
                    meta["chunk_type_reason"] = chunk_type_details.get("chunk_type_reason")
                    meta["is_navigation_index"] = meta["chunk_type"] == "navigation_index"
                    quality = text_quality_flags(chunk.text, page=elem.page, meta=meta)
                    meta.update(quality)
                    meta["display_text"] = chunk.text
                    meta["search_text"] = _build_search_text(chunk.text, meta, document_profile)
                    fragment = CanonicalFragment(
                        fragment_id=fragment_id,
                        element_index=element_sort_index,
                        source_uri=source_uri,
                        type=elem.type,
                        page=elem.page,
                        text=chunk.text,
                        snippet=snippet_from_text(chunk.text),
                        meta=meta,
                    )
                    vectors = self.repository.insert_fragment_with_embeddings(doc, fragment)
                    if vectors > 0:
                        indexed_fragments += 1
                        indexed_vectors += vectors

        fallback_ratio = (fallback_docs / indexed_docs) if indexed_docs else 0.0
        logger.info(
            "parser_observability",
            extra={
                "indexed_docs": indexed_docs,
                "fallback_docs": fallback_docs,
                "fallback_ratio": round(fallback_ratio, 4),
            },
        )
        if fallback_ratio > settings.parser_fallback_alert_threshold:
            logger.warning(
                "parser_fallback_ratio_alert",
                extra={
                    "fallback_ratio": round(fallback_ratio, 4),
                    "threshold": settings.parser_fallback_alert_threshold,
                },
            )

        self.repository.db.commit()
        return {
            "indexed_docs": indexed_docs,
            "indexed_fragments": indexed_fragments,
            "indexed_vectors": indexed_vectors,
        }

    def retrieve(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
        return_text: bool,
        include_toc: bool = False,
        include_low_quality: bool = False,
        include_navigation: bool = False,
    ) -> list[dict]:
        rows = self.repository.retrieve(
            query,
            top_k,
            min_score,
            collection,
            source_uris,
            include_toc=include_toc,
            include_low_quality=include_low_quality,
            include_navigation=include_navigation,
        )
        return self._rows_to_hits(rows, return_text=return_text, debug=False)

    def retrieve_with_debug(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
        return_text: bool,
        include_toc: bool = False,
        include_low_quality: bool = False,
        include_navigation: bool = False,
    ) -> tuple[list[dict], dict | None]:
        result = self.repository.retrieve_with_debug(
            query,
            top_k,
            min_score,
            collection,
            source_uris,
            include_toc=include_toc,
            include_low_quality=include_low_quality,
            include_navigation=include_navigation,
            debug=True,
        )
        return self._rows_to_hits(result.hits, return_text=return_text, debug=True), result.debug

    @staticmethod
    def _rows_to_hits(rows, *, return_text: bool, debug: bool) -> list[dict]:
        hits: list[dict] = []
        for r in rows:
            payload = {
                "fragment_id": r.fragment_id,
                "source_uri": r.source_uri,
                "title": r.title,
                "type": r.type,
                "page": r.page,
                "section_title": (r.meta or {}).get("section_title"),
                "section_path": (r.meta or {}).get("section_path") or (r.meta or {}).get("heading_path") or [],
                "snippet": r.text,
                "score": float(r.final_score or r.score),
                "text": r.text if return_text else None,
                "dense_score": float(r.dense_score),
                "lexical_score": float(r.lexical_score),
                "phrase_score": float(r.phrase_score),
                "subject_score": float(r.subject_score),
                "section_score": float(r.section_score),
                "rerank_score": float(r.rerank_score) if r.rerank_score is not None else 0.0,
                "final_score": float(r.final_score or r.score),
                "lexical_overlap": float(r.lexical_overlap),
                "document_score": float(r.document_score),
                "rrf_score": float(r.rrf_score),
                "exact_phrases": list(r.exact_phrases),
                "matched_phrases": list(r.matched_phrases),
                "missing_required_modifiers": list(r.missing_required_modifiers),
                "wrong_entity_modifier": bool(r.wrong_entity_modifier),
                "phrase_score_before_penalty": float(r.phrase_score_before_penalty),
                "phrase_score_after_penalty": float(r.phrase_score_after_penalty),
                "is_toc": bool(r.is_toc),
                "toc_filtered": bool(r.toc_filtered),
                "toc_penalty_applied": bool(r.toc_penalty_applied),
                "quality_score": float(r.quality_score),
                "is_index": bool(r.is_index),
                "is_bibliography": bool(r.is_bibliography),
                "is_caption": bool(r.is_caption),
                "is_fragmented": bool(r.is_fragmented),
                "is_too_short": bool(r.is_too_short),
                "starts_mid_word": bool(r.starts_mid_word),
                "low_text_quality": bool(r.low_text_quality),
                "low_quality_filtered": bool(r.low_quality_filtered),
                "low_text_quality_reason": r.low_text_quality_reason,
                "query_type": r.query_type,
                "chunk_type": r.chunk_type,
                "chunk_type_reason": r.chunk_type_reason,
                "section_title_reason": r.section_title_reason,
                "inferred_section_title": r.inferred_section_title,
                "required_terms": list(r.required_terms),
                "required_term_score": float(r.required_term_score),
                "required_term_match_type": r.required_term_match_type,
                "full_phrase_match": bool(r.full_phrase_match),
                "missing_required_terms": list(r.missing_required_terms),
                "is_navigation_index": bool(r.is_navigation_index),
                "navigation_filtered": bool(r.navigation_filtered),
                "intent_boost_applied": bool(r.intent_boost_applied),
                "exercise_penalty_applied": bool(r.exercise_penalty_applied),
                "test_question_penalty_applied": bool(r.test_question_penalty_applied),
                "schema_boost_applied": bool(r.schema_boost_applied),
                "term_penalty_applied": bool(r.term_penalty_applied),
                "full_term_boost_applied": bool(r.full_term_boost_applied),
                "concept_boost_applied": bool(r.concept_boost_applied),
                "exercise_demoted_for_concept_lookup": bool(r.exercise_demoted_for_concept_lookup),
                "schema_or_rule_boost_applied": bool(r.schema_or_rule_boost_applied),
                "final_rank_reason": r.final_rank_reason,
                "score_before_boosts": float(r.score_before_boosts),
                "concept_full_phrase_boost_value": float(r.concept_full_phrase_boost_value),
                "section_title_boost_value": float(r.section_title_boost_value),
                "schema_or_rule_boost_value": float(r.schema_or_rule_boost_value),
                "quality_penalty_value": float(r.quality_penalty_value),
                "boundary_penalty_value": float(r.boundary_penalty_value),
                "exercise_penalty_value": float(r.exercise_penalty_value),
                "score_after_boosts_before_clamp": float(r.score_after_boosts_before_clamp),
                "expanded_from_neighbors": bool(r.expanded_from_neighbors),
                "penalties_applied": list(r.penalties_applied),
            }
            if debug:
                payload["final_score"] = float(r.final_score or r.score)
            hits.append(payload)
        return hits

    def query(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
        return_sources: bool = True,
    ) -> QueryResponse:
        final_top_k = min(top_k, settings.rag_final_top_k)
        hits = self.retrieve(query, final_top_k, min_score, collection, source_uris, return_text=False)
        if not hits:
            return QueryResponse(answer="Недостаточно данных в источниках.", sources=[])

        sources = [
            Source(
                n=i,
                fragment_id=h["fragment_id"],
                source_uri=h["source_uri"],
                snippet=h["snippet"],
                score=h["score"],
                page=h["page"],
                type=h["type"],
            )
            for i, h in enumerate(hits, start=1)
        ]
        bullets = "\n".join([f"[{s.n}] {s.snippet}" for s in sources[:3]])
        answer = f"Найденные подтверждённые фрагменты:\n{bullets}"
        return QueryResponse(answer=answer, sources=sources if return_sources else [])

    def list_sources(self, collection: str) -> list[SourceInfo]:
        rows = self.repository.list_sources(collection)
        return [SourceInfo(source_uri=row.source_uri, title=row.title) for row in rows]


def _infer_document_metadata(file_path: Path, root: Path, collection: str) -> dict:
    relative = file_path.relative_to(root)
    parts = list(relative.parts)
    parent_parts = parts[:-1]
    class_match = re.search(r"(?P<class>\d{1,2})\s*(?:класс|klass|class)", file_path.stem, re.IGNORECASE)
    return {
        "collection": collection,
        "file_type": file_path.suffix.lower().lstrip("."),
        "path_subject_hint": parent_parts[0] if parent_parts else None,
        "grade": int(class_match.group("class")) if class_match else None,
        "path_parts": parent_parts,
    }


def _build_search_text(text: str, meta: dict, document_profile: dict) -> str:
    parts = [
        meta.get("title"),
        " ".join(str(item) for item in meta.get("section_path") or []),
        meta.get("section_title"),
        meta.get("parent_heading"),
        document_profile.get("subject"),
        " ".join(str(item) for item in document_profile.get("keywords") or []),
        text,
    ]
    return re.sub(r"\s+", " ", " ".join(str(part) for part in parts if part)).strip()


def _heading_path_from_meta(meta: dict) -> list[str]:
    for key in ("section_path", "heading_path"):
        value = meta.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if item]
    for key in ("section_title", "heading", "title"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    return []


def coalesce_parsed_elements(elements: list[ParsedElement]) -> list[ParsedElement]:
    """Merge adjacent short text elements on the same page before chunking."""
    merged: list[ParsedElement] = []
    buffer: list[ParsedElement] = []

    def flush() -> None:
        if not buffer:
            return
        first = buffer[0]
        content = "\n".join(item.content.strip() for item in buffer if item.content.strip()).strip()
        meta = dict(first.meta)
        if len(buffer) > 1:
            meta["merged_element_indices"] = [item.element_index for item in buffer]
        merged.append(
            ParsedElement(
                element_index=first.element_index,
                type=first.type,
                content=content,
                page=first.page,
                meta=meta,
            )
        )
        buffer.clear()

    for element in elements:
        if element.type != "text":
            flush()
            merged.append(element)
            continue

        if buffer and buffer[0].page != element.page:
            flush()

        candidate = "\n".join([*(item.content for item in buffer), element.content]).strip()
        if buffer and len(candidate) > settings.adaptive_chunk_max_chars:
            flush()

        buffer.append(element)
        current_size = sum(len(item.content) for item in buffer) + max(0, len(buffer) - 1)
        if current_size >= settings.adaptive_chunk_min_chars:
            flush()

    flush()
    return merged
