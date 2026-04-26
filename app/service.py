import logging
from pathlib import Path
import re

from app.chunking import split_structured_chunks
from app.config import settings
from app.parser import RAGAnythingParser
from app.repository import RagRepository
from app.schemas import CanonicalFragment, QueryResponse, Source, SourceInfo
from app.utils import snippet_from_text, stable_fragment_id

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".txt", ".md", ".png", ".jpg", ".jpeg"}
logger = logging.getLogger("rag_service")


class RagService:
    def __init__(self, parser: RAGAnythingParser, repository: RagRepository) -> None:
        self.parser = parser
        self.repository = repository

    def ingest(self, input_path: str, collection: str, reindex: bool) -> dict[str, int]:
        root = Path(input_path)
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]

        indexed_docs = indexed_fragments = indexed_vectors = 0
        fallback_docs = 0
        for file_path in files:
            source_uri = str(file_path.relative_to(root)).replace("\\", "/")
            parsed, parse_mode = self.parser.parse_file_with_mode(source_uri=source_uri, path=file_path, reindex=reindex)
            if parse_mode == "fallback":
                fallback_docs += 1

            doc = self.repository.upsert_document(
                source_uri,
                file_path.name,
                collection,
                {
                    "path": str(file_path),
                    "parse_mode": parse_mode,
                    **_infer_document_metadata(file_path, root, collection),
                },
                reindex,
            )
            indexed_docs += 1

            for elem in parsed:
                structured_chunks = split_structured_chunks(elem.content)
                if not structured_chunks:
                    continue

                for chunk_idx, chunk in enumerate(structured_chunks):
                    fragment_id = stable_fragment_id(source_uri, elem.element_index * 10_000 + chunk_idx, chunk.text)
                    meta = dict(elem.meta)
                    meta["heading_path"] = chunk.heading_path
                    meta["source_uri"] = source_uri
                    meta["title"] = getattr(doc, "title", file_path.name)
                    meta["collection"] = collection
                    meta["page"] = elem.page
                    meta["chunk_index"] = chunk_idx
                    fragment = CanonicalFragment(
                        fragment_id=fragment_id,
                        element_index=elem.element_index,
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
    ) -> list[dict]:
        rows = self.repository.retrieve(query, top_k, min_score, collection, source_uris)
        return self._rows_to_hits(rows, return_text=return_text, debug=False)

    def retrieve_with_debug(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
        return_text: bool,
    ) -> tuple[list[dict], dict | None]:
        result = self.repository.retrieve_with_debug(
            query,
            top_k,
            min_score,
            collection,
            source_uris,
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
                "snippet": r.text,
                "score": float(r.final_score or r.score),
                "text": r.text if return_text else None,
            }
            if debug:
                payload.update(
                    {
                        "dense_score": float(r.dense_score),
                        "lexical_score": float(r.lexical_score),
                        "rerank_score": float(r.rerank_score) if r.rerank_score is not None else None,
                        "final_score": float(r.final_score or r.score),
                        "lexical_overlap": float(r.lexical_overlap),
                        "document_score": float(r.document_score),
                        "rrf_score": float(r.rrf_score),
                    }
                )
            hits.append(payload)
        return hits

    def query(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
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
        return QueryResponse(answer=answer, sources=sources)

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
        "subject": parent_parts[0] if parent_parts else None,
        "grade": int(class_match.group("class")) if class_match else None,
        "path_parts": parent_parts,
    }
