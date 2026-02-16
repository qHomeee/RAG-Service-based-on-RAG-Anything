from pathlib import Path

from app.parser import RAGAnythingParser
from app.repository import RagRepository
from app.schemas import CanonicalFragment, QueryResponse, Source, SourceInfo
from app.utils import snippet_from_text, stable_fragment_id

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".txt", ".md", ".png", ".jpg", ".jpeg"}


class RagService:
    def __init__(self, parser: RAGAnythingParser, repository: RagRepository) -> None:
        self.parser = parser
        self.repository = repository

    def ingest(self, input_path: str, collection: str, reindex: bool) -> dict[str, int]:
        root = Path(input_path)
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]

        indexed_docs = indexed_fragments = indexed_vectors = 0
        for file_path in files:
            source_uri = str(file_path.relative_to(root)).replace("\\", "/")
            doc = self.repository.upsert_document(source_uri, file_path.name, collection, {"path": str(file_path)}, reindex)
            indexed_docs += 1
            parsed = self.parser.parse_file(source_uri=source_uri, path=file_path)
            for elem in parsed:
                fragment_id = stable_fragment_id(source_uri, elem.element_index, elem.content)
                fragment = CanonicalFragment(
                    fragment_id=fragment_id,
                    element_index=elem.element_index,
                    source_uri=source_uri,
                    type=elem.type,
                    page=elem.page,
                    text=elem.content,
                    snippet=snippet_from_text(elem.content),
                    meta=elem.meta,
                )
                vectors = self.repository.insert_fragment_with_embeddings(doc, fragment)
                if vectors > 0:
                    indexed_fragments += 1
                    indexed_vectors += vectors

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
        return [
            {
                "fragment_id": r.fragment_id,
                "source_uri": r.source_uri,
                "title": r.title,
                "type": r.type,
                "page": r.page,
                "snippet": r.snippet,
                "score": float(round(r.score, 4)),
                "text": r.text if return_text else None,
            }
            for r in rows
        ]

    def query(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
    ) -> QueryResponse:
        hits = self.retrieve(query, top_k, min_score, collection, source_uris, return_text=False)
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
