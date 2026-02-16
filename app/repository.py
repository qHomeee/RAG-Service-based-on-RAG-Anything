from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.chunking import split_to_subchunks
from app.embeddings import EmbeddingProvider
from app.models import Document, Embedding, Fragment
from app.schemas import CanonicalFragment
from app.utils import cosine_similarity


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
    def __init__(self, db: Session, embeddings: EmbeddingProvider) -> None:
        self.db = db
        self.embeddings = embeddings

    def upsert_document(self, source_uri: str, title: str | None, collection: str, meta: dict, reindex: bool) -> Document:
        doc = self.db.scalar(select(Document).where(Document.source_uri == source_uri))
        if doc and reindex:
            self.db.execute(delete(Embedding).where(Embedding.fragment_id.in_(select(Fragment.fragment_id).where(Fragment.doc_id == doc.doc_id))))
            self.db.execute(delete(Fragment).where(Fragment.doc_id == doc.doc_id))
            self.db.flush()
        if doc:
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
        qvec = self.embeddings.embed(query)
        docs_stmt = select(Document.doc_id).where(Document.collection == collection)
        if source_uris:
            docs_stmt = docs_stmt.where(Document.source_uri.in_(source_uris))
        docs = self.db.scalars(docs_stmt).all()
        if not docs:
            return []

        emb_rows = self.db.scalars(
            select(Embedding).join(Fragment, Embedding.fragment_id == Fragment.fragment_id).where(Fragment.doc_id.in_(docs))
        ).all()

        best_by_fragment: dict[str, RetrievalRow] = {}
        for emb in emb_rows:
            frag = emb.fragment
            score = cosine_similarity(qvec, emb.embedding)
            if score < min_score:
                continue
            current = best_by_fragment.get(frag.fragment_id)
            if current and current.score >= score:
                continue
            best_by_fragment[frag.fragment_id] = RetrievalRow(
                fragment_id=frag.fragment_id,
                source_uri=frag.source_uri,
                title=frag.document.title,
                type=frag.type,
                page=frag.page,
                snippet=frag.snippet,
                score=score,
                text=emb.text,
            )
        return sorted(best_by_fragment.values(), key=lambda r: r.score, reverse=True)[:top_k]

    def list_sources(self, collection: str) -> list[SourceRow]:
        rows = self.db.scalars(
            select(Document).where(Document.collection == collection).order_by(Document.source_uri.asc())
        ).all()
        return [SourceRow(source_uri=row.source_uri, title=row.title) for row in rows]
