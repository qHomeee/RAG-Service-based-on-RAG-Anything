import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.chunking import split_to_subchunks
from app.config import settings
from app.embeddings import EmbeddingProvider
from app.models import Document, Embedding, Fragment
from app.reranker import CrossEncoderReranker
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

        by_fragment: dict[str, RetrievalRow] = {}
        for emb in emb_rows:
            frag = emb.fragment
            score = cosine_similarity(qvec, emb.embedding)
            current = by_fragment.get(frag.fragment_id)
            if current and current.score >= score:
                continue
            by_fragment[frag.fragment_id] = RetrievalRow(
                fragment_id=frag.fragment_id,
                source_uri=frag.source_uri,
                title=frag.document.title,
                type=frag.type,
                page=frag.page,
                snippet=frag.snippet,
                score=score,
                text=frag.text,
            )

        if not by_fragment:
            return []

        vector_ranked = sorted(by_fragment.values(), key=lambda r: r.score, reverse=True)
        recall_top_n = max(top_k, settings.vector_recall_top_n)
        recall_candidates = vector_ranked[:recall_top_n]

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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w-]+", (text or "").lower())


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
