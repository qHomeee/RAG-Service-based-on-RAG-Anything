import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

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
        expanded_query = _expand_query(query)
        qvec = self.embeddings.embed(expanded_query)
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

        query_terms = _tokenize(expanded_query)
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


QUERY_SYNONYMS: dict[str, list[str]] = {
    "инфляция": ["рост цен", "индекс потребительских цен", "обесценивание"],
    "ввп": ["валовой внутренний продукт", "gdp"],
    "стили": ["стиль", "жанр", "речь"],
    "налог": ["налогообложение", "сбор", "пошлина"],
}


def _expand_query(query: str) -> str:
    if not settings.query_expansion_enabled:
        return query

    base_terms = _tokenize(query)
    extras: list[str] = []
    for term in base_terms:
        extras.extend(QUERY_SYNONYMS.get(term, []))

    if not extras:
        return query

    seen: set[str] = set()
    deduped: list[str] = []
    for piece in [query, *extras]:
        norm = piece.strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(piece.strip())

    return " ".join(deduped)


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
