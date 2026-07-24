import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.db import engine
from app.embeddings import EmbeddingProvider


INGEST_ADVISORY_LOCK_ID = 7_240_716_001
SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


def _validate_backup_prefix(value: str) -> str:
    prefix = value.strip().lower()
    if not SAFE_IDENTIFIER.fullmatch(prefix):
        raise argparse.ArgumentTypeError(
            "backup prefix must match [a-z][a-z0-9_]{0,39}; "
            "the script appends _embeddings and _documents"
        )
    return prefix


def _default_backup_prefix() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"reembed_backup_{timestamp}"


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _document_metadata_with_embedding(
    meta: dict[str, Any] | None,
    *,
    summary_embedding: list[float],
    fingerprint: str,
) -> dict[str, Any]:
    updated = dict(meta or {})
    profile = dict(updated.get("document_profile") or {})
    profile["summary_embedding"] = summary_embedding
    updated["document_profile"] = profile
    updated["embedding_fingerprint"] = fingerprint
    return updated


def _table_exists(connection, table_name: str) -> bool:
    return bool(
        connection.execute(
            text("SELECT to_regclass(:table_name) IS NOT NULL"),
            {"table_name": f"public.{table_name}"},
        ).scalar()
    )


def _load_index_rows(connection, collection: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    embedding_rows = [
        dict(row)
        for row in connection.execute(
            text(
                """
                SELECT e.id::text AS id, e.text
                FROM embeddings e
                JOIN fragments f ON f.fragment_id = e.fragment_id
                JOIN documents d ON d.doc_id = f.doc_id
                WHERE d.collection = :collection
                ORDER BY e.id
                """
            ),
            {"collection": collection},
        ).mappings()
    ]
    document_rows = [
        dict(row)
        for row in connection.execute(
            text(
                """
                SELECT d.doc_id::text AS doc_id, d.source_uri, d.title, d.meta
                FROM documents d
                WHERE d.collection = :collection
                  AND EXISTS (
                      SELECT 1
                      FROM fragments f
                      JOIN embeddings e ON e.fragment_id = f.fragment_id
                      WHERE f.doc_id = d.doc_id
                  )
                ORDER BY d.source_uri
                """
            ),
            {"collection": collection},
        ).mappings()
    ]
    return embedding_rows, document_rows


def _profile_text(row: dict[str, Any]) -> str:
    meta = dict(row.get("meta") or {})
    profile = dict(meta.get("document_profile") or {})
    value = str(profile.get("profile_text") or "").strip()
    if value:
        return value
    return " ".join(
        part
        for part in (
            str(row.get("title") or "").strip(),
            str(row.get("source_uri") or "").strip(),
        )
        if part
    )


def _encode_rows(
    provider: EmbeddingProvider,
    embedding_rows: list[dict[str, Any]],
    document_rows: list[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    encoded_embeddings: list[dict[str, Any]] = []
    for batch_number, batch in enumerate(_chunks(embedding_rows, batch_size), start=1):
        vectors = provider.embed_many([str(row["text"]) for row in batch], batch_size=batch_size)
        encoded_embeddings.extend(
            {
                "id": row["id"],
                "embedding": json.dumps(vector, separators=(",", ":")),
                "fingerprint": provider.model_fingerprint,
            }
            for row, vector in zip(batch, vectors, strict=True)
        )
        completed = min(batch_number * batch_size, len(embedding_rows))
        print(f"\rEncoded embeddings: {completed}/{len(embedding_rows)}", end="", flush=True)
    if embedding_rows:
        print()

    profile_texts = [_profile_text(row) for row in document_rows]
    summary_vectors = provider.embed_many(profile_texts, batch_size=batch_size)
    encoded_documents = [
        {
            "doc_id": row["doc_id"],
            "metadata": json.dumps(
                _document_metadata_with_embedding(
                    row.get("meta"),
                    summary_embedding=vector,
                    fingerprint=provider.model_fingerprint,
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
        for row, vector in zip(document_rows, summary_vectors, strict=True)
    ]
    return encoded_embeddings, encoded_documents


def _create_backup(connection, collection: str, prefix: str) -> tuple[str, str]:
    embeddings_table = f"{prefix}_embeddings"
    documents_table = f"{prefix}_documents"
    if _table_exists(connection, embeddings_table) or _table_exists(connection, documents_table):
        raise RuntimeError(f"Backup tables for prefix {prefix!r} already exist")

    connection.execute(
        text(
            f"""
            CREATE TABLE public.{embeddings_table} AS
            SELECT e.*
            FROM embeddings e
            JOIN fragments f ON f.fragment_id = e.fragment_id
            JOIN documents d ON d.doc_id = f.doc_id
            WHERE d.collection = :collection
            """
        ),
        {"collection": collection},
    )
    connection.execute(
        text(
            f"""
            CREATE TABLE public.{documents_table} AS
            SELECT d.*
            FROM documents d
            WHERE d.collection = :collection
            """
        ),
        {"collection": collection},
    )
    return embeddings_table, documents_table


def _apply_updates(
    connection,
    *,
    collection: str,
    backup_prefix: str,
    encoded_embeddings: list[dict[str, Any]],
    encoded_documents: list[dict[str, Any]],
    fingerprint: str,
) -> tuple[str, str]:
    embeddings_table, documents_table = _create_backup(connection, collection, backup_prefix)
    connection.execute(
        text(
            """
            UPDATE embeddings
            SET embedding = CAST(:embedding AS vector),
                meta = jsonb_set(
                    COALESCE(meta, '{}'::jsonb),
                    '{embedding_fingerprint}',
                    to_jsonb(CAST(:fingerprint AS text)),
                    true
                )
            WHERE id = CAST(:id AS uuid)
            """
        ),
        encoded_embeddings,
    )
    connection.execute(
        text(
            """
            UPDATE documents
            SET meta = CAST(:metadata AS jsonb)
            WHERE doc_id = CAST(:doc_id AS uuid)
            """
        ),
        encoded_documents,
    )

    vector_count = connection.execute(
        text(
            """
            SELECT count(*)
            FROM embeddings e
            JOIN fragments f ON f.fragment_id = e.fragment_id
            JOIN documents d ON d.doc_id = f.doc_id
            WHERE d.collection = :collection
              AND e.meta->>'embedding_fingerprint' = :fingerprint
              AND vector_dims(e.embedding) = :dimension
            """
        ),
        {
            "collection": collection,
            "fingerprint": fingerprint,
            "dimension": settings.embed_dim,
        },
    ).scalar_one()
    document_count = connection.execute(
        text(
            """
            SELECT count(*)
            FROM documents d
            WHERE d.collection = :collection
              AND d.meta->>'embedding_fingerprint' = :fingerprint
            """
        ),
        {"collection": collection, "fingerprint": fingerprint},
    ).scalar_one()
    if vector_count != len(encoded_embeddings) or document_count != len(encoded_documents):
        raise RuntimeError(
            "Post-update verification failed: "
            f"vectors={vector_count}/{len(encoded_embeddings)}, "
            f"documents={document_count}/{len(encoded_documents)}"
        )
    return embeddings_table, documents_table


def _restore_backup(connection, prefix: str) -> tuple[int, int]:
    embeddings_table = f"{prefix}_embeddings"
    documents_table = f"{prefix}_documents"
    if not _table_exists(connection, embeddings_table) or not _table_exists(connection, documents_table):
        raise RuntimeError(f"Both backup tables for prefix {prefix!r} must exist")

    embedding_result = connection.execute(
        text(
            f"""
            UPDATE embeddings current
            SET embedding = backup.embedding,
                meta = backup.meta
            FROM public.{embeddings_table} backup
            WHERE current.id = backup.id
            """
        )
    )
    document_result = connection.execute(
        text(
            f"""
            UPDATE documents current
            SET meta = backup.meta
            FROM public.{documents_table} backup
            WHERE current.doc_id = backup.doc_id
            """
        )
    )
    return int(embedding_result.rowcount), int(document_result.rowcount)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically re-embed stored RAG fragments without reparsing source files."
    )
    parser.add_argument("--collection", default="default")
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    parser.add_argument("--backup-prefix", type=_validate_backup_prefix, default=_default_backup_prefix())
    parser.add_argument("--apply", action="store_true", help="Apply changes; otherwise perform a dry run")
    parser.add_argument(
        "--restore-prefix",
        type=_validate_backup_prefix,
        help="Restore embedding and document metadata from existing backup tables",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.restore_prefix and args.apply:
        parser.error("--restore-prefix and --apply are mutually exclusive")

    with engine.connect() as connection:
        acquired = bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(:lock_id)"),
                {"lock_id": INGEST_ADVISORY_LOCK_ID},
            ).scalar()
        )
        connection.commit()
        if not acquired:
            raise RuntimeError("Another ingest or re-embedding job is already running")

        try:
            connection.execute(text("SET statement_timeout TO 0"))
            connection.commit()
            if args.restore_prefix:
                with connection.begin():
                    embeddings_restored, documents_restored = _restore_backup(
                        connection, args.restore_prefix
                    )
                connection.execute(text("ANALYZE embeddings"))
                connection.commit()
                print(
                    f"Restored {embeddings_restored} embeddings and "
                    f"{documents_restored} documents from {args.restore_prefix}"
                )
                return

            embedding_rows, document_rows = _load_index_rows(connection, args.collection)
            connection.commit()
            if not embedding_rows:
                raise RuntimeError(f"No indexed embeddings found in collection {args.collection!r}")

            provider = EmbeddingProvider()
            if provider.using_fallback:
                raise RuntimeError("Refusing to re-embed with fallback hash embeddings")
            print(f"Collection: {args.collection}")
            print(f"Embeddings: {len(embedding_rows)}")
            print(f"Documents: {len(document_rows)}")
            print(f"Model fingerprint: {provider.model_fingerprint}")
            if not args.apply:
                print("Dry run only. Re-run with --apply to create a backup and update the index.")
                return

            encoded_embeddings, encoded_documents = _encode_rows(
                provider,
                embedding_rows,
                document_rows,
                batch_size=args.batch_size,
            )
            with connection.begin():
                embeddings_table, documents_table = _apply_updates(
                    connection,
                    collection=args.collection,
                    backup_prefix=args.backup_prefix,
                    encoded_embeddings=encoded_embeddings,
                    encoded_documents=encoded_documents,
                    fingerprint=provider.model_fingerprint,
                )
            connection.execute(text("ANALYZE embeddings"))
            connection.commit()
            print(
                f"Updated {len(encoded_embeddings)} embeddings and "
                f"{len(encoded_documents)} documents atomically."
            )
            print(f"Backup tables: {embeddings_table}, {documents_table}")
            print(
                "Rollback: python -m scripts.reembed_existing "
                f"--restore-prefix {args.backup_prefix}"
            )
        finally:
            connection.execute(
                text("SELECT pg_advisory_unlock(:lock_id)"),
                {"lock_id": INGEST_ADVISORY_LOCK_ID},
            )
            connection.commit()


if __name__ == "__main__":
    main()
