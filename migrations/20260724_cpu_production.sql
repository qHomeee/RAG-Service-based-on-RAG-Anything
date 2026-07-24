BEGIN;

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_source_uri_key;

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS uq_documents_collection_source;

ALTER TABLE documents
    ADD CONSTRAINT uq_documents_collection_source UNIQUE (collection, source_uri);

ALTER TABLE fragments
    ADD COLUMN IF NOT EXISTS search_tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector(
            'russian'::regconfig,
            coalesce(text, '') || ' ' ||
            coalesce(snippet, '') || ' ' ||
            coalesce(meta->>'section_title', '') || ' ' ||
            coalesce(meta->>'search_text', '')
        )
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_fragments_search_tsv
    ON fragments USING gin (search_tsv);

DROP INDEX IF EXISTS idx_embeddings_vector_ivfflat;

CREATE INDEX IF NOT EXISTS idx_embeddings_vector_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMIT;
