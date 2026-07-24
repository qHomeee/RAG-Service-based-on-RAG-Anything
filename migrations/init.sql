CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY,
    source_uri TEXT NOT NULL,
    title TEXT,
    collection TEXT NOT NULL DEFAULT 'default',
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_documents_collection_source UNIQUE (collection, source_uri)
);

COMMENT ON COLUMN documents.meta IS
    'Document routing metadata: document_profile, subject, grade, doc_type, language, keywords, section_titles.';

CREATE TABLE IF NOT EXISTS fragments (
    fragment_id TEXT PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    source_uri TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('text', 'table', 'image', 'equation')),
    page INT,
    element_index INT NOT NULL,
    text TEXT NOT NULL,
    snippet TEXT NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector(
            'russian'::regconfig,
            coalesce(text, '') || ' ' ||
            coalesce(snippet, '') || ' ' ||
            coalesce(meta->>'section_title', '') || ' ' ||
            coalesce(meta->>'search_text', '')
        )
    ) STORED
);

COMMENT ON COLUMN fragments.meta IS
    'Chunk routing metadata: section_path, section_title, subject, grade, doc_type, language, is_toc, heading_path.';

CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
CREATE INDEX IF NOT EXISTS idx_documents_meta_gin ON documents USING gin (meta);
CREATE INDEX IF NOT EXISTS idx_documents_meta_subject ON documents ((meta->>'subject'));
CREATE INDEX IF NOT EXISTS idx_documents_meta_grade ON documents ((meta->>'grade'));
CREATE INDEX IF NOT EXISTS idx_documents_meta_doc_type ON documents ((meta->>'doc_type'));
CREATE INDEX IF NOT EXISTS idx_fragments_doc_id ON fragments(doc_id);
CREATE INDEX IF NOT EXISTS idx_fragments_source_uri ON fragments(source_uri);
CREATE INDEX IF NOT EXISTS idx_fragments_meta_gin ON fragments USING gin (meta);
CREATE INDEX IF NOT EXISTS idx_fragments_meta_subject ON fragments ((meta->>'subject'));
CREATE INDEX IF NOT EXISTS idx_fragments_meta_section_title ON fragments ((meta->>'section_title'));
CREATE INDEX IF NOT EXISTS idx_fragments_meta_is_toc ON fragments ((meta->>'is_toc'));
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY,
    fragment_id TEXT NOT NULL REFERENCES fragments(fragment_id) ON DELETE CASCADE,
    subchunk_index INT NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT uq_fragment_subchunk UNIQUE(fragment_id, subchunk_index)
);

COMMENT ON COLUMN embeddings.meta IS
    'Embedding metadata copied from chunk/document routing fields where available.';

CREATE INDEX IF NOT EXISTS idx_embeddings_fragment_id ON embeddings(fragment_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
