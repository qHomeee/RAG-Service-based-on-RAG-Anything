CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY,
    source_uri TEXT UNIQUE NOT NULL,
    title TEXT,
    collection TEXT NOT NULL DEFAULT 'default',
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fragments (
    fragment_id TEXT PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    source_uri TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('text', 'table', 'image', 'equation')),
    page INT,
    element_index INT NOT NULL,
    text TEXT NOT NULL,
    snippet TEXT NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_fragments_doc_id ON fragments(doc_id);
CREATE INDEX IF NOT EXISTS idx_fragments_source_uri ON fragments(source_uri);

CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY,
    fragment_id TEXT NOT NULL REFERENCES fragments(fragment_id) ON DELETE CASCADE,
    subchunk_index INT NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT uq_fragment_subchunk UNIQUE(fragment_id, subchunk_index)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_fragment_id ON embeddings(fragment_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_ivfflat ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
