BEGIN;

-- Existing databases may still have fragments.snippet as varchar(450).
-- Retrieval chunking now stores 700-1200 character contexts, so this must be TEXT.
ALTER TABLE fragments
    ALTER COLUMN snippet TYPE TEXT;

-- Query understanding, document routing, and section-aware retrieval store structured
-- profiles in metadata. Keep the physical type JSONB so it can be indexed.
ALTER TABLE documents ALTER COLUMN meta DROP DEFAULT;
ALTER TABLE documents
    ALTER COLUMN meta TYPE JSONB USING COALESCE(meta::jsonb, '{}'::jsonb),
    ALTER COLUMN meta SET DEFAULT '{}'::jsonb,
    ALTER COLUMN meta SET NOT NULL;

ALTER TABLE fragments ALTER COLUMN meta DROP DEFAULT;
ALTER TABLE fragments
    ALTER COLUMN meta TYPE JSONB USING COALESCE(meta::jsonb, '{}'::jsonb),
    ALTER COLUMN meta SET DEFAULT '{}'::jsonb,
    ALTER COLUMN meta SET NOT NULL;

ALTER TABLE embeddings ALTER COLUMN meta DROP DEFAULT;
ALTER TABLE embeddings
    ALTER COLUMN meta TYPE JSONB USING COALESCE(meta::jsonb, '{}'::jsonb),
    ALTER COLUMN meta SET DEFAULT '{}'::jsonb,
    ALTER COLUMN meta SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
CREATE INDEX IF NOT EXISTS idx_documents_meta_gin ON documents USING gin (meta);
CREATE INDEX IF NOT EXISTS idx_documents_meta_subject ON documents ((meta->>'subject'));
CREATE INDEX IF NOT EXISTS idx_documents_meta_grade ON documents ((meta->>'grade'));
CREATE INDEX IF NOT EXISTS idx_documents_meta_doc_type ON documents ((meta->>'doc_type'));

CREATE INDEX IF NOT EXISTS idx_fragments_meta_gin ON fragments USING gin (meta);
CREATE INDEX IF NOT EXISTS idx_fragments_meta_subject ON fragments ((meta->>'subject'));
CREATE INDEX IF NOT EXISTS idx_fragments_meta_section_title ON fragments ((meta->>'section_title'));
CREATE INDEX IF NOT EXISTS idx_fragments_meta_is_toc ON fragments ((meta->>'is_toc'));

COMMENT ON COLUMN documents.meta IS
    'Document routing metadata: document_profile, subject, grade, doc_type, language, keywords, section_titles.';
COMMENT ON COLUMN fragments.meta IS
    'Chunk routing metadata: section_path, section_title, subject, grade, doc_type, language, is_toc, heading_path.';
COMMENT ON COLUMN embeddings.meta IS
    'Embedding metadata copied from chunk/document routing fields where available.';

COMMIT;
