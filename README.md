# RAG Anything FastAPI Microservice

Production-oriented RAG microservice for WordPress integrations. The service ingests raw files, parses with **HKUDS/RAG-Anything** adapter, normalizes to stable fragments, stores vectors in PostgreSQL+pgvector, and returns grounded responses with citations on `source_uri + fragment_id`.

## Features
- `POST /ingest` for batch indexing from a directory.
- `POST /retrieve` for fragment-level semantic retrieval.
- `POST /query` for grounded answer generation with source list.
- `POST /sources` for listing available source files in a collection (for WordPress source picker).
- Stable `fragment_id = sha256(source_uri + element_index + normalized_content_prefix)`.
- Structure-aware fragmenting (headings/paragraphs) with adaptive 800-1200 char chunks and `heading_path` in fragment metadata.
- Hybrid retrieval pipeline: vector recall top-N + BM25 signal + cross-encoder reranking to final top-k.
- Fragment-level indexing with subchunking (`chunk_size=1500`, overlap `180` ≈ 12%).
- Parser observability logs with parse mode and fallback-ratio alerts.
- Quality monitoring CLI for Recall@k / nDCG regression checks across reference query sets.
- X-API-Key authentication, per-IP in-memory rate limiting, JSON-only errors, query size limits, JSON logs.
- `/healthz` liveness endpoint with DB connectivity check.

## Project tree

```text
.
├── app
│   ├── chunking.py
│   ├── config.py
│   ├── db.py
│   ├── embeddings.py
│   ├── main.py
│   ├── models.py
│   ├── parser.py
│   ├── repository.py
│   ├── schemas.py
│   ├── service.py
│   └── utils.py
├── docker-compose.yml
├── migrations
│   └── init.sql
├── requirements.txt
└── tests
    └── test_api.py
```

## Environment

Create `.env`:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
EMBED_DIM=384
EMBED_MODEL=all-MiniLM-L6-v2
FAIL_ON_EMBEDDING_FALLBACK=true
API_KEY=super-secret-key
STORAGE_RAW=storage/raw
REDIS_URL=
APP_ENV=production
INGEST_PATH_MUST_BE_UNDER_STORAGE_RAW=true
RATE_LIMIT_PER_MINUTE=120
```

## Run with Docker Postgres

```bash
docker compose up -d
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run parser only (without API)

Use the standalone script:

```bash
python scripts/run_parser.py --input storage/raw --json
```

Single file example:

```bash
python scripts/run_parser.py --input storage/raw/your_file.pdf --preview-limit 10 --json
```

## Quality monitoring pipeline

Prepare a JSON evaluation set (usually 50-100 labeled queries):

```json
[
  {
    "query": "что такое инфляция",
    "relevant_fragment_ids": ["abc123", "def456"],
    "graded_relevance": {"abc123": 2, "def456": 1}
  }
]
```

Run evaluation after reindex/model changes:

```bash
python scripts/run_quality_eval.py --eval-set eval_set.json --collection default --top-k 10
```

The report includes per-query and mean `Recall@k` / `nDCG@k` for regression monitoring.

## API examples

### Health

```bash
curl http://localhost:8000/healthz
```

### Ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: super-secret-key" \
  -d '{"input_path":"storage/raw","collection":"default","reindex":false}'
```

### Retrieve

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -H "X-API-Key: super-secret-key" \
  -d '{"query":"тема урока: стили речи","top_k":8,"min_score":0.45,"collection":"default","source_uris":["1741176546_rodnoj-jazyk_-9-klass_-voiteleva-t_-m_-2022.pdf"]}'
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: super-secret-key" \
  -d '{"query":"тема урока: стили речи","top_k":8,"min_score":0.45,"mode":"grounded","citation_style":"fragments","return_sources":true,"collection":"default","source_uris":["1741176546_rodnoj-jazyk_-9-klass_-voiteleva-t_-m_-2022.pdf"]}'
```

### Sources (for source picker)

```bash
curl -X POST http://localhost:8000/sources \
  -H "Content-Type: application/json" \
  -H "X-API-Key: super-secret-key" \
  -d '{"collection":"default"}'
```

## Notes
- In production, set a strong `API_KEY` and never keep the default `change-me`.
- In production, `APP_ENV=production` enforces non-default API key at startup.
- `INGEST_PATH_MUST_BE_UNDER_STORAGE_RAW=true` protects from indexing arbitrary directories.
- The parser uses RAG-Anything when available in runtime; if unavailable it degrades to lightweight local parsers for TXT/MD/PDF/DOCX.
- `page` remains optional in all APIs.
- In production keep `FAIL_ON_EMBEDDING_FALLBACK=true` to avoid silent hash-embedding fallback and low-quality retrieval.

## Troubleshooting dependencies
- If your environment cannot resolve a pinned wheel for `pillow`, use the unpinned `pillow` entry from `requirements.txt` (already configured in this repo) so `pip` can pick a compatible build.
- `RAG-Anything` pulls `mineru[core]`, which requires `pypdf>=5.6.0`; therefore this repo uses `pypdf>=5.6.0,<6` to avoid resolver conflicts.
- If installation is still slow because of resolver backtracking, install core deps first and then install RAG-Anything last:
  1. `pip install -r requirements.txt --no-deps`
  2. `pip install "pypdf>=5.6.0,<6"`
  3. `pip install git+https://github.com/HKUDS/RAG-Anything.git`
- For minimal text/PDF/docx ingestion, image-specific packages can be treated as optional if your deployment does not process image OCR/caption pipelines.
