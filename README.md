# RAG Anything FastAPI Microservice

Production-oriented RAG microservice for WordPress integrations. The service ingests raw files, parses with **HKUDS/RAG-Anything** adapter, normalizes to stable fragments, stores vectors in PostgreSQL+pgvector, and returns grounded responses with citations on `source_uri + fragment_id`.

## Features
- `POST /ingest` for batch indexing from a directory.
- `POST /retrieve` for fragment-level semantic retrieval.
- `POST /query` for grounded answer generation with source list.
- Stable `fragment_id = sha256(source_uri + element_index + normalized_content_prefix)`.
- Fragment-level indexing with subchunking (`chunk_size=1500`, overlap `180` ≈ 12%).
- X-API-Key authentication, JSON-only errors, query size limits, JSON logs.

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
API_KEY=super-secret-key
STORAGE_RAW=storage/raw
REDIS_URL=
```

## Run with Docker Postgres

```bash
docker compose up -d
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API examples

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
  -d '{"query":"elasticity of demand","top_k":12,"min_score":0.2,"collection":"default"}'
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: super-secret-key" \
  -d '{"query":"What is inflation?","top_k":10,"mode":"grounded","citation_style":"fragments","return_sources":true}'
```

## Notes
- The parser uses RAG-Anything when available in runtime; if unavailable it degrades to lightweight local parsers for TXT/MD/PDF/DOCX.
- `page` remains optional in all APIs.

## Troubleshooting dependencies
- If your environment cannot resolve a pinned wheel for `pillow`, use the unpinned `pillow` entry from `requirements.txt` (already configured in this repo) so `pip` can pick a compatible build.
- For minimal text/PDF/docx ingestion, image-specific packages can be treated as optional if your deployment does not process image OCR/caption pipelines.

