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
- X-API-Key authentication, Redis-backed rate limiting (fallback in-memory), JSON-only errors, query size limits, JSON logs.
- `/healthz` liveness endpoint with DB connectivity check.
- `/readyz` readiness endpoint with model-load checks and pgvector extension verification.
- `/metrics` endpoint with SLO metrics (p95/p99 latency, 5xx error rate).

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
ADMIN_API_KEY=super-admin-secret-key
STORAGE_RAW=storage/raw
STORAGE_PARSED=storage/parsed
REDIS_URL=
APP_ENV=production
INGEST_PATH_MUST_BE_UNDER_STORAGE_RAW=true
RATE_LIMIT_PER_MINUTE=120
UVICORN_WORKERS=2
DISABLE_MINERU_LLM=1
```

## Run with Docker Postgres

```bash
docker compose up -d
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```


## Windows install (two venv strategy)

Core service environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
```

MinerU environment (isolated):

```powershell
python -m venv .venv-mineru
.\.venv-mineru\Scripts\Activate.ps1
pip install -r requirements-mineru.txt
# optional reproducible install
# pip install -r requirements-mineru.txt -c constraints-mineru.txt
```

`requirements-mineru.txt` intentionally contains runtime-critical dependencies (`mineru`, `torch`, `transformers`, `ultralytics`, `doclayout-yolo`, `rapid-table`, `shapely`, `fast-langdetect`) so `.venv-mineru` is self-sufficient.


Ubuntu / VPS setup for MinerU venv:

```bash
python3 -m venv .venv-mineru
./.venv-mineru/bin/pip install -U pip
./.venv-mineru/bin/pip install -r requirements-mineru.txt
```

Point service to MinerU python:

```powershell
$env:MINERU_PYTHON=".venv-mineru\Scripts\python.exe"
```

Linux/macOS:

```bash
export MINERU_PYTHON="./.venv-mineru/bin/python"
```

You can also run:

```powershell
.\scripts\setup_core.ps1
.\scripts\setup_mineru.ps1
.\scripts\doctor.ps1
.\scripts\mineru_setup.ps1
```


- MinerU artifacts are stored in deterministic directories under `STORAGE_PARSED/<doc_id>` (derived from file name), not temp folders.
- With `reindex=false`, existing artifacts in `STORAGE_PARSED` are reused; with `reindex=true`, the directory is recreated before parsing.

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


## Load testing

Locust:

```bash
locust -f scripts/loadtest/locustfile.py --host http://localhost:8000
```

k6:

```bash
k6 run scripts/loadtest/k6_retrieve.js
```

The k6 script includes threshold checks for `p95`, `p99`, and error rate.

## API examples

### Health

```bash
curl http://localhost:8000/healthz -H "X-API-Key: super-secret-key"
```

### Readiness

```bash
curl http://localhost:8000/readyz -H "X-API-Key: super-secret-key"
```

### SLO metrics

```bash
curl http://localhost:8000/metrics -H "X-API-Key: super-secret-key"
```

### Ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-Admin-API-Key: super-admin-secret-key" \
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


### Готовые запросы и типовые ответы

> Ниже примеры в одну строку для **Windows CMD** (`cmd.exe`).

**GET /healthz**

```bat
curl http://localhost:8000/healthz -H "X-API-Key: super-secret-key"
```

Ответ:

```json
{"status":"ok"}
```

**POST /ingest** (только admin key)

```bat
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -H "X-Admin-API-Key: super-admin-secret-key" -d "{\"input_path\":\"storage/raw\",\"collection\":\"default\",\"reindex\":false}"
```

Ответ (пример):

```json
{"indexed_docs":1,"indexed_fragments":42,"indexed_vectors":58}
```

**POST /retrieve**

```bat
curl -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" -H "X-API-Key: super-secret-key" -d "{\"query\":\"тема урока: стили речи\",\"top_k\":3,\"min_score\":0.45,\"collection\":\"default\"}"
```

Ответ (пример):

```json
{"hits":[{"fragment_id":"f1","source_uri":"pub_1167883.pdf","title":"pub_1167883.pdf","type":"text","page":12,"snippet":"...","score":0.9123,"text":null}]}
```

**POST /query**

```bat
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -H "X-API-Key: super-secret-key" -d "{\"query\":\"стили речи\",\"top_k\":3,\"min_score\":0.45,\"mode\":\"grounded\",\"citation_style\":\"fragments\",\"return_sources\":true,\"collection\":\"default\"}"
```

Ответ (пример):

```json
{"answer":"Найденные подтверждённые фрагменты:
[1] ...","sources":[{"n":1,"fragment_id":"f1","source_uri":"pub_1167883.pdf","snippet":"...","score":0.9123,"page":12,"type":"text"}]}
```

**POST /sources**

```bat
curl -X POST http://localhost:8000/sources -H "Content-Type: application/json" -H "X-API-Key: super-secret-key" -d "{\"collection\":\"default\"}"
```

Ответ (пример):

```json
{"sources":[{"source_uri":"pub_1167883.pdf","title":"pub_1167883.pdf"}]}
```

Частые ошибки:

```json
{"error":"Invalid API key"}
```

```json
{"error":"Invalid admin API key"}
```

```json
{"error":"input_path must be an existing directory"}
```

## Notes
- In production, set strong `API_KEY` and `ADMIN_API_KEY`, never keep defaults (`change-me`, `change-me-admin`).
- In production, `APP_ENV=production` enforces non-default API keys at startup.
- `INGEST_PATH_MUST_BE_UNDER_STORAGE_RAW=true` protects from indexing arbitrary directories.
- GET endpoints (`/healthz`, `/readyz`, `/metrics`) are protected by `X-API-Key`.
- `POST /ingest` is admin-only and requires `X-Admin-API-Key`.
- Models are initialized once at startup and reused across requests for better parallel performance.
- Retrieval candidate recall uses pgvector ANN in SQL (`embedding <=> query_vector` + top-N) before hybrid BM25 and rerank to reduce Python CPU/RAM on large datasets.
- The parser uses RAG-Anything when available in runtime; if unavailable it degrades to lightweight local parsers for TXT/MD/PDF/DOCX.
- `page` remains optional in all APIs.
- In production keep `FAIL_ON_EMBEDDING_FALLBACK=true` to avoid silent hash-embedding fallback and low-quality retrieval.

## Troubleshooting dependencies
- If your environment cannot resolve a pinned wheel for `pillow`, use the unpinned `pillow` entry from `requirements.txt` (already configured in this repo) so `pip` can pick a compatible build.
- `RAG-Anything` pulls `mineru[core]`, which requires `pypdf>=5.6.0`; therefore this repo uses `pypdf>=5.6.0,<6` to avoid resolver conflicts.
- If installation is still slow because of resolver backtracking, install core deps first and then install RAG-Anything last:
  1. `pip install -r requirements-core.txt --no-deps`
  2. `pip install "pypdf>=5.6.0,<6"`
  3. `pip install git+https://github.com/HKUDS/RAG-Anything.git`
- For minimal text/PDF/docx ingestion, image-specific packages can be treated as optional if your deployment does not process image OCR/caption pipelines.
- Newer releases may expose package name/layout as `raganything` (without underscore) and different internal modules; this service now auto-detects supported parser entrypoints across known layouts.
- MinerU runs only via `MINERU_PYTHON` (required). If it is unset, `/ingest` fails fast with a configuration error so the runtime is explicit and reproducible.
- MinerU CLI flags vary by version; service auto-detects supported options via `--help` and reads parsing artifacts from `output_dir` when `--json` is unavailable.
- MinerU readiness is checked only when parsing PDF, not during FastAPI startup. If `.venv-mineru` is missing/broken, API returns `503` with remediation (`pip install -r requirements-mineru.txt` in `.venv-mineru`).
- During MinerU execution, `ModuleNotFoundError` is auto-detected from stderr and logged as `mineru_missing_dependency_detected` with module→package hints (for example `doclayout_yolo -> doclayout-yolo`, `fast_langdetect -> fast-langdetect`, `ultralytics -> ultralytics`, `rapid_table -> rapid-table`).
- Missing MinerU dependencies are never auto-installed at runtime; keep `.venv-mineru` reproducible and reinstall it with `pip install -r requirements-mineru.txt` after dependency changes.
- Before parsing, service performs fail-fast import check in `.venv-mineru` (`dill, shapely, pyclipper, torch, transformers`) and returns an error immediately when something is missing.
- LLM-aided title enhancement is disabled by default (`DISABLE_MINERU_LLM=1`) and falls back to identity title function, so `openai` package is not required for MinerU pipeline execution.
- MinerU subprocess execution has no default timeout (supports long CPU parsing for large PDFs); success is validated by return code, stderr patterns (`Traceback`, `ModuleNotFoundError`, `ERROR`), and non-empty recursive artifacts; failures may trigger one text-only retry before fallback.
- MinerU + transformers incompatibility (`cache_position`): if logs show `UnimerMBartForCausalLM.forward() got an unexpected keyword argument 'cache_position'`, pin transformers to MinerU-compatible version and restart service:
  1. `pip install "transformers==4.35.0"`
  2. restart API process (`uvicorn`/systemd).
- Embeddings dependency mismatch (`split_torch_state_dict_into_shards` / `huggingface_hub` / `accelerate`): run `scripts/repair_env.ps1` (PowerShell) to reinstall a compatible core stack (`huggingface-hub<0.18`, `tokenizers==0.14.1`), then restart API.
- Recommended compatible ML stack for this service: `transformers==4.35.0`, `huggingface-hub>=0.16.4,<0.18`, `tokenizers==0.14.1`, `sentence-transformers>=2.2`, `safetensors` (accelerate moved to optional `requirements-accelerate.txt`).
