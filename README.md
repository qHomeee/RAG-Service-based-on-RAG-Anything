# RAG Anything FastAPI Microservice

CPU production profile for a VPS with **8 CPU cores, 16 GB RAM and no GPU** is documented in
[docs/vps-production.md](docs/vps-production.md). The ready-to-run stack is
`docker-compose.vps.yml`; it uses two API workers, PostgreSQL/pgvector, Redis, offline local
embedding/reranker models, bounded memory and CPU, health checks, migrations and log rotation.
The latest local CPU measurements are in
[docs/benchmark-2026-07-24.md](docs/benchmark-2026-07-24.md).

Production-oriented RAG microservice for WordPress integrations. The service ingests raw files, parses with **HKUDS/RAG-Anything** adapter, normalizes to stable fragments, stores vectors in PostgreSQL+pgvector, and returns grounded responses with citations on `source_uri + fragment_id`.

## Features
- `POST /ingest` for batch indexing from a directory.
- `POST /retrieve` for fragment-level semantic retrieval.
- `POST /query` for an extractive grounded response with source list (this endpoint does not call a generative LLM).
- `POST /sources` for listing available source files in a collection (for WordPress source picker).
- Stable `fragment_id = sha256(source_uri + element_index + normalized_content_prefix)`.
- Structure-aware fragmenting (headings/paragraphs) with adaptive 800-1200 char chunks and `heading_path` in fragment metadata.
- Hybrid retrieval pipeline: HNSW vector recall + PostgreSQL full-text search + balanced CPU cross-encoder reranking.
- Fragment-level indexing with subchunking (`chunk_size=1500`, overlap `180` ≈ 12%).
- Parser observability logs with parse mode and fallback-ratio alerts.
- Quality monitoring CLI for Recall@k / nDCG regression checks across reference query sets.
- A checked-in 50-query CPU acceptance set with source/page/term evidence labels and negative-query gates.
- X-API-Key authentication, mandatory production Redis rate limiting, bounded requests and JSON logs.
- Public `/livez`, authenticated `/healthz`, and fail-closed `/readyz` with model/index compatibility checks.
- `/metrics` for SLO snapshots and `/metrics/prometheus` for Prometheus scraping.

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
APP_ENV=production
ALLOWED_HOSTS=["rag.example.com","localhost","127.0.0.1"]
DATABASE_URL=postgresql+psycopg://rag:strong-password@localhost:5432/rag
REDIS_URL=redis://localhost:6379/0
EMBED_DIM=384
EMBED_MODEL=storage/models/paraphrase-multilingual-MiniLM-L12-v2
EMBED_OFFLINE=true
FAIL_ON_EMBEDDING_FALLBACK=true
ENFORCE_EMBEDDING_MODEL_COMPATIBILITY=true
RERANKER_MODEL=storage/models/mmarco-mMiniLMv2-L12-H384-v1
RERANKER_OFFLINE=true
API_KEY=at-least-32-random-characters
ADMIN_API_KEY=another-32-random-characters
STORAGE_RAW=storage/raw
STORAGE_PARSED=storage/parsed
INGEST_PATH_MUST_BE_UNDER_STORAGE_RAW=true
RATE_LIMIT_PER_MINUTE=60
UVICORN_WORKERS=2
CPU_THREADS_PER_WORKER=3
VECTOR_RECALL_TOP_N=60
RERANK_TOP_N=12
RERANKER_BATCH_SIZE=8
RERANKER_MAX_LENGTH=256
HYBRID_VECTOR_WEIGHT=0.6
QUERY_EXPANSION_ENABLED=true
QUERY_SYNONYMS_BY_COLLECTION={}
TOPIC_EXPANSIONS_BY_COLLECTION={}
QUERY_SYNONYMS_BY_DOMAIN={}
TOPIC_EXPANSIONS_BY_DOMAIN={}
SEMANTIC_CHUNKING_ENABLED=true
SEMANTIC_TABLE_CHUNK_MAX_CHARS=700
SEMANTIC_FAQ_CHUNK_MAX_CHARS=900
DISABLE_MINERU_LLM=1
AUTO_CREATE_SCHEMA=false
```

## Run with Docker Postgres

```bash
docker compose up -d
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```


## Windows install (two venv strategy)

Core service environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

MinerU environment (isolated):

```powershell
python -m venv .venv-mineru
.\.venv-mineru\Scripts\Activate.ps1
pip install -r requirements-mineru.txt
# optional reproducible install
# pip install -r requirements-mineru.txt -c constraints-mineru.txt
```

`requirements-mineru.txt` intentionally contains runtime-critical dependencies (`mineru`, `torch`, `transformers`, `ultralytics`, `doclayout-yolo`, `rapid-table`, `shapely`, `pyclipper`, `dill`, `fast-langdetect`) so `.venv-mineru` is self-sufficient.


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
  },
  {
    "query": "вопрос, ответа на который нет в корпусе",
    "is_negative": true
  }
]
```

Run evaluation after reindex/model changes:

```bash
python -m scripts.run_quality_eval \
  --eval-set eval_set.json \
  --collection default \
  --top-k 10 \
  --min-mean-recall 0.85 \
  --min-mean-ndcg 0.75 \
  --min-mrr 0.75 \
  --min-negative-abstention 0.90
```

The report includes `Recall@k`, `nDCG@k`, MRR, and the false-positive/abstention rate
for negative queries. The command exits with code 2 when a quality gate fails.

The repository also contains a source/page/term-labeled CPU acceptance suite:

```bash
python -m scripts.run_acceptance_eval \
  --eval-set eval/cpu_vps_acceptance.json \
  --top-k 3 \
  --min-evidence-recall 0.90 \
  --min-evidence-ndcg 0.85 \
  --min-negative-abstention 0.90 \
  --max-p95-ms 4000
```

Hyperparameter tuning (grid search for `HYBRID_VECTOR_WEIGHT`, `VECTOR_RECALL_TOP_N`, `RERANK_TOP_N`):

```bash
python -m scripts.tune_retrieval --eval-set eval_set.json --collection default --top-k 10
```


## Load testing

Locust:

```bash
API_KEY='<YOUR_API_KEY>' locust -f scripts/loadtest/locustfile.py --host http://localhost:8000
```

k6:

```bash
API_KEY='<YOUR_API_KEY>' VUS=4 DURATION=2m k6 run scripts/loadtest/k6_retrieve.js
```

The k6 script includes threshold checks for `p95`, `p99`, and error rate.


## Retrieval quality tuning

Current defaults are tuned for better recall on long/noisy corpora:

- `VECTOR_RECALL_TOP_N=60`
- `RERANK_TOP_N=12`
- `RERANKER_MAX_LENGTH=256`
- `RAG_FINAL_TOP_K=5`
- `HYBRID_VECTOR_WEIGHT=0.6`
- `QUERY_EXPANSION_ENABLED=true`
- `QUERY_SYNONYMS_BY_COLLECTION / TOPIC_EXPANSIONS_BY_COLLECTION`
- `QUERY_SYNONYMS_BY_DOMAIN / TOPIC_EXPANSIONS_BY_DOMAIN`

How it works:

1. query is normalized (`тема урока:`/`тема:`/`урок:` prefixes are removed);
2. short topic queries and goal/cause/consequence questions are expanded into bounded search variants;
3. HNSW vector recall gets top-N candidate fragments for each variant in pgvector space;
4. PostgreSQL Russian full-text search supplies a separate lexical recall path;
5. semantic and lexical candidates are fused while preserving both paths in the bounded rerank pool;
6. at most 12 candidates, truncated to 256 tokens, are reranked by the CPU cross-encoder;
7. keyword relevance rerank applies topic marker penalties/bonuses;
8. final grounded context in `/query` is limited by `RAG_FINAL_TOP_K`.

`/retrieve` keeps expanded context in `hits[].snippet` for backward compatibility.
For orchestrators, request `return_text=true`: `hits[].text` is the exact, complete
`fragments.text` value for the selected `fragment_id`. Request `return_context=true`
to additionally receive:

- `context_text`: the bounded expanded context used around the hit;
- `context_fragments`: exact neighboring fragments with their own IDs, pages and boundaries.

Do not treat legacy `snippet` as the exact target fragment.

### PostgreSQL backups on VPS

Create and validate a logical backup after every reindex:

```bash
./scripts/backup-postgres.sh
pg_restore --list /root/rag-backups/rag-logical-*.dump >/dev/null
```

The script writes a custom-format `pg_dump`, a SHA-256 sidecar and removes only
`rag-logical-*` backup files older than `RAG_BACKUP_RETENTION_DAYS` (14 by
default). To enable the included daily 03:17 UTC schedule:

```bash
sudo install -m 0644 deploy/rag-service-backup.cron /etc/cron.d/rag-service-backup
sudo systemctl is-active cron
```

If cross-encoder fails to load, logs include `cross_encoder_unavailable`, and `/readyz` reports:
- `checks.reranker_loaded`
- `checks.reranker_model`
- `checks.reranker_error`

Query expansion is dictionary-based and configurable via env JSON:
- global defaults: `QUERY_SYNONYMS_DEFAULT`, `TOPIC_EXPANSIONS_DEFAULT`;
- collection-level overrides: `QUERY_SYNONYMS_BY_COLLECTION`, `TOPIC_EXPANSIONS_BY_COLLECTION`;
- domain-level overrides (based on `source_uris` host): `QUERY_SYNONYMS_BY_DOMAIN`, `TOPIC_EXPANSIONS_BY_DOMAIN`.

Semantic chunking can adapt chunk boundaries for table-like and FAQ-like blocks via:
- `SEMANTIC_CHUNKING_ENABLED`
- `SEMANTIC_TABLE_CHUNK_MAX_CHARS`
- `SEMANTIC_FAQ_CHUNK_MAX_CHARS`

For retrieval diagnostics, debug logs include:
- original and normalized query;
- expanded query variants;
- raw hits count, deduplicated hits, post-keyword-rerank hits;
- final selected hit count.

When reranker is unavailable during retrieval, service emits both:
- `reranker_unavailable_fallback`
- `reranker_unavailable_fallback_alert`

Use `/readyz` checks `reranker_loaded` and `reranker_error` as monitoring signals.


## VPS deployment with Docker (app + Postgres)

1. Copy env template and set secrets:

```bash
cp .env.docker.example .env.docker
# edit .env.docker: set API_KEY, ADMIN_API_KEY and the same POSTGRES_PASSWORD
# in both POSTGRES_PASSWORD and DATABASE_URL
```

2. Download the two multilingual CPU models before enabling offline mode:

```bash
python3 -m venv .model-download
.model-download/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
.model-download/bin/pip install -r requirements-core.txt
.model-download/bin/python scripts/download_cpu_models.py
rm -rf .model-download
```

3. Make writable directories accessible to the non-root container user:

```bash
mkdir -p storage/raw storage/parsed storage/models
sudo chown -R 10001:10001 storage/parsed
chmod 600 .env.docker
```

4. Build and start services:

```bash
docker compose -f docker-compose.vps.yml --env-file .env.docker up -d --build
```

5. Check status and logs:

```bash
docker compose -f docker-compose.vps.yml ps
docker compose -f docker-compose.vps.yml logs -f app
```

6. Verify liveness and readiness:

```bash
curl http://localhost:8000/livez
curl http://localhost:8000/readyz -H "X-API-Key: <YOUR_API_KEY>"
```

`/readyz` deliberately returns `503` if existing vectors were created by another embedding
model. Put every original file back into `storage/raw` and call `/ingest` with
`"reindex": true`; vectors produced by different embedding models must never be mixed.
The service is bound to `127.0.0.1:8000`; expose it only through a TLS reverse proxy.

### Change number of workers

You can change uvicorn workers without rebuilding image. For the target 8-core/16-GB VPS,
keep `UVICORN_WORKERS=2` and `CPU_THREADS_PER_WORKER=3`; four model copies usually reduce
throughput and leave too little memory for MinerU and PostgreSQL.

For CPU parsing of textbooks with hundreds of pages, keep the admin-only ingest limits
explicit: `MAX_FILE_SIZE_MB=128`, `MAX_INGEST_BATCH_MB=1000`, and
`MINERU_TIMEOUT_SECONDS=7200`. The shared VPS profile reserves 9 GB for the app container:
a large scanned textbook pushed MinerU and two API workers above 7.5 GB and caused cgroup
memory pressure at an 8 GB limit.

- edit `.env.docker` and set `UVICORN_WORKERS` (for example `4`), then restart app:

```bash
docker compose -f docker-compose.vps.yml --env-file .env.docker up -d app
```

The container command uses `--workers ${UVICORN_WORKERS:-2}`.

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
  -d '{"input_path":"/app/storage/raw","collection":"default","reindex":false}'
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

Для локального оркестратора используйте точный fragment/context-контракт:

```bat
curl -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" -H "X-API-Key: super-secret-key" -d "{\"query\":\"какие цели преследовали реформы Танзимат\",\"top_k\":5,\"min_score\":0.35,\"collection\":\"default\",\"return_text\":true,\"return_context\":true}"
```

Рекомендуемые one-line варианты для **подбора более точных фрагментов** (Windows CMD):

1) **Базовый balanced-поиск** (хорошая стартовая точка):

```bat
curl -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" -H "X-API-Key: super-secret-key" -d "{\"query\":\"тема урока: османская империя\",\"top_k\":8,\"min_score\":0.25,\"collection\":\"default\",\"return_text\":true}"
```

2) **Узкий поиск по конкретному учебнику** (`source_uris` снижает шум):

```bat
curl -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" -H "X-API-Key: super-secret-key" -d "{\"query\":\"тема урока: османская империя\",\"top_k\":8,\"min_score\":0.2,\"collection\":\"default\",\"source_uris\":[\"Russkiy_yazyk_2019.pdf\"],\"return_text\":true}"
```

3) **Широкий recall для сложной темы** (больше кандидатов под rerank):

```bat
curl -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" -H "X-API-Key: super-secret-key" -d "{\"query\":\"османская империя реформы танзим султан стамбул\",\"top_k\":15,\"min_score\":0.15,\"collection\":\"default\",\"return_text\":true}"
```

4) **Строгий режим** (если хотите меньше, но точнее):

```bat
curl -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" -H "X-API-Key: super-secret-key" -d "{\"query\":\"османская империя\",\"top_k\":5,\"min_score\":0.4,\"collection\":\"default\",\"return_text\":true}"
```

Практика: сначала используйте balanced/recall-вариант, потом ужесточайте `min_score` и/или добавляйте `source_uris`.

Ответ (пример):

```json
{"hits":[{"fragment_id":"f1","source_uri":"pub_1167883.pdf","title":"pub_1167883.pdf","type":"text","page":12,"snippet":"legacy expanded context","score":0.9123,"text":"exact complete target fragment","context_text":"bounded expanded context","context_fragments":[{"fragment_id":"f0","page":11,"element_index":10,"text":"exact neighboring fragment"},{"fragment_id":"f1","page":12,"element_index":11,"text":"exact complete target fragment"}]}]}
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
- For Russian production/offline deployments, use the local multilingual embedding/reranker paths from `.env.docker.example` and keep both offline flags enabled.
- Retrieval candidate recall uses pgvector HNSW plus a GIN-indexed Russian full-text path before a bounded cross-encoder rerank.
- The embedding fingerprint stored with every indexed document is checked by `/readyz` and every retrieval; changing `EMBED_MODEL` requires a complete reindex.
- Retrieval debug mode exposes ranking reasons such as `full_phrase_match`, `missing_required_terms`, `concept_boost_applied`, `exercise_demoted_for_concept_lookup`, `schema_or_rule_boost_applied`, and `final_rank_reason`.
- The parser uses RAG-Anything when available in runtime; if unavailable it degrades to lightweight local parsers for TXT/MD/PDF/DOCX.
- `page` remains optional in all APIs.
- In production keep `FAIL_ON_EMBEDDING_FALLBACK=true` to avoid silent hash-embedding fallback and low-quality retrieval.

## Troubleshooting dependencies
- Dependencies are pinned for reproducible Linux x86_64 builds; change a pin only after
  rebuilding the image and rerunning unit, quality and load gates.
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
- Retrieval quality is controlled by `VECTOR_RECALL_TOP_N`, `RERANK_TOP_N`, `HYBRID_VECTOR_WEIGHT`, and `QUERY_EXPANSION_ENABLED`; adjust these for your corpus size/domain.
- To inspect intent-aware ranking for a concept query:
  ```bash
  curl -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" -H "X-API-Key: super-secret-key" -d "{\"query\":\"морфологический разбор\",\"top_k\":10,\"min_score\":0.35,\"debug\":true}"
  ```
- Keep the core and MinerU environments isolated: their independently pinned Torch/Transformers
  stacks are intentionally different. Recreate the affected environment from
  `requirements-core.txt` or `requirements-mineru.txt` plus `constraints-mineru.txt`
  instead of installing ad-hoc compatibility pins into the running container.
- If an offline model is unavailable, run `scripts/download_cpu_models.py` before starting
  Compose and verify that both local directories from `.env.docker` exist.
