# Production на VPS: 8 CPU / 16 GB RAM / без GPU

Этот профиль рассчитан на один VPS. FastAPI и обе ML-модели работают только на CPU;
PostgreSQL, Redis и MinerU запускаются в том же Docker Compose.

## Распределение ресурсов

| Компонент | CPU | RAM limit | Назначение |
|---|---:|---:|---|
| FastAPI + модели + MinerU subprocess | 6 | 10 GB | retrieval и последовательный ingest |
| PostgreSQL + pgvector | 1.5 | 3 GB | HNSW, GIN FTS, метаданные |
| Redis | 0.5 | 256 MB | общий rate limit для двух workers |

В API используются 2 workers по 3 Torch-потока. Это оставляет место ОС и не создаёт
восемь конкурирующих копий моделей. Одновременно допускается только один ingest:
PostgreSQL advisory lock возвращает `409`, если другой ingest уже выполняется.

## Первый запуск

```bash
cp .env.docker.example .env.docker
python3 -m venv .model-download
.model-download/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
.model-download/bin/pip install -r requirements-core.txt
.model-download/bin/python scripts/download_cpu_models.py

mkdir -p storage/raw storage/parsed storage/models
sudo chown -R 10001:10001 storage/parsed
chmod 600 .env.docker

docker compose --env-file .env.docker -f docker-compose.vps.yml config --quiet
docker compose --env-file .env.docker -f docker-compose.vps.yml up -d --build
```

В `.env.docker` обязательно замените три placeholder-секрета. Значение пароля в
`POSTGRES_PASSWORD` и в `DATABASE_URL` должно совпадать. Модели не включаются в Docker
image и монтируются read-only из `storage/models`.

Проверка:

```bash
curl http://127.0.0.1:8000/livez
curl http://127.0.0.1:8000/readyz -H "X-API-Key: $API_KEY"
docker compose --env-file .env.docker -f docker-compose.vps.yml ps
docker compose --env-file .env.docker -f docker-compose.vps.yml logs --tail=200 app
```

`/livez` проверяет только процесс. `/readyz` проверяет PostgreSQL, pgvector, embedding,
reranker и совместимость embedding-модели с уже созданными векторами.

## Индексация и смена модели

Сначала скопируйте оригиналы в `storage/raw`, затем:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  -d '{"input_path":"/app/storage/raw","collection":"default","reindex":true}'
```

Для больших школьных учебников на CPU задайте в `.env.docker`
`MAX_FILE_SIZE_MB=128` и `MINERU_TIMEOUT_SECONDS=7200`. `/ingest` защищён отдельным
admin-ключом, поэтому этот лимит не расширяет публичный retrieval API. В shared-профиле
контейнер приложения ограничен 9 GB: большой сканированный учебник превысил 7.5 GB вместе
с двумя API workers и вызвал cgroup memory pressure при лимите 8 GB. Таймаут передаётся
непосредственно в MinerU subprocess; при его превышении дочерний процесс завершается,
а ingest возвращает ошибку вместо зависшего worker. Предел `memswap_limit: 13g` явно
разрешает до 4 GB swap сверх лимита RAM и не зависит от неявной Docker-политики swap.

Перед MinerU сервис проверяет текстовый слой PDF по выборке страниц. Для скана он запускает
CPU OCRmyPDF/Tesseract с языками `rus+eng` и сохраняет `parse_mode=ocrmypdf`; для PDF с
текстовым слоем остаётся MinerU. Это обязательно для русскоязычных сканов: MinerU 2.6 не
имеет русского OCR-профиля. На VPS с 8 CPU используется `OCR_JOBS=4`; не поднимайте значение
без замера RAM и влияния на retrieval API.

Нельзя выполнять поиск новой embedding-моделью по векторам старой модели, даже если у
обеих размерность 384. Сервис хранит fingerprint модели и в таком случае возвращает
`503`, пока коллекция не переиндексирована.

Если меняется только embedding-модель, а сохранённые фрагменты корректны, исходные PDF не
обязательны. Выполните атомарную переэмбеддингацию существующих фрагментов:

```bash
docker compose --env-file .env.docker -f docker-compose.vps.yml exec app \
  python -m scripts.reembed_existing --collection default
docker compose --env-file .env.docker -f docker-compose.vps.yml exec app \
  python -m scripts.reembed_existing \
  --collection default \
  --backup-prefix reembed_backup_20260724_multi \
  --apply
```

Первая команда — dry-run. Вторая берёт тот же advisory lock, что и `/ingest`, рассчитывает
векторы пакетами на CPU, внутри одной транзакции создаёт backup-таблицы и обновляет векторы,
summary embeddings и fingerprint. При ошибке транзакция откатывается. Явный rollback:

```bash
docker compose --env-file .env.docker -f docker-compose.vps.yml exec app \
  python -m scripts.reembed_existing \
  --restore-prefix reembed_backup_20260724_multi
```

Полный `/ingest` с `reindex=true` всё ещё нужен при изменении parser/chunking или когда
фрагменты требуется построить заново.

## Reverse proxy и TLS

Контейнер слушает только `127.0.0.1:8000`. Пример Nginx находится в
`deploy/nginx/rag.conf.example`. Выпустите сертификат через ACME/Certbot, разрешите наружу
только 80/443 и не публикуйте порты PostgreSQL/Redis.

API-ключи передаются в заголовках. Не помещайте их в URL, access log или клиентский
JavaScript. `/metrics/prometheus` тоже требует `X-API-Key`.

## SLO и проверки перед релизом

Рекомендуемый начальный SLO для этого VPS:

- retrieval p95 ≤ 5 секунд при 4 одновременных virtual users;
- p99 ≤ 10 секунд;
- 5xx < 1%;
- Recall@10 ≥ 0.85, nDCG@10 ≥ 0.75, MRR ≥ 0.75;
- корректный отказ на ≥ 90% отрицательных запросов.

```bash
API_KEY="$API_KEY" VUS=4 DURATION=2m \
  k6 run scripts/loadtest/k6_retrieve.js

python -m scripts.run_quality_eval \
  --eval-set eval_set.json \
  --collection default \
  --top-k 10 \
  --min-mean-recall 0.85 \
  --min-mean-ndcg 0.75 \
  --min-mrr 0.75 \
  --min-negative-abstention 0.90

python -m scripts.run_acceptance_eval \
  --eval-set eval/cpu_vps_acceptance.json \
  --top-k 3 \
  --min-evidence-recall 0.90 \
  --min-evidence-ndcg 0.85 \
  --min-negative-abstention 0.90 \
  --max-p95-ms 4000
```

Не утверждайте качество production только по smoke-запросам. Golden set должен содержать
50–100 реальных вопросов, релевантные `fragment_id`, сложные перефразировки и вопросы вне
корпуса. Прогон обязателен после смены модели, chunking или весов retrieval.

## Мониторинг и резервное копирование

Собирайте:

- `/metrics/prometheus`: latency, HTTP status, размер выдачи;
- `/readyz`: деградация модели или несовместимый индекс;
- Docker/container RSS, CPU throttling, free disk;
- логи `reranker_unavailable_fallback_alert`, `parser_fallback_ratio_alert`,
  `redis_rate_limiter_unavailable`.

Ежедневно сохраняйте `pg_dump` и каталоги `storage/raw`, `storage/parsed`,
`storage/models`; периодически проверяйте восстановление на отдельной БД. Не считайте
Docker volume резервной копией.

## Обновление

1. Сделать backup.
2. Запустить `pytest`.
3. Собрать image и проверить `docker compose ... config --quiet`.
4. Применить миграции через сервис `migrate`.
5. Проверить `/readyz`, smoke retrieval, затем quality/load gates.
6. При смене embedding-модели выполнить атомарный `reembed_existing`; полный reindex нужен
   при смене parser/chunking.
