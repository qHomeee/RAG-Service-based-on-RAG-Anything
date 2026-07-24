# CPU RAG benchmark — 2026-07-24

Профиль: PostgreSQL corpus из 2 документов, 2 787 фрагментов, 3 095 векторов размерности
384, только CPU. Использованы локальные offline-модели:

- `paraphrase-multilingual-MiniLM-L12-v2`;
- `mmarco-mMiniLMv2-L12-H384-v1`;
- 3 Torch threads на worker, recall pool 60, rerank pool 12, reranker max length 256.

Индекс атомарно переэмбеддирован из сохранённых текстов фрагментов. До обновления были
созданы backup-таблицы с 3 095 прежними векторами и 2 документами. После обновления все
3 095 векторов и оба документа имеют fingerprint новой модели; compatibility guard зелёный.

## Quality gate

`eval/cpu_vps_acceptance.json` содержит 50 размеченных вопросов: 40 позитивных по истории
и русскому языку и 10 вопросов по отсутствующим темам.

| Метрика | Результат |
|---|---:|
| Evidence Recall@3 | 1,0000 |
| Evidence nDCG@3 | 0,9658 |
| MRR@3 | 0,9542 |
| Top-1 source accuracy | 1,0000 |
| Negative abstention | 1,0000 (10/10) |
| Однопроцессный mean latency | 1,387 s |
| Однопроцессный p50 | 1,508 s |
| Однопроцессный p95 | 2,110 s |
| Однопроцессный max | 2,224 s |

Релевантность размечена ожидаемым source, диапазоном страниц и существенными термами.
Это воспроизводимый regression/acceptance set для текущего корпуса, а не универсальная
оценка качества на любом наборе документов.

## Два API worker / четыре параллельных клиента

Локальный HTTP-тест запускал Uvicorn с двумя отдельными worker, по 3 CPU threads на worker:
24 запроса, concurrency 4, 21 позитивный и 3 OOD-запроса.

| Метрика | Результат |
|---|---:|
| HTTP 200 | 24/24 |
| Корректные пустые OOD-ответы | 3/3 |
| Throughput | 1,197 req/s |
| Mean latency | 3,177 s |
| p50 | 3,274 s |
| p95 | 3,796 s |
| Max | 3,929 s |
| RSS всего process tree | 2 595 MB |

Это измерение CPU-профиля на локальной машине. На целевом VPS после deploy необходимо
повторить двухминутный k6-тест; начальный production gate — p95 ≤ 5 s, 5xx < 1%.

## Что дало улучшение

- multilingual embedding index вместо англоязычно-ориентированного legacy index;
- HNSW vector recall и generated Russian `tsvector` + GIN;
- document routing без сканов всех embeddings;
- bounded rerank pool 12 и длина 256 токенов;
- фильтрация поздних оглавлений и navigation fragments;
- сохранение исходного `section_title` после context expansion;
- theoretical blocks с нумерованными примерами больше не считаются упражнениями;
- fail-closed routing для известных отсутствующих предметов и строгий downstream
  lexical/phrase noise filter для неизвестных OOD-запросов.
