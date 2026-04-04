from app.chunking import split_structured_chunks
from app.quality import ndcg_at_k, recall_at_k


def test_split_structured_chunks_preserves_heading_path():
    text = """
# Раздел 1

Первый абзац. Второй текст.

## Подраздел

Третий абзац с терминами.
"""
    chunks = split_structured_chunks(text, min_size=10, max_size=120)
    assert len(chunks) == 2
    assert chunks[0].heading_path == ["Раздел 1"]
    assert chunks[1].heading_path == ["Раздел 1", "Подраздел"]


def test_recall_and_ndcg_metrics():
    retrieved = ["f1", "f2", "f3"]
    relevant = {"f2", "f4"}
    graded = {"f2": 2.0, "f4": 1.0}

    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert ndcg_at_k(retrieved, graded, 3) > 0


def test_semantic_chunking_splits_table_like_blocks_more_aggressively():
    table_text = """
# Данные

| Год | Значение |
| --- | --- |
| 2020 | 10 |
| 2021 | 20 |
| 2022 | 30 |
| 2023 | 40 |
| 2024 | 50 |
"""
    chunks = split_structured_chunks(table_text, min_size=300, max_size=1200)
    assert len(chunks) >= 1
    assert all(len(chunk.text) <= 700 for chunk in chunks)
