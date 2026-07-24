from app.chunking import split_structured_chunks
from types import SimpleNamespace

from app.quality import ndcg_at_k, recall_at_k, run_quality_eval
from app.schemas import ParsedElement
from app.service import coalesce_parsed_elements


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
    assert chunks[0].text.startswith("Раздел 1")
    assert chunks[1].heading_path == ["Раздел 1", "Подраздел"]
    assert chunks[1].text.startswith("Раздел 1 / Подраздел")


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


def test_coalesce_parsed_elements_merges_short_text_on_same_page():
    elements = [
        ParsedElement(element_index=1, type="text", content="Первый абзац.", page=1),
        ParsedElement(element_index=2, type="text", content="Второй абзац.", page=1),
        ParsedElement(element_index=3, type="table", content="A | B", page=1),
    ]

    merged = coalesce_parsed_elements(elements)

    assert len(merged) == 2
    assert merged[0].content == "Первый абзац.\nВторой абзац."
    assert merged[0].meta["merged_element_indices"] == [1, 2]
    assert merged[1].type == "table"


def test_quality_eval_excludes_negative_queries_from_recall():
    class Repository:
        def retrieve(self, query, **kwargs):
            if query == "известный вопрос":
                return [SimpleNamespace(fragment_id="f1")]
            return []

    report = run_quality_eval(
        Repository(),
        [
            {"query": "известный вопрос", "relevant_fragment_ids": ["f1"]},
            {"query": "вне корпуса", "is_negative": True},
        ],
        collection="default",
        top_k=5,
        min_score=0.2,
    )

    assert report["mean_recall_at_k"] == 1.0
    assert report["mean_reciprocal_rank"] == 1.0
    assert report["negative_abstention_rate"] == 1.0
