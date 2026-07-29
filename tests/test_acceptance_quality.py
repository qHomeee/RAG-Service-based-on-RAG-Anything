from types import SimpleNamespace

from app.quality import run_acceptance_eval


class FakeRepository:
    def retrieve(self, *, query, top_k, min_score, collection, source_uris):
        if query == "negative":
            return []
        return [
            SimpleNamespace(
                fragment_id="wrong",
                source_uri="wrong.pdf",
                page=5,
                text="unrelated",
                snippet="unrelated",
                final_score=0.81,
            ),
            SimpleNamespace(
                fragment_id="right",
                source_uri="book.pdf",
                page=10,
                text="Фонетика изучает звуки речи.",
                snippet="Фонетика изучает звуки речи.",
                final_score=0.72,
            ),
        ]


def test_acceptance_eval_scores_evidence_rank_and_negative_abstention():
    report = run_acceptance_eval(
        repository=FakeRepository(),
        eval_set=[
            {
                "query": "Что изучает фонетика?",
                "category": "russian",
                "expected_source": "book.pdf",
                "expected_terms": ["фонетик", "звук"],
            },
            {"query": "negative", "is_negative": True},
        ],
        collection="default",
        top_k=3,
        min_score=0.2,
    )

    assert report["evidence_recall_at_k"] == 1.0
    assert report["evidence_ndcg_at_k"] == 0.6309
    assert report["mean_reciprocal_rank"] == 0.5
    assert report["top1_source_accuracy"] == 0.0
    assert report["negative_abstention_rate"] == 1.0
    result = report["results"][0]
    assert result["category"] == "russian"
    assert result["retrieved_hits"][1] == {
        "fragment_id": "right",
        "source_uri": "book.pdf",
        "page": 10,
        "score": 0.72,
    }


def test_acceptance_eval_treats_pages_as_strict_labels():
    report = run_acceptance_eval(
        repository=FakeRepository(),
        eval_set=[
            {
                "query": "Что изучает фонетика?",
                "expected_source": "book.pdf",
                "expected_pages": [11],
                "expected_terms": ["фонетик", "звук"],
            }
        ],
        collection="default",
        top_k=3,
        min_score=0.2,
    )

    assert report["evidence_recall_at_k"] == 0.0


def test_acceptance_eval_requires_terms_within_labeled_page():
    report = run_acceptance_eval(
        repository=FakeRepository(),
        eval_set=[
            {
                "query": "Что изучает фонетика?",
                "expected_source": "book.pdf",
                "expected_pages": [10],
                "expected_terms": ["морфология"],
            }
        ],
        collection="default",
        top_k=3,
        min_score=0.2,
    )

    assert report["evidence_recall_at_k"] == 0.0
