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
    assert report["categories"]["russian"] == {
        "queries_count": 1,
        "positive_queries_count": 1,
        "negative_queries_count": 0,
        "evidence_recall_at_k": 1.0,
        "evidence_ndcg_at_k": 0.6309,
        "mean_reciprocal_rank": 0.5,
        "top1_source_accuracy": 0.0,
        "negative_abstention_rate": None,
        "latency_ms": report["categories"]["russian"]["latency_ms"],
    }
    assert report["categories"]["uncategorized"]["queries_count"] == 1
    assert report["categories"]["uncategorized"]["negative_abstention_rate"] == 1.0
    assert report["categories"]["uncategorized"]["evidence_recall_at_k"] is None
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


def test_acceptance_eval_checks_the_returned_full_fragment_not_neighbor_context():
    class ExpandedRepository:
        def retrieve(self, **kwargs):
            return [
                SimpleNamespace(
                    fragment_id="right-fragment",
                    source_uri="book.pdf",
                    page=12,
                    text="Соседний контекст без ответа.",
                    fragment_text=(
                        "Выбор знака препинания зависит от смысловых отношений "
                        "между частями бессоюзного предложения."
                    ),
                    snippet="Выбор знака препинания",
                    final_score=0.7,
                )
            ]

    report = run_acceptance_eval(
        repository=ExpandedRepository(),
        eval_set=[
            {
                "query": "От чего зависит выбор знака?",
                "expected_source": "book.pdf",
                "expected_pages": [12],
                "expected_terms": ["знак", "смыслов"],
            }
        ],
        collection="default",
        top_k=5,
        min_score=0.2,
    )

    assert report["evidence_recall_at_k"] == 1.0


def test_acceptance_eval_checks_structured_exact_neighbor_fragments_with_own_pages():
    class ContextRepository:
        def retrieve(self, **kwargs):
            return [
                SimpleNamespace(
                    fragment_id="anchor",
                    source_uri="book.pdf",
                    page=159,
                    text="Opaque expanded context must not define the evidence page.",
                    fragment_text="Anchor text on page 159.",
                    snippet="legacy context",
                    meta={
                        "context_fragments": [
                            {
                                "fragment_id": "exact-neighbor",
                                "page": 160,
                                "element_index": 10,
                                "text": (
                                    "В Поволжье металлургические предприятия "
                                    "сосредоточены в Волгоградской и Саратовской областях."
                                ),
                            }
                        ]
                    },
                    final_score=0.7,
                )
            ]

    report = run_acceptance_eval(
        repository=ContextRepository(),
        eval_set=[
            {
                "query": "Где предприятия Поволжья?",
                "expected_source": "book.pdf",
                "expected_pages": [160],
                "expected_terms": ["поволж", "металлург"],
            }
        ],
        collection="default",
        top_k=5,
        min_score=0.2,
    )

    assert report["evidence_recall_at_k"] == 1.0
