import json
import math
from pathlib import Path

from app.repository import RagRepository


def load_eval_set(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evaluation set must be a JSON list")
    return data


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    top = set(retrieved[:k])
    return len(top.intersection(relevant)) / len(relevant)


def ndcg_at_k(retrieved: list[str], graded_relevance: dict[str, float], k: int) -> float:
    def dcg(items: list[str]) -> float:
        total = 0.0
        for idx, item in enumerate(items[:k], start=1):
            rel = graded_relevance.get(item, 0.0)
            total += (2**rel - 1) / math.log2(idx + 1)
        return total

    ideal = [item for item, _ in sorted(graded_relevance.items(), key=lambda x: x[1], reverse=True)]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(retrieved) / ideal_dcg


def run_quality_eval(
    repository: RagRepository,
    eval_set: list[dict],
    collection: str,
    top_k: int,
    min_score: float,
) -> dict:
    results: list[dict] = []
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []

    for item in eval_set:
        query = item["query"]
        relevant_ids = set(item.get("relevant_fragment_ids", []))
        graded = item.get("graded_relevance", {rid: 1.0 for rid in relevant_ids})

        rows = repository.retrieve(query=query, top_k=top_k, min_score=min_score, collection=collection, source_uris=None)
        retrieved_ids = [row.fragment_id for row in rows]

        r_at_k = recall_at_k(retrieved_ids, relevant_ids, top_k)
        n_at_k = ndcg_at_k(retrieved_ids, graded, top_k)
        recall_scores.append(r_at_k)
        ndcg_scores.append(n_at_k)

        results.append(
            {
                "query": query,
                "recall_at_k": round(r_at_k, 4),
                "ndcg_at_k": round(n_at_k, 4),
                "retrieved_fragment_ids": retrieved_ids,
                "relevant_fragment_ids": sorted(relevant_ids),
            }
        )

    return {
        "queries_count": len(eval_set),
        "mean_recall_at_k": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0,
        "mean_ndcg_at_k": round(sum(ndcg_scores) / len(ndcg_scores), 4) if ndcg_scores else 0.0,
        "results": results,
    }
