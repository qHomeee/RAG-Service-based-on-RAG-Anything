import json
import math
import time
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


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, fragment_id in enumerate(retrieved[:k], start=1):
        if fragment_id in relevant:
            return 1.0 / rank
    return 0.0


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
    reciprocal_ranks: list[float] = []
    negative_results: list[bool] = []

    for item in eval_set:
        query = item["query"]
        relevant_ids = set(item.get("relevant_fragment_ids", []))
        graded = item.get("graded_relevance", {rid: 1.0 for rid in relevant_ids})
        is_negative = bool(item.get("is_negative", False))
        if not is_negative and not relevant_ids:
            raise ValueError(f"Positive eval query has no relevant_fragment_ids: {query}")

        rows = repository.retrieve(query=query, top_k=top_k, min_score=min_score, collection=collection, source_uris=None)
        retrieved_ids = [row.fragment_id for row in rows]

        if is_negative:
            correct_abstention = not retrieved_ids
            negative_results.append(correct_abstention)
            results.append(
                {
                    "query": query,
                    "is_negative": True,
                    "correct_abstention": correct_abstention,
                    "retrieved_fragment_ids": retrieved_ids,
                    "relevant_fragment_ids": [],
                }
            )
            continue

        r_at_k = recall_at_k(retrieved_ids, relevant_ids, top_k)
        n_at_k = ndcg_at_k(retrieved_ids, graded, top_k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids, top_k)
        recall_scores.append(r_at_k)
        ndcg_scores.append(n_at_k)
        reciprocal_ranks.append(rr)

        results.append(
            {
                "query": query,
                "is_negative": False,
                "recall_at_k": round(r_at_k, 4),
                "ndcg_at_k": round(n_at_k, 4),
                "reciprocal_rank": round(rr, 4),
                "retrieved_fragment_ids": retrieved_ids,
                "relevant_fragment_ids": sorted(relevant_ids),
            }
        )

    return {
        "queries_count": len(eval_set),
        "positive_queries_count": len(recall_scores),
        "negative_queries_count": len(negative_results),
        "mean_recall_at_k": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0,
        "mean_ndcg_at_k": round(sum(ndcg_scores) / len(ndcg_scores), 4) if ndcg_scores else 0.0,
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)
        if reciprocal_ranks
        else 0.0,
        "negative_abstention_rate": round(sum(negative_results) / len(negative_results), 4)
        if negative_results
        else None,
        "negative_false_positive_rate": round(
            1.0 - (sum(negative_results) / len(negative_results)),
            4,
        )
        if negative_results
        else None,
        "results": results,
    }


def run_acceptance_eval(
    repository: RagRepository,
    eval_set: list[dict],
    collection: str,
    top_k: int,
    min_score: float,
) -> dict:
    results: list[dict] = []
    positive_matches: list[bool] = []
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    top1_source_matches: list[bool] = []
    negative_results: list[bool] = []
    latencies_ms: list[float] = []

    for item in eval_set:
        query = str(item["query"])
        is_negative = bool(item.get("is_negative", False))
        expected_source = item.get("expected_source")
        expected_terms = [str(term).casefold() for term in item.get("expected_terms", [])]
        expected_pages = {int(page) for page in item.get("expected_pages", [])}
        if not is_negative and (not expected_source or (not expected_terms and not expected_pages)):
            raise ValueError(
                "Positive acceptance query requires expected_source and evidence labels: "
                f"{query}"
            )

        started = time.perf_counter()
        rows = repository.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            collection=collection,
            source_uris=None,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(elapsed_ms)

        if is_negative:
            correct_abstention = not rows
            negative_results.append(correct_abstention)
            results.append(
                {
                    "query": query,
                    "category": item.get("category"),
                    "is_negative": True,
                    "correct_abstention": correct_abstention,
                    "retrieved_fragment_ids": [row.fragment_id for row in rows],
                    "retrieved_hits": [_acceptance_hit_payload(row) for row in rows],
                    "elapsed_ms": round(elapsed_ms, 2),
                }
            )
            continue

        relevant_ranks = [
            rank
            for rank, row in enumerate(rows, start=1)
            if _matches_acceptance_evidence(
                row,
                expected_source=str(expected_source),
                expected_terms=expected_terms,
                expected_pages=expected_pages,
            )
        ]
        first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
        matched = first_relevant_rank is not None
        reciprocal_rank_score = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        ndcg_score = 1.0 / math.log2(first_relevant_rank + 1) if first_relevant_rank else 0.0
        top1_source_match = bool(rows and rows[0].source_uri == expected_source)

        positive_matches.append(matched)
        reciprocal_ranks.append(reciprocal_rank_score)
        ndcg_scores.append(ndcg_score)
        top1_source_matches.append(top1_source_match)
        results.append(
            {
                "query": query,
                "category": item.get("category"),
                "is_negative": False,
                "matched": matched,
                "first_relevant_rank": first_relevant_rank,
                "reciprocal_rank": round(reciprocal_rank_score, 4),
                "ndcg_at_k": round(ndcg_score, 4),
                "top1_source_match": top1_source_match,
                "retrieved_fragment_ids": [row.fragment_id for row in rows],
                "retrieved_hits": [_acceptance_hit_payload(row) for row in rows],
                "elapsed_ms": round(elapsed_ms, 2),
            }
        )

    latency_sorted = sorted(latencies_ms)
    return {
        "queries_count": len(eval_set),
        "positive_queries_count": len(positive_matches),
        "negative_queries_count": len(negative_results),
        "evidence_recall_at_k": _mean_booleans(positive_matches),
        "evidence_ndcg_at_k": _mean_numbers(ndcg_scores),
        "mean_reciprocal_rank": _mean_numbers(reciprocal_ranks),
        "top1_source_accuracy": _mean_booleans(top1_source_matches),
        "negative_abstention_rate": _mean_booleans(negative_results)
        if negative_results
        else None,
        "latency_ms": {
            "mean": round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0,
            "p50": round(_percentile(latency_sorted, 0.50), 2),
            "p95": round(_percentile(latency_sorted, 0.95), 2),
            "max": round(max(latencies_ms), 2) if latencies_ms else 0.0,
        },
        "results": results,
    }


def _matches_acceptance_evidence(
    row,
    *,
    expected_source: str,
    expected_terms: list[str],
    expected_pages: set[int],
) -> bool:
    if row.source_uri != expected_source:
        return False
    if expected_pages:
        return getattr(row, "page", None) in expected_pages
    haystack = f"{getattr(row, 'text', '')} {getattr(row, 'snippet', '')}".casefold()
    return bool(expected_terms) and all(term in haystack for term in expected_terms)


def _acceptance_hit_payload(row) -> dict:
    score = getattr(row, "final_score", None)
    if score is None:
        score = getattr(row, "score", None)
    return {
        "fragment_id": row.fragment_id,
        "source_uri": row.source_uri,
        "page": getattr(row, "page", None),
        "score": round(float(score), 4) if score is not None else None,
    }


def _mean_booleans(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _mean_numbers(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[index]
