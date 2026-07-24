import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.db import SessionLocal
from app.embeddings import EmbeddingProvider
from app.quality import load_eval_set, run_quality_eval
from app.reranker import CrossEncoderReranker
from app.repository import RagRepository


def _parse_float_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_grid(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search retrieval hyperparams using Recall@k + nDCG@k")
    parser.add_argument("--eval-set", required=True, help="Path to JSON evaluation set")
    parser.add_argument("--collection", default="default")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--hybrid-vector-weight-grid", default="0.4,0.6,0.75")
    parser.add_argument("--vector-recall-top-n-grid", default="80,120,180")
    parser.add_argument("--rerank-top-n-grid", default="20,40,60")
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)

    weight_grid = _parse_float_grid(args.hybrid_vector_weight_grid)
    recall_grid = _parse_int_grid(args.vector_recall_top_n_grid)
    rerank_grid = _parse_int_grid(args.rerank_top_n_grid)

    original = {
        "hybrid_vector_weight": settings.hybrid_vector_weight,
        "vector_recall_top_n": settings.vector_recall_top_n,
        "rerank_top_n": settings.rerank_top_n,
    }

    db = SessionLocal()
    try:
        repo = RagRepository(db=db, embeddings=EmbeddingProvider(), reranker=CrossEncoderReranker())
        results: list[dict] = []
        for hybrid_weight, recall_top_n, rerank_top_n in itertools.product(weight_grid, recall_grid, rerank_grid):
            settings.hybrid_vector_weight = hybrid_weight
            settings.vector_recall_top_n = recall_top_n
            settings.rerank_top_n = rerank_top_n

            report = run_quality_eval(
                repository=repo,
                eval_set=eval_set,
                collection=args.collection,
                top_k=args.top_k,
                min_score=args.min_score,
            )
            score = 0.5 * float(report["mean_recall_at_k"]) + 0.5 * float(report["mean_ndcg_at_k"])
            results.append(
                {
                    "hybrid_vector_weight": hybrid_weight,
                    "vector_recall_top_n": recall_top_n,
                    "rerank_top_n": rerank_top_n,
                    "mean_recall_at_k": report["mean_recall_at_k"],
                    "mean_ndcg_at_k": report["mean_ndcg_at_k"],
                    "combined_score": round(score, 6),
                }
            )

        results.sort(key=lambda item: item["combined_score"], reverse=True)
        payload = {"best": results[0] if results else None, "runs": results}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        settings.hybrid_vector_weight = original["hybrid_vector_weight"]
        settings.vector_recall_top_n = original["vector_recall_top_n"]
        settings.rerank_top_n = original["rerank_top_n"]
        db.close()


if __name__ == "__main__":
    main()
