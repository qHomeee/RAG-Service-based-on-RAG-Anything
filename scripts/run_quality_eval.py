import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import SessionLocal
from app.embeddings import EmbeddingProvider
from app.quality import load_eval_set, run_quality_eval
from app.reranker import CrossEncoderReranker
from app.repository import RagRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval quality evaluation (Recall@k / nDCG@k)")
    parser.add_argument("--eval-set", required=True, help="Path to JSON file with 50-100 reference queries")
    parser.add_argument("--collection", default="default")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--min-mean-recall", type=float, default=0.0)
    parser.add_argument("--min-mean-ndcg", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    parser.add_argument("--min-negative-abstention", type=float, default=0.0)
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    db = SessionLocal()
    try:
        repo = RagRepository(db=db, embeddings=EmbeddingProvider(), reranker=CrossEncoderReranker())
        report = run_quality_eval(
            repository=repo,
            eval_set=eval_set,
            collection=args.collection,
            top_k=args.top_k,
            min_score=args.min_score,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        failures = []
        for metric, threshold in (
            ("mean_recall_at_k", args.min_mean_recall),
            ("mean_ndcg_at_k", args.min_mean_ndcg),
            ("mean_reciprocal_rank", args.min_mrr),
        ):
            if float(report[metric]) < threshold:
                failures.append(f"{metric}={report[metric]} < {threshold}")
        abstention = report.get("negative_abstention_rate")
        if abstention is not None and float(abstention) < args.min_negative_abstention:
            failures.append(
                f"negative_abstention_rate={abstention} < {args.min_negative_abstention}"
            )
        if failures:
            print("Quality gate failed: " + "; ".join(failures), file=sys.stderr)
            raise SystemExit(2)
    finally:
        db.close()


if __name__ == "__main__":
    main()
