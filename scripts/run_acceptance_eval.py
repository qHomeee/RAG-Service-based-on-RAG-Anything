import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import SessionLocal
from app.embeddings import EmbeddingProvider
from app.quality import load_eval_set, run_acceptance_eval
from app.reranker import CrossEncoderReranker
from app.repository import RagRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run source-and-evidence retrieval acceptance gates."
    )
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--collection", default="default")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--min-evidence-recall", type=float, default=0.9)
    parser.add_argument("--min-evidence-ndcg", type=float, default=0.85)
    parser.add_argument("--min-negative-abstention", type=float, default=0.9)
    parser.add_argument("--max-p95-ms", type=float, default=8000)
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    with SessionLocal() as db:
        repository = RagRepository(
            db=db,
            embeddings=EmbeddingProvider(),
            reranker=CrossEncoderReranker(),
        )
        report = run_acceptance_eval(
            repository=repository,
            eval_set=eval_set,
            collection=args.collection,
            top_k=args.top_k,
            min_score=args.min_score,
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    failures = []
    for metric, threshold in (
        ("evidence_recall_at_k", args.min_evidence_recall),
        ("evidence_ndcg_at_k", args.min_evidence_ndcg),
        ("negative_abstention_rate", args.min_negative_abstention),
    ):
        value = report.get(metric)
        if value is not None and float(value) < threshold:
            failures.append(f"{metric}={value} < {threshold}")
    p95_ms = float(report["latency_ms"]["p95"])
    if p95_ms > args.max_p95_ms:
        failures.append(f"latency.p95={p95_ms}ms > {args.max_p95_ms}ms")
    if failures:
        print("Acceptance gate failed: " + "; ".join(failures), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
