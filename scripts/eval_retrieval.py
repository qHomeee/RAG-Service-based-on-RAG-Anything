import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.db import SessionLocal
from app.embeddings import EmbeddingProvider
from app.repository import RagRepository, lexical_overlap, _query_terms_for_scoring
from app.reranker import CrossEncoderReranker


DEFAULT_EVAL_SET = [
    {
        "query": "Александр I и его правление",
        "expected_terms": ["александр", "правление", "реформ", "сперанск", "1812"],
        "forbidden_source_terms": ["граждан", "русск", "одуван", "техник"],
    },
    {
        "query": "Отечественная война 1812 года",
        "expected_terms": ["1812", "войн", "отечествен"],
        "forbidden_source_terms": ["русск", "граждан", "одуван"],
    },
    {
        "query": "реформы Сперанского",
        "expected_terms": ["реформ", "сперанск"],
        "forbidden_source_terms": ["русск", "граждан", "одуван"],
    },
    {
        "query": "правописание Н и НН в причастиях",
        "expected_terms": ["правопис", "нн", "причаст"],
        "forbidden_source_terms": ["истор", "граждан", "одуван"],
    },
    {
        "query": "виды придаточных предложений",
        "expected_terms": ["придаточ", "предлож"],
        "forbidden_source_terms": ["истор", "граждан", "одуван"],
    },
    {
        "query": "АСДНР при чрезвычайных ситуациях",
        "expected_terms": ["асднр", "чрезвычай", "ситуац"],
        "forbidden_source_terms": ["истор", "русск", "одуван"],
    },
    {
        "query": "средства гражданской обороны",
        "expected_terms": ["граждан", "оборон", "средств"],
        "forbidden_source_terms": ["истор", "русск", "одуван"],
    },
]


def _load_eval_set(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_EVAL_SET
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _warn_if_suspicious(spec: dict[str, Any], hit: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    source = (hit.get("source_uri") or "").lower()
    text = (hit.get("text") or hit.get("snippet") or "").lower()
    expected_terms = [str(term).lower() for term in spec.get("expected_terms", [])]
    forbidden_source_terms = [str(term).lower() for term in spec.get("forbidden_source_terms", [])]

    if expected_terms and not any(term in text or term in source for term in expected_terms):
        warnings.append("no_expected_terms")
    if forbidden_source_terms and any(term in source for term in forbidden_source_terms):
        warnings.append("forbidden_source_term")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline retrieval smoke/eval runner.")
    parser.add_argument("--eval-set", help="JSON file with query specs", default=None)
    parser.add_argument("--collection", default="default")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--min-score", type=float, default=settings.default_min_score)
    parser.add_argument("--source-uri", action="append", dest="source_uris", default=None)
    args = parser.parse_args()

    eval_set = _load_eval_set(args.eval_set)
    embeddings = EmbeddingProvider()
    reranker = CrossEncoderReranker()

    with SessionLocal() as db:
        repository = RagRepository(db=db, embeddings=embeddings, reranker=reranker)
        for spec in eval_set:
            query = spec["query"]
            result = repository.retrieve_with_debug(
                query,
                top_k=args.top_k,
                min_score=args.min_score,
                collection=args.collection,
                source_uris=args.source_uris,
                debug=True,
            )
            print(f"\nQUERY: {query}")
            print(
                "COUNTS:",
                {
                    "dense": result.debug.get("dense_candidates") if result.debug else None,
                    "lexical": result.debug.get("lexical_candidates") if result.debug else None,
                    "fusion": result.debug.get("candidates_after_fusion") if result.debug else None,
                    "threshold": result.debug.get("candidates_after_threshold") if result.debug else None,
                    "final": len(result.hits),
                },
            )
            for idx, hit in enumerate(result.hits, start=1):
                hit_payload = {
                    "source_uri": hit.source_uri,
                    "page": hit.page,
                    "score": round(hit.score, 4),
                    "dense_score": round(hit.dense_score, 4),
                    "lexical_score": round(hit.lexical_score, 4),
                    "rerank_score": round(hit.rerank_score, 4) if hit.rerank_score is not None else None,
                    "lexical_overlap": round(lexical_overlap(_query_terms_for_scoring(query), hit.text), 4),
                }
                warnings = _warn_if_suspicious(spec, {"source_uri": hit.source_uri, "text": hit.text})
                warning_text = f" WARNING={','.join(warnings)}" if warnings else ""
                print(f"{idx}. {json.dumps(hit_payload, ensure_ascii=False)}{warning_text}")

            if result.debug and result.debug.get("rejected_results"):
                rejected = result.debug["rejected_results"][:5]
                print("REJECTED_SAMPLE:", json.dumps(rejected, ensure_ascii=False))


if __name__ == "__main__":
    main()
