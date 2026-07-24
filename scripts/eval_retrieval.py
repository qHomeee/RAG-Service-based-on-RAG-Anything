import argparse
import json
import sys
import time
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
        "query": "Морфологический разбор",
        "expected_terms": ["морфолог", "разбор"],
        "forbidden_source_terms": ["истор", "граждан", "биолог"],
    },
    {
        "query": "упражнения на морфологический разбор",
        "expected_terms": ["морфолог", "разбор", "упражнен", "задан"],
        "forbidden_source_terms": ["истор", "граждан", "биолог"],
    },
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
        "query": "синтаксический разбор предложения",
        "expected_terms": ["синтакс", "разбор", "предлож"],
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
    {
        "query": "квадратное уравнение",
        "expected_terms": ["квадрат", "уравнен", "дискриминант", "корн"],
        "forbidden_source_terms": ["истор", "русск", "граждан", "биолог"],
    },
    {
        "query": "фотосинтез",
        "expected_terms": ["фотосинтез", "хлорофилл", "растен", "кислород"],
        "forbidden_source_terms": ["истор", "русск", "граждан", "математ"],
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
    parser.add_argument("--include-toc", action="store_true")
    parser.add_argument("--include-low-quality", action="store_true")
    parser.add_argument("--include-navigation", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Print one compact JSON record per query")
    args = parser.parse_args()

    eval_set = _load_eval_set(args.eval_set)
    embeddings = EmbeddingProvider()
    reranker = CrossEncoderReranker()

    with SessionLocal() as db:
        repository = RagRepository(db=db, embeddings=embeddings, reranker=reranker)
        for spec in eval_set:
            query = spec["query"]
            started = time.perf_counter()
            result = repository.retrieve_with_debug(
                query,
                top_k=args.top_k,
                min_score=args.min_score,
                collection=args.collection,
                source_uris=args.source_uris,
                include_toc=args.include_toc,
                include_low_quality=args.include_low_quality,
                include_navigation=args.include_navigation,
                debug=True,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            if args.compact:
                print(
                    json.dumps(
                        {
                            "query": query,
                            "elapsed_ms": elapsed_ms,
                            "hits": [
                                {
                                    "fragment_id": hit.fragment_id,
                                    "source_uri": hit.source_uri,
                                    "page": hit.page,
                                    "score": round(hit.score, 4),
                                    "text": hit.text.replace("\n", " ")[:240],
                                }
                                for hit in result.hits
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            print(f"\nQUERY: {query}")
            if result.debug:
                print("QUERY_ANALYSIS:", json.dumps(result.debug.get("query_analysis"), ensure_ascii=False))
                print("SELECTED_DOCS:", json.dumps(result.debug.get("selected_documents", [])[:5], ensure_ascii=False))
                if result.debug.get("rejected_documents"):
                    print("REJECTED_DOCS:", json.dumps(result.debug.get("rejected_documents", [])[:5], ensure_ascii=False))
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
                    "phrase_score": round(hit.phrase_score, 4),
                    "phrase_score_before_penalty": round(hit.phrase_score_before_penalty, 4),
                    "phrase_score_after_penalty": round(hit.phrase_score_after_penalty, 4),
                    "exact_phrases": hit.exact_phrases,
                    "matched_phrases": hit.matched_phrases,
                    "missing_required_modifiers": hit.missing_required_modifiers,
                    "wrong_entity_modifier": hit.wrong_entity_modifier,
                    "is_toc": hit.is_toc,
                    "toc_filtered": hit.toc_filtered,
                    "toc_penalty_applied": hit.toc_penalty_applied,
                    "quality_score": round(hit.quality_score, 4),
                    "low_text_quality": hit.low_text_quality,
                    "is_fragmented": hit.is_fragmented,
                    "starts_mid_word": hit.starts_mid_word,
                    "query_type": hit.query_type,
                    "chunk_type": hit.chunk_type,
                    "chunk_type_reason": hit.chunk_type_reason,
                    "section_title": (hit.meta or {}).get("section_title"),
                    "section_title_reason": hit.section_title_reason,
                    "inferred_section_title": hit.inferred_section_title,
                    "required_terms": hit.required_terms,
                    "required_term_score": round(hit.required_term_score, 4),
                    "required_term_match_type": hit.required_term_match_type,
                    "full_phrase_match": hit.full_phrase_match,
                    "missing_required_terms": hit.missing_required_terms,
                    "is_navigation_index": hit.is_navigation_index,
                    "navigation_filtered": hit.navigation_filtered,
                    "intent_boost_applied": hit.intent_boost_applied,
                    "exercise_penalty_applied": hit.exercise_penalty_applied,
                    "test_question_penalty_applied": hit.test_question_penalty_applied,
                    "schema_boost_applied": hit.schema_boost_applied,
                    "term_penalty_applied": hit.term_penalty_applied,
                    "full_term_boost_applied": hit.full_term_boost_applied,
                    "concept_boost_applied": hit.concept_boost_applied,
                    "exercise_demoted_for_concept_lookup": hit.exercise_demoted_for_concept_lookup,
                    "schema_or_rule_boost_applied": hit.schema_or_rule_boost_applied,
                    "final_rank_reason": hit.final_rank_reason,
                    "score_before_boosts": round(hit.score_before_boosts, 4),
                    "concept_full_phrase_boost_value": round(hit.concept_full_phrase_boost_value, 4),
                    "section_title_boost_value": round(hit.section_title_boost_value, 4),
                    "schema_or_rule_boost_value": round(hit.schema_or_rule_boost_value, 4),
                    "quality_penalty_value": round(hit.quality_penalty_value, 4),
                    "boundary_penalty_value": round(hit.boundary_penalty_value, 4),
                    "exercise_penalty_value": round(hit.exercise_penalty_value, 4),
                    "score_after_boosts_before_clamp": round(hit.score_after_boosts_before_clamp, 4),
                    "expanded_from_neighbors": hit.expanded_from_neighbors,
                    "low_text_quality_reason": hit.low_text_quality_reason,
                    "penalties_applied": hit.penalties_applied,
                    "subject_score": round(hit.subject_score, 4),
                    "section_score": round(hit.section_score, 4),
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
