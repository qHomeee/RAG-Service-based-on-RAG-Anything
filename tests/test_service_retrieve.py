from app.repository import RetrievalRow
from app.service import RagService


class _FakeParser:
    pass


class _FakeRepository:
    def retrieve(
        self,
        query: str,
        top_k: int,
        min_score: float,
        collection: str,
        source_uris: list[str] | None,
        include_toc: bool = False,
        include_low_quality: bool = False,
        include_navigation: bool = False,
    ):
        return [
            RetrievalRow(
                fragment_id="f1",
                source_uri="doc.md",
                title="doc.md",
                type="text",
                page=1,
                snippet="short preview",
                score=0.95,
                text="full fragment text that should be returned in API snippet",
                dense_score=0.8,
                lexical_score=0.7,
                phrase_score=0.65,
                subject_score=0.9,
                section_score=0.4,
                rerank_score=0.6,
                final_score=0.75,
                lexical_overlap=0.5,
                document_score=0.9,
                rrf_score=0.4,
                exact_phrases=["query 1"],
                matched_phrases=["query 1"],
                missing_required_modifiers=[],
                wrong_entity_modifier=False,
                phrase_score_before_penalty=0.8,
                phrase_score_after_penalty=0.65,
                is_toc=False,
                toc_filtered=False,
                toc_penalty_applied=False,
                quality_score=1.0,
                low_text_quality=False,
                low_text_quality_reason=None,
                query_type="concept_lookup",
                chunk_type="explanatory",
                chunk_type_reason="connected_prose",
                section_title_reason="meta:section_title",
                inferred_section_title="query 1",
                required_terms=["query 1"],
                required_term_score=1.0,
                required_term_match_type="full_phrase_prefix",
                full_phrase_match=True,
                missing_required_terms=[],
                is_navigation_index=False,
                navigation_filtered=False,
                intent_boost_applied=True,
                exercise_penalty_applied=False,
                test_question_penalty_applied=False,
                schema_boost_applied=False,
                term_penalty_applied=False,
                full_term_boost_applied=True,
                concept_boost_applied=True,
                exercise_demoted_for_concept_lookup=False,
                schema_or_rule_boost_applied=True,
                final_rank_reason="concept_full_phrase_reference_boost",
                score_before_boosts=0.72,
                concept_full_phrase_boost_value=0.05,
                section_title_boost_value=0.02,
                schema_or_rule_boost_value=0.03,
                quality_penalty_value=0.0,
                boundary_penalty_value=0.0,
                exercise_penalty_value=0.0,
                score_after_boosts_before_clamp=0.82,
                expanded_from_neighbors=True,
                penalties_applied=[],
            )
        ]


def test_retrieve_returns_full_fragment_in_snippet_field():
    service = RagService(parser=_FakeParser(), repository=_FakeRepository())
    hits = service.retrieve("query", 5, 0.2, "default", None, return_text=False)
    assert hits[0]["snippet"] == "full fragment text that should be returned in API snippet"
    assert hits[0]["text"] is None
    assert hits[0]["score"] == 0.75
    assert hits[0]["dense_score"] == 0.8
    assert hits[0]["lexical_score"] == 0.7
    assert hits[0]["phrase_score"] == 0.65
    assert hits[0]["subject_score"] == 0.9
    assert hits[0]["section_score"] == 0.4
    assert hits[0]["rerank_score"] == 0.6
    assert hits[0]["final_score"] == 0.75
    assert hits[0]["lexical_overlap"] == 0.5
    assert hits[0]["document_score"] == 0.9
    assert hits[0]["rrf_score"] == 0.4
    assert hits[0]["exact_phrases"] == ["query 1"]
    assert hits[0]["matched_phrases"] == ["query 1"]
    assert hits[0]["missing_required_modifiers"] == []
    assert hits[0]["wrong_entity_modifier"] is False
    assert hits[0]["phrase_score_before_penalty"] == 0.8
    assert hits[0]["phrase_score"] == hits[0]["phrase_score_after_penalty"]
    assert hits[0]["is_toc"] is False
    assert hits[0]["toc_filtered"] is False
    assert hits[0]["toc_penalty_applied"] is False
    assert hits[0]["quality_score"] == 1.0
    assert hits[0]["low_text_quality"] is False
    assert hits[0]["low_text_quality_reason"] is None
    assert hits[0]["query_type"] == "concept_lookup"
    assert hits[0]["chunk_type"] == "explanatory"
    assert hits[0]["chunk_type_reason"] == "connected_prose"
    assert hits[0]["section_title_reason"] == "meta:section_title"
    assert hits[0]["inferred_section_title"] == "query 1"
    assert hits[0]["required_terms"] == ["query 1"]
    assert hits[0]["required_term_score"] == 1.0
    assert hits[0]["required_term_match_type"] == "full_phrase_prefix"
    assert hits[0]["full_phrase_match"] is True
    assert hits[0]["missing_required_terms"] == []
    assert hits[0]["is_navigation_index"] is False
    assert hits[0]["navigation_filtered"] is False
    assert hits[0]["intent_boost_applied"] is True
    assert hits[0]["exercise_penalty_applied"] is False
    assert hits[0]["test_question_penalty_applied"] is False
    assert hits[0]["schema_boost_applied"] is False
    assert hits[0]["term_penalty_applied"] is False
    assert hits[0]["full_term_boost_applied"] is True
    assert hits[0]["concept_boost_applied"] is True
    assert hits[0]["exercise_demoted_for_concept_lookup"] is False
    assert hits[0]["schema_or_rule_boost_applied"] is True
    assert hits[0]["final_rank_reason"] == "concept_full_phrase_reference_boost"
    assert hits[0]["score_before_boosts"] == 0.72
    assert hits[0]["concept_full_phrase_boost_value"] == 0.05
    assert hits[0]["section_title_boost_value"] == 0.02
    assert hits[0]["schema_or_rule_boost_value"] == 0.03
    assert hits[0]["quality_penalty_value"] == 0.0
    assert hits[0]["boundary_penalty_value"] == 0.0
    assert hits[0]["exercise_penalty_value"] == 0.0
    assert hits[0]["score_after_boosts_before_clamp"] == 0.82
    assert hits[0]["expanded_from_neighbors"] is True
    assert hits[0]["penalties_applied"] == []
