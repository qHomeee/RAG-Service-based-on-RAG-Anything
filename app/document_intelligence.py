import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from app.config import settings


SUPPORTED_SUBJECTS = {
    "history",
    "russian_language",
    "safety",
    "biology",
    "math",
    "literature",
    "geography",
    "physics",
    "chemistry",
    "social_studies",
    "unknown",
}

_STOPWORDS = {
    "а",
    "без",
    "в",
    "во",
    "для",
    "до",
    "его",
    "ее",
    "и",
    "или",
    "их",
    "к",
    "на",
    "над",
    "о",
    "об",
    "от",
    "по",
    "под",
    "при",
    "про",
    "как",
    "что",
    "кто",
    "какой",
    "какая",
    "какое",
    "какие",
    "когда",
    "почему",
    "где",
    "чем",
    "сколько",
    "такое",
    "был",
    "была",
    "было",
    "были",
    "бывает",
    "бывают",
    "изучает",
    "изучают",
    "отвечает",
    "отвечают",
    "отличить",
    "расставлять",
    "ставится",
    "ставятся",
    "устроено",
    "с",
    "со",
    "у",
    "the",
    "and",
    "of",
}

_ROMAN_RE = re.compile(r"^[ivxlcdm]+$")
_TOKEN_RE = re.compile(r"[0-9a-zа-я]+")

EXERCISE_QUERY_TERMS = {
    "упражнение",
    "упражнения",
    "задание",
    "задания",
    "задачи",
    "тренировка",
    "тренировочные",
    "практика",
    "спишите",
    "выполните",
    "выполнить",
    "найдите",
    "подберите",
    "подчеркните",
}

PROBLEM_SOLVING_TERMS = {
    "решить",
    "реши",
    "решите",
    "вычислить",
    "вычислите",
    "найти",
    "найдите",
    "доказать",
    "докажите",
}

RULE_LOOKUP_TERMS = {
    "правило",
    "правила",
    "правописание",
    "орфография",
    "пишется",
    "употребляется",
    "склонение",
    "спряжение",
}

EXPLANATION_TERMS = {"почему", "как", "объясни", "объяснить", "объясните", "расскажи", "рассказать"}

ANSWER_FOCUS_STEMS: dict[str, tuple[str, ...]] = {
    "goal": ("цел", "задач", "предназнач", "направлен", "преслед", "добива", "стрем"),
    "cause": ("причин", "почему", "обуслов", "вследств", "из-за"),
    "consequence": ("последств", "результат", "итог", "привел", "привёл"),
}

QUERY_INITIAL_NON_ENTITIES = {
    "в",
    "для",
    "как",
    "какая",
    "какие",
    "какой",
    "каков",
    "какова",
    "каково",
    "каковы",
    "когда",
    "кто",
    "почему",
    "расскажи",
    "назови",
    "объясни",
    "опиши",
    "перечисли",
    "покажи",
    "приведи",
    "сравни",
    "сравните",
    "сформулируй",
    "что",
    "чем",
}

EXERCISE_COMMANDS = {
    "выполните",
    "выполнить",
    "спишите",
    "найдите",
    "подчеркните",
    "составьте",
    "объясните",
    "ответьте",
    "прочитайте",
    "укажите",
    "запишите",
    "подберите",
    "определите",
    "сравните",
    "докажите",
    "разберите",
    "решите",
    "подготовьте",
}

TEST_QUESTION_MARKERS = {
    "верно определены",
    "верно определена",
    "неправильно определены",
    "неправильно определена",
    "укажите",
    "выберите",
    "какой вариант",
    "какие варианты",
}

NAVIGATION_MARKERS = {
    "приложение",
    "содержание",
    "оглавление",
    "словарь",
    "указатель",
    "страница",
    "страницы",
}

SCHEMA_MARKERS = {
    "план",
    "порядок",
    "схема",
    "алгоритм",
    "шаг",
    "этап",
    "разбор",
    "часть речи",
    "морфологические признаки",
    "синтаксическая роль",
    "грамматическая основа",
    "члены предложения",
    "формула",
    "последовательность",
}

EDUCATIONAL_PARSING_HEADING_RE = re.compile(
    r"^\s*((?:морфологический|синтаксический|фонетический|морфемный|лексический)\s+разбор(?:\s+[а-яё]+){0,4})"
    r"(?=\s+(?:[IVXLCDM]+|[1-9]\d*)\s*[.)]|\s+[А-ЯA-ZЁ]\.|[.:]|$)",
    flags=re.IGNORECASE,
)

RULE_MARKERS = {
    "правило",
    "следует",
    "пишется",
    "употребляется",
    "необходимо",
    "нужно",
    "надо",
}

DEFINITION_MARKERS = {
    "это",
    "называется",
    "называют",
    "является",
    "включает",
    "состоит",
    "характеризуется",
    "представляет собой",
    "определение",
    "означает",
}


def normalize_for_matching(text: str) -> str:
    normalized = (text or "").lower().replace("ё", "е")
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = re.sub(r"([a-zа-я]+)-\s+([a-zа-я]+)", r"\1\2", normalized)
    return normalized


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_for_matching(text))


def analyze_query(query: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", (query or "").strip().lower().replace("ё", "е"))
    tokens = _significant_tokens(normalized)
    phrases = _query_phrases(query)
    exact_phrases = _exact_phrases(query)
    detected_subjects = detect_subjects(normalized)
    primary_subject = detected_subjects[0]["subject"] if detected_subjects else "unknown"
    subject_confidence = detected_subjects[0]["confidence"] if detected_subjects else 0.0
    query_type = detect_query_type(normalized, primary_subject)
    answer_focus = detect_answer_focus(normalized)
    return {
        "normalized_query": normalized,
        "detected_subjects": detected_subjects,
        "primary_subject": primary_subject,
        "subject_confidence": subject_confidence,
        "query_type": query_type,
        "answer_focus": answer_focus,
        "key_terms": tokens,
        "named_entities": extract_named_entities(query),
        "required_entities": extract_required_entities(query),
        "phrases": [" ".join(item) for item in phrases],
        "exact_phrases": exact_phrases,
        "required_terms": _required_multiword_terms(normalized, query_type=query_type, exact_phrases=exact_phrases),
        "numbers": [token for token in tokenize(normalized) if token.isdigit()],
    }


def detect_subjects(text: str) -> list[dict[str, Any]]:
    scores: list[tuple[str, float, list[str]]] = []
    for subject, hints in settings.subject_hints_default.items():
        subject_key = _normalize_subject(subject)
        if subject_key == "unknown":
            continue
        score, matched = _hint_score(text, hints)
        if score <= 0:
            continue
        confidence = min(0.98, 0.55 + 0.16 * score)
        scores.append((subject_key, confidence, matched[:8]))
    if _contains_name_with_roman_modifier(text) and not any(item[0] == "history" for item in scores):
        scores.append(("history", 0.72, ["name_roman_modifier"]))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [
        {"subject": subject, "confidence": round(confidence, 4), "matched_hints": matched}
        for subject, confidence, matched in scores
    ]


def detect_query_type(normalized_query: str, primary_subject: str) -> str:
    tokens = set(tokenize(normalized_query))
    significant = _significant_tokens(normalized_query)
    if not tokens:
        return "unknown"
    if tokens & EXERCISE_QUERY_TERMS:
        return "exercise_lookup"
    if tokens & PROBLEM_SOLVING_TERMS:
        return "problem_solving"
    if tokens & RULE_LOOKUP_TERMS:
        return "rule_lookup"
    if detect_answer_focus(normalized_query) != "none":
        return "explanation"
    if tokens & EXPLANATION_TERMS:
        return "explanation"
    if tokens & {"что", "такое", "определение"}:
        return "definition"
    if len(significant) <= 4:
        return "concept_lookup"
    if primary_subject == "russian_language" and tokens & {"причастие", "причастиях", "деепричастие", "разбор"}:
        return "rule_lookup"
    return "unknown"


def detect_answer_focus(normalized_query: str) -> str:
    tokens = tokenize(normalized_query)
    normalized = normalize_for_matching(normalized_query)
    for focus, stems in ANSWER_FOCUS_STEMS.items():
        if any(
            token.startswith(stem)
            for token in tokens
            for stem in stems
            if "-" not in stem
        ):
            return focus
        if any(stem in normalized for stem in stems if "-" in stem):
            return focus
    if "для чего" in normalized or "с какой целью" in normalized:
        return "goal"
    return "none"


def extract_named_entities(query: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b[А-ЯЁ][а-яё]+(?:\s+[IVXLCDM]+|\s+\d+)?", query or ""):
        value = match.group(0).strip()
        value_tokens = tokenize(value)
        if value_tokens and value_tokens[0] in QUERY_INITIAL_NON_ENTITIES:
            continue
        key = value.lower().replace("ё", "е")
        if key not in seen:
            entities.append(value)
            seen.add(key)
    for match in re.finditer(r"\b[А-ЯЁA-Z]{2,}\b", query or ""):
        value = match.group(0).strip()
        key = value.lower()
        if key not in seen:
            entities.append(value)
            seen.add(key)
    return entities


def extract_required_entities(query: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"\b[А-ЯЁ][а-яё]+(?:\s+[IVXLCDM]+|\s+\d+)?|\b[А-ЯЁA-Z]{2,}\b")
    for match in pattern.finditer(query or ""):
        value = match.group(0).strip()
        normalized = value.lower().replace("ё", "е")
        tokens = tokenize(value)
        if not tokens:
            continue
        first = tokens[0]
        has_modifier = any(token.isdigit() or bool(_ROMAN_RE.fullmatch(token)) for token in tokens[1:])
        is_acronym = value.isupper() and len(value) >= 2
        if first in QUERY_INITIAL_NON_ENTITIES:
            continue
        if normalized in seen:
            continue
        entities.append(value)
        seen.add(normalized)
    return entities


def build_document_profile(
    *,
    source_uri: str,
    title: str | None,
    file_path: Path,
    parsed_elements: Iterable[Any],
    collection: str,
) -> dict[str, Any]:
    elements = list(parsed_elements)
    first_text = " ".join(str(getattr(elem, "content", "")) for elem in elements[:8])
    all_text_sample = " ".join(str(getattr(elem, "content", "")) for elem in elements[:30])
    section_titles = extract_section_titles(elements)
    classification_text = " ".join([source_uri, title or "", file_path.stem, first_text, " ".join(section_titles)])
    detected_subjects = detect_subjects(classification_text)
    subject = detected_subjects[0]["subject"] if detected_subjects else "unknown"
    subject_confidence = detected_subjects[0]["confidence"] if detected_subjects else 0.0
    grade = extract_grade(" ".join([source_uri, title or "", file_path.stem, first_text]))
    doc_type = detect_doc_type(" ".join([source_uri, title or "", file_path.stem, first_text]))
    language = detect_language(first_text or all_text_sample or source_uri)
    keywords = extract_keywords(" ".join([classification_text, all_text_sample]))
    profile_text = " ".join(
        [
            title or file_path.name,
            source_uri,
            subject,
            doc_type,
            " ".join(keywords),
            " ".join(section_titles[:40]),
        ]
    )
    return {
        "source_uri": source_uri,
        "title": title or file_path.name,
        "collection": collection,
        "subject": subject,
        "subject_confidence": round(subject_confidence, 4),
        "detected_subjects": detected_subjects[:5],
        "grade": grade,
        "doc_type": doc_type,
        "language": language,
        "keywords": keywords[:50],
        "section_titles": section_titles[:80],
        "profile_text": re.sub(r"\s+", " ", profile_text).strip()[:4000],
    }


def extract_grade(text: str) -> int | None:
    match = re.search(r"(?<!\d)([1-9]|1[0-2])\s*(?:класс|klass|class)(?![a-zа-я0-9])", text or "", re.IGNORECASE)
    if match:
        return int(match.group(1))

    filename_like = re.sub(r"[_-]+", " ", text or "")
    transliterated_subject_grade = re.search(
        r"(?:russkij\s+jazyk|jazyk|geografija|khimiya|istorija|biologija|fizika|literatura)"
        r"\s+([1-9]|1[0-2])(?=\s|$)",
        filename_like,
        re.IGNORECASE,
    )
    return int(transliterated_subject_grade.group(1)) if transliterated_subject_grade else None


def detect_doc_type(text: str) -> str:
    normalized = normalize_for_matching(text)
    checks = {
        "workbook": ("рабочая тетрадь", "workbook", "тетрадь"),
        "textbook": ("учебник", "textbook", "uchebnik"),
        "manual": ("пособие", "руководство", "manual", "методич"),
        "collection": ("сборник", "задачник", "хрестоматия", "collection", "sbornik"),
    }
    for doc_type, hints in checks.items():
        if any(hint in normalized for hint in hints):
            return doc_type
    return "unknown"


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    cyr = len(re.findall(r"[а-яёА-ЯЁ]", text))
    lat = len(re.findall(r"[a-zA-Z]", text))
    if cyr > lat * 1.5 and cyr > 20:
        return "ru"
    if lat > cyr * 1.5 and lat > 20:
        return "en"
    return "unknown"


def extract_section_titles(elements: Iterable[Any]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for elem in elements:
        meta = getattr(elem, "meta", {}) or {}
        for key in ("heading", "title", "section", "section_title"):
            value = meta.get(key)
            if isinstance(value, str):
                _append_title(value, titles, seen)
        for line in str(getattr(elem, "content", "") or "").splitlines()[:30]:
            cleaned = re.sub(r"^#{1,6}\s*", "", line).strip()
            if _looks_like_section_title(cleaned):
                _append_title(cleaned, titles, seen)
            if len(titles) >= 120:
                return titles
    return titles


def extract_keywords(text: str, *, limit: int = 60) -> list[str]:
    counts: Counter[str] = Counter()
    for token in _significant_tokens(text):
        if len(token) < 4 or token.isdigit():
            continue
        counts[_stem(token)] += 1
    return [token for token, _ in counts.most_common(limit)]


def profile_text(profile: dict[str, Any]) -> str:
    parts = [
        profile.get("title"),
        profile.get("source_uri"),
        profile.get("subject"),
        profile.get("doc_type"),
        " ".join(profile.get("keywords") or []),
        " ".join(profile.get("section_titles") or []),
        profile.get("profile_text"),
    ]
    return " ".join(str(part) for part in parts if part)


def profile_subject_score(query_analysis: dict[str, Any], profile: dict[str, Any]) -> float:
    detected = query_analysis.get("detected_subjects") or []
    if not detected:
        return 0.5
    subject = _normalize_subject(str(profile.get("subject") or "unknown"))
    if subject == "unknown":
        return 0.5
    for item in detected:
        if item.get("subject") == subject:
            return max(0.8, float(item.get("confidence") or 0.0))
    top_confidence = float(detected[0].get("confidence") or 0.0)
    return 0.12 if top_confidence >= settings.retrieval_subject_confidence_threshold else 0.35


def profile_lexical_score(query_analysis: dict[str, Any], profile: dict[str, Any]) -> float:
    query_terms = query_analysis.get("key_terms") or []
    if not query_terms:
        return 0.0
    haystack = {_stem(token) for token in tokenize(profile_text(profile))}
    if not haystack:
        return 0.0
    matched = 0
    for term in query_terms:
        if _stem(str(term)) in haystack:
            matched += 1
    phrase_hits = 0
    for phrase in query_analysis.get("phrases") or []:
        if _phrase_match(str(phrase), profile_text(profile)):
            phrase_hits += 1
    term_score = matched / len(query_terms)
    phrase_score = min(1.0, phrase_hits / max(1, len(query_analysis.get("phrases") or [])))
    return min(1.0, max(term_score, phrase_score))


def subject_expansions_for_query(query_analysis: dict[str, Any]) -> list[str]:
    if not settings.query_expansion_enabled:
        return []
    subjects = [item.get("subject") for item in query_analysis.get("detected_subjects") or [] if item.get("confidence", 0) >= 0.55]
    if not subjects:
        return []
    query = query_analysis.get("normalized_query") or ""
    expansions: list[str] = []
    seen: set[str] = set()
    for subject in subjects[:2]:
        mapping = settings.query_expansions_by_subject.get(str(subject), {})
        for key, values in mapping.items():
            if not _phrase_match(key, query):
                continue
            for value in values:
                normalized = re.sub(r"\s+", " ", str(value).strip().lower())
                if normalized and normalized not in seen:
                    expansions.append(normalized)
                    seen.add(normalized)
    return expansions


def is_toc_text(text: str, page: int | None = None) -> bool:
    normalized = normalize_for_matching(text)
    marker_found = any(marker in normalized[:300] for marker in ("оглавление", "содержание", "contents"))
    if marker_found:
        return True
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    tokens = tokenize(text)
    if not tokens:
        return False

    page_numbers = re.findall(r"(?<!\d)([1-9]\d{0,2})(?!\d)", normalized)
    numbered_lines = sum(1 for line in lines[:30] if re.search(r"\S.{3,120}\s\d{1,3}\s*$", line))
    short_title_lines = sum(1 for line in lines[:30] if 2 <= len(tokenize(line)) <= 12 and not re.search(r"[.!?]\s*$", line))
    coherent_lines = sum(1 for line in lines[:30] if len(tokenize(line)) >= 14 and re.search(r"[.!?]\s*$", line))
    flat_entries = _flat_toc_entry_count(normalized)
    early_page = page is not None and page <= 8
    page_number_ratio = len(page_numbers) / max(1, len(tokens))
    title_line_ratio = short_title_lines / max(1, len(lines))
    coherent_ratio = coherent_lines / max(1, len(lines))

    score = 0.0
    if early_page:
        score += 1.0
    if numbered_lines >= 5:
        score += 2.0
    elif numbered_lines >= 3:
        score += 1.0
    if flat_entries >= 8:
        score += 2.5
    elif flat_entries >= 5:
        score += 1.5
    if len(page_numbers) >= 8 or page_number_ratio >= 0.12:
        score += 1.0
    if title_line_ratio >= 0.6:
        score += 1.0
    if coherent_ratio <= 0.25:
        score += 0.75
    if marker_found and (numbered_lines >= 2 or flat_entries >= 2):
        score += 2.0

    return score >= 3.5


def text_quality_flags(text: str, *, page: int | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or {}
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    tokens = tokenize(normalized)
    first_alpha = _first_alpha(normalized)
    starts_mid_word = bool(first_alpha and first_alpha.islower())
    is_toc = bool(meta.get("is_toc") is True or is_toc_text(normalized, page=page))
    is_index = _looks_like_index(normalized)
    is_bibliography = _looks_like_bibliography(normalized)
    is_caption = _looks_like_caption(normalized)
    is_too_short = len(normalized) < 180 or len(tokens) < 22
    is_fragmented = starts_mid_word or _looks_fragmented(normalized)
    has_expanded_context = bool(meta.get("expanded_text") or meta.get("parent_context") or meta.get("expanded_from_neighbors"))
    low_text_quality = bool(
        is_toc
        or is_index
        or is_bibliography
        or (is_caption and is_too_short)
        or (is_fragmented and len(normalized) < 420)
        or (is_too_short and len(tokens) < 8)
    )

    quality_score = 1.0
    reason: str | None = None
    if low_text_quality:
        quality_score = 0.25
        if is_toc:
            reason = "toc"
        elif is_index:
            reason = "index"
        elif is_bibliography:
            reason = "bibliography"
        elif is_fragmented:
            reason = "fragmented"
        elif is_too_short:
            reason = "too_short_without_context"
    elif is_fragmented:
        quality_score = 0.45
        reason = "fragmented"
    elif is_too_short:
        quality_score = 0.75 if has_expanded_context else 0.5
        reason = None if has_expanded_context else "too_short_without_context"
    elif is_caption:
        quality_score = 0.75
        reason = "caption"

    return {
        "is_toc": is_toc,
        "is_index": is_index,
        "is_bibliography": is_bibliography,
        "is_caption": is_caption,
        "is_fragmented": is_fragmented,
        "is_too_short": is_too_short,
        "starts_mid_word": starts_mid_word,
        "low_text_quality": low_text_quality,
        "quality_score": round(quality_score, 4),
        "low_text_quality_reason": reason,
    }


def detect_chunk_type(text: str, *, meta: dict[str, Any] | None = None, page: int | None = None) -> str:
    return str(detect_chunk_type_details(text, meta=meta, page=page)["chunk_type"])


def detect_chunk_type_details(text: str, *, meta: dict[str, Any] | None = None, page: int | None = None) -> dict[str, str]:
    meta = meta or {}
    normalized = normalize_for_matching(text)
    tokens = tokenize(normalized)
    if not tokens:
        return {"chunk_type": "unknown", "chunk_type_reason": "empty_text"}

    if _looks_like_chapter_opener(text):
        return {"chunk_type": "navigation_index", "chunk_type_reason": "chapter_terms_and_timeline"}

    if _looks_like_navigation_reference(text):
        return {"chunk_type": "navigation_index", "chunk_type_reason": "navigation_reference"}

    if _looks_like_test_question(text):
        return {"chunk_type": "test_question", "chunk_type_reason": "test_labels_or_answer_options"}

    command_count = sum(1 for command in EXERCISE_COMMANDS if _phrase_match(command, normalized))
    numbered_tasks = len(re.findall(r"(?<!\d)(?:упр\.\s*)?\d{1,4}\s*[.)]\s+[А-ЯЁA-Zа-яёa-z]", text or ""))
    dense_numbered_items = len(re.findall(r"(?<!\d)\d{1,2}\s*[.)]\s+", text or ""))
    theory_context = (
        "теоретическ" in normalized
        and ("информац" in normalized or "сведен" in normalized)
        and any(_phrase_match(marker, normalized) for marker in DEFINITION_MARKERS)
    )
    if theory_context:
        return {"chunk_type": "definition", "chunk_type_reason": "theory_with_definition"}
    if command_count >= 1 and (numbered_tasks >= 1 or len(tokens) <= 160):
        return {"chunk_type": "exercise", "chunk_type_reason": "exercise_command"}
    if command_count >= 2 or (command_count >= 1 and dense_numbered_items >= 5):
        return {"chunk_type": "exercise", "chunk_type_reason": "multiple_tasks_or_commands"}

    if _looks_like_navigation_index(text, page=page):
        return {"chunk_type": "navigation_index", "chunk_type_reason": "navigation_titles_with_page_numbers"}

    if meta.get("is_toc") is True or is_toc_text(text, page=page):
        return {"chunk_type": "toc", "chunk_type_reason": "table_of_contents"}
    if _looks_like_index(normalized):
        return {"chunk_type": "index", "chunk_type_reason": "alphabetical_or_subject_index"}
    if _looks_like_bibliography(normalized):
        return {"chunk_type": "bibliography", "chunk_type_reason": "bibliography"}

    schema_hits = sum(1 for marker in SCHEMA_MARKERS if _phrase_match(marker, normalized))
    roman_steps = len(re.findall(r"(?<![A-Za-zА-Яа-яЁё])(?:[IVXLCDM]+|[1-9]\d*)\s*[.)]\s+", text or ""))
    bullet_steps = len(re.findall(r"(?:^|\s)[-•]\s+\S+", text or ""))
    if schema_hits >= 1 and (roman_steps >= 1 or bullet_steps >= 2 or schema_hits >= 3):
        return {"chunk_type": "schema_or_plan", "chunk_type_reason": "schema_markers_with_steps"}
    if roman_steps >= 3 and any(_phrase_match(marker, normalized) for marker in ("признаки", "порядок", "план", "формула", "роль")):
        return {"chunk_type": "schema_or_plan", "chunk_type_reason": "structured_steps"}

    if re.match(r"^[А-ЯЁA-Z][^.!?]{2,100}\s+[—-]\s+\S+", text or "") and len(tokens) >= 4:
        return {"chunk_type": "definition", "chunk_type_reason": "dash_definition"}
    if any(_phrase_match(marker, normalized) for marker in RULE_MARKERS):
        return {"chunk_type": "rule", "chunk_type_reason": "rule_marker"}
    if any(_phrase_match(marker, normalized) for marker in DEFINITION_MARKERS):
        return {"chunk_type": "definition", "chunk_type_reason": "definition_marker"}
    if any(_phrase_match(marker, normalized) for marker in ("например", "пример", "образец")):
        return {"chunk_type": "example", "chunk_type_reason": "example_marker"}
    if re.match(r"^[^:]{6,140}:\s+\S+", text or "") and len(tokens) >= 8:
        return {"chunk_type": "explanatory", "chunk_type_reason": "title_colon_explanation"}
    if len(tokens) >= 28 and re.search(r"[.!?]", text or ""):
        return {"chunk_type": "explanatory", "chunk_type_reason": "connected_prose"}
    return {"chunk_type": "unknown", "chunk_type_reason": "no_chunk_type_signal"}


def infer_section_title(text: str, *, meta: dict[str, Any] | None = None) -> str | None:
    title = infer_section_title_details(text, meta=meta).get("section_title")
    return str(title) if title else None


def infer_section_title_details(text: str, *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or {}
    for key in ("section_title", "parent_heading", "heading", "nearest_heading"):
        value = meta.get(key)
        if isinstance(value, str):
            cleaned, reason = _clean_section_title(value)
            if cleaned:
                return {"section_title": cleaned, "section_path": [cleaned], "section_title_reason": f"meta:{key}"}
            if reason:
                rejected_reason = reason
                break
    else:
        rejected_reason = None

    for key in ("section_path", "heading_path", "nearest_headings"):
        value = meta.get(key)
        if isinstance(value, list):
            cleaned_path: list[str] = []
            for item in reversed(value):
                if not isinstance(item, str):
                    continue
                cleaned, reason = _clean_section_title(item)
                if cleaned:
                    cleaned_path.insert(0, cleaned)
                    return {"section_title": cleaned, "section_path": cleaned_path or [cleaned], "section_title_reason": f"meta:{key}"}
                rejected_reason = reason or rejected_reason

    raw = (text or "").strip()
    first_line = raw.splitlines()[0].strip() if raw else ""
    first_line = re.sub(r"^#{1,6}\s*", "", first_line)
    prefix_title = _heading_prefix_title(first_line)
    if prefix_title:
        cleaned, reason = _clean_section_title(prefix_title)
        if cleaned and detect_chunk_type(prefix_title, meta=meta) not in {"exercise", "test_question", "navigation_index", "toc", "index"}:
            return {"section_title": cleaned, "section_path": [cleaned], "section_title_reason": "heading_prefix"}
        rejected_reason = reason or rejected_reason
    cleaned, reason = _clean_section_title(first_line)
    if cleaned and detect_chunk_type(first_line, meta=meta) not in {"exercise", "test_question", "navigation_index", "toc", "index"}:
        return {"section_title": cleaned, "section_path": [cleaned], "section_title_reason": "first_line"}
    rejected_reason = reason or rejected_reason

    normalized = re.sub(r"\s+", " ", raw)
    if not normalized:
        return {"section_title": None, "section_path": [], "section_title_reason": "empty_text"}
    prefix_title = _heading_prefix_title(normalized)
    if prefix_title:
        cleaned, reason = _clean_section_title(prefix_title)
        if cleaned and detect_chunk_type(prefix_title, meta=meta) not in {"exercise", "test_question", "navigation_index", "toc", "index"}:
            return {"section_title": cleaned, "section_path": [cleaned], "section_title_reason": "heading_prefix"}
        rejected_reason = reason or rejected_reason
    sentence_heading_match = re.match(r"^(.{6,140}?)\.\s+(?:[IVXLCDM]+|[1-9]\d*)\s*[.)]\s+", normalized)
    if not sentence_heading_match:
        sentence_heading_match = re.match(r"^(.{6,140}?)\.\s+[А-ЯЁA-Z]", normalized)
    if sentence_heading_match:
        candidate = sentence_heading_match.group(1).strip(" .:-")
        cleaned, reason = _clean_section_title(candidate)
        if cleaned and detect_chunk_type(candidate, meta=meta) not in {"exercise", "test_question", "navigation_index", "toc", "index"}:
            return {"section_title": cleaned, "section_path": [cleaned], "section_title_reason": "inline_heading"}
        rejected_reason = reason or rejected_reason
    marker_match = re.search(r"\s(?:[IVXLCDM]+|[1-9]\d*)\s*[.)]\s+", normalized)
    if marker_match:
        candidate = normalized[: marker_match.start()].strip(" .:-")
        cleaned, reason = _clean_section_title(candidate)
        if cleaned and detect_chunk_type(candidate, meta=meta) not in {"exercise", "test_question", "navigation_index", "toc", "index"}:
            return {"section_title": cleaned, "section_path": [cleaned], "section_title_reason": "before_step_marker"}
        rejected_reason = reason or rejected_reason
    colon_match = re.match(r"^([^:]{8,140}):\s+\S+", normalized)
    if colon_match:
        candidate = colon_match.group(1).strip()
        cleaned, reason = _clean_section_title(candidate)
        if cleaned and detect_chunk_type(candidate, meta=meta) not in {"exercise", "test_question", "navigation_index", "toc", "index"}:
            return {"section_title": cleaned, "section_path": [cleaned], "section_title_reason": "before_colon"}
        rejected_reason = reason or rejected_reason
    return {"section_title": None, "section_path": [], "section_title_reason": rejected_reason or "not_found"}


def _heading_prefix_title(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return None
    if re.match(r"^(?:[ivxlcdm]+|[1-9]\d*)\s*[.)]\s+", normalized, flags=re.IGNORECASE):
        return None
    if re.match(r"^\d+(?:\s*[,;]\s*\d+)*\s*\.\s+", normalized):
        return None
    if re.match(r"^[АAВB]\d{1,2}\s*[.)]\s+", normalized):
        return None

    match = EDUCATIONAL_PARSING_HEADING_RE.match(normalized)
    if match:
        return match.group(1).strip(" .:-")

    marker_match = re.search(r"\s(?:[IVXLCDM]+|[1-9]\d*)\s*[.)]\s+", normalized)
    if marker_match:
        candidate = normalized[: marker_match.start()].strip(" .:-")
        if 2 <= len(candidate.split()) <= 8 and _looks_like_known_heading(candidate):
            return candidate
    return None


def _looks_like_test_question(text: str) -> bool:
    raw = text or ""
    normalized = normalize_for_matching(raw)
    labels = len(re.findall(r"(?<![0-9A-Za-zА-Яа-яЁё])[аaвb]\d{1,2}\s*[.)]", normalized))
    answer_options = len(re.findall(r"(?<!\d)[1-4]\s*[)]", raw))
    marker_hits = sum(1 for marker in TEST_QUESTION_MARKERS if _phrase_match(marker, normalized))
    if labels >= 2:
        return True
    if labels >= 1 and (answer_options >= 2 or marker_hits >= 1):
        return True
    return answer_options >= 4 and marker_hits >= 1


def _looks_like_navigation_reference(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return False
    normalized = normalize_for_matching(cleaned)
    if re.match(r"^\d+(?:\s*[,;]\s*\d+)*\s*\.\s*(?:см\.?|смотрите)\s+", cleaned, flags=re.IGNORECASE):
        return True
    if normalized.startswith("см ") and len(tokenize(normalized)) <= 8:
        return True
    return "см. разбор" in cleaned.lower() or "см разбор" in normalized


def _looks_like_chapter_opener(text: str) -> bool:
    """Detect textbook openers made of a guiding question, terms and timeline."""
    raw = text or ""
    if "?" not in raw[:350]:
        return False
    bullet_terms = len(re.findall(r"[•●]\s*[A-Za-zА-Яа-яЁё«\"]", raw))
    year_entries = len(
        re.findall(
            r"(?<!\d)(?:1[5-9]\d{2}|20\d{2})(?:\s*[—-]\s*(?:1[5-9]\d{2}|20\d{2}))?\s*(?:гг?\.?|год(?:а|у)?)?",
            raw,
            flags=re.IGNORECASE,
        )
    )
    return bullet_terms >= 3 and year_entries >= 3


def _looks_like_navigation_index(text: str, *, page: int | None = None) -> bool:
    normalized = normalize_for_matching(text)
    tokens = tokenize(normalized)
    if not tokens:
        return False
    marker_found = any(_phrase_match(marker, normalized[:220]) for marker in NAVIGATION_MARKERS)
    page_numbers = re.findall(r"(?<!\d)([1-9]\d{0,2})(?!\d)", normalized)
    flat_entries = _flat_toc_entry_count(normalized)
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    coherent_lines = sum(1 for line in lines[:30] if len(tokenize(line)) >= 14 and re.search(r"[.!?]\s*$", line))
    coherent_ratio = coherent_lines / max(1, len(lines))
    early_page = page is not None and page <= 10

    if marker_found and len(page_numbers) >= 1 and len(tokens) <= 24:
        return True
    if normalized.count("§") >= 3 and len(page_numbers) >= 3:
        return True
    if normalized.count("§") >= 2 and len(page_numbers) >= 2 and len(tokens) <= 30:
        return True
    if marker_found and (flat_entries >= 2 or len(page_numbers) >= 4) and coherent_ratio <= 0.35:
        return True
    if flat_entries >= 5 and coherent_ratio <= 0.35:
        return True
    if early_page and flat_entries >= 5 and coherent_ratio <= 0.35:
        return True
    return False


def _clean_section_title(text: str) -> tuple[str | None, str | None]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return None, "empty"
    normalized = normalize_for_matching(cleaned)
    if re.fullmatch(r"[\d\s.,;:()]+", cleaned):
        return None, "numeric_or_punctuation"
    if re.match(r"^(?:[ivxlcdm]+|[1-9]\d*)\s*[.)]\s+", cleaned, flags=re.IGNORECASE):
        return None, "plan_step"
    if re.match(r"^\d+(?:\s*[,;]\s*\d+)*\s*\.\s+", cleaned):
        return None, "numeric_plan_step"
    if re.match(r"^[АAВB]\d{1,2}\s*[.)]\s+", cleaned):
        return None, "test_question_label"
    if _looks_like_test_question(cleaned):
        return None, "test_question_label"
    if _looks_like_navigation_reference(cleaned):
        return None, "navigation_reference"
    if normalized.startswith("см ") or "см разбор" in normalized or re.search(r"\bп\.\s", cleaned.lower()):
        return None, "cross_reference"
    if cleaned.count("(") > cleaned.count(")"):
        return None, "unclosed_parenthetical"
    if len(cleaned) > 180:
        return None, "too_long"
    words = cleaned.split()
    if re.search(r"\s[—-]\s", cleaned) and len(words) > 3:
        return None, "definition_like"
    if len(words) > 7 and not _looks_like_known_heading(cleaned):
        return None, "too_many_words"
    if _looks_like_sentence_instruction(cleaned):
        return None, "instruction_like"
    if cleaned.endswith((".", ",", ";")) and len(words) > 5:
        return None, "sentence_like"
    if len(words) <= 1 and len(cleaned) <= 3:
        return None, "too_short"
    if _looks_like_navigation_index(cleaned):
        stripped = _strip_trailing_page_number(cleaned)
        if stripped and _looks_like_section_title(stripped):
            return stripped, "navigation_title"
        return None, "navigation_index"
    stripped = _strip_trailing_page_number(cleaned)
    if stripped:
        cleaned = stripped
    if not _looks_like_section_title(cleaned):
        return None, "invalid_title"
    return cleaned[:180], None


def _looks_like_known_heading(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return False
    normalized = normalize_for_matching(cleaned)
    if EDUCATIONAL_PARSING_HEADING_RE.match(normalized):
        return True
    if re.match(r"^(?:глава|раздел|параграф|урок|тема)\s+\S+", normalized):
        return True
    if re.search(r"\b(?:план|схема|алгоритм|формула|правило|теорема|закон)\b", normalized):
        return True
    return False


def _looks_like_sentence_instruction(text: str) -> bool:
    normalized = normalize_for_matching(text)
    command_hits = sum(1 for command in EXERCISE_COMMANDS if _phrase_match(command, normalized))
    if command_hits:
        return True
    return bool(re.search(r"\b(?:укажите|выберите|найдите|сравните|объясните|докажите)\b", normalized))


def _required_multiword_terms(normalized_query: str, *, query_type: str, exact_phrases: list[str]) -> list[str]:
    if query_type != "concept_lookup":
        return []
    tokens = tokenize(normalized_query)
    if not tokens or exact_phrases:
        return []
    token_set = set(tokens)
    if token_set & (EXERCISE_QUERY_TERMS | PROBLEM_SOLVING_TERMS | EXPLANATION_TERMS | {"что", "такое", "определение"}):
        return []
    significant = [token for token in tokens if token not in _STOPWORDS and not _is_weak_token(token)]
    if 2 <= len(significant) <= 5:
        return [" ".join(significant)]
    return []


def _strip_trailing_page_number(text: str) -> str | None:
    match = re.match(r"^(.{4,120}?)\s+([1-9]\d{0,2})$", text or "")
    if not match:
        return None
    candidate = match.group(1).strip(" .:-")
    first = tokenize(candidate)[:1]
    if first and first[0] in {"глава", "параграф", "раздел", "урок"}:
        return None
    if re.search(r"\b(?:год|года|году|век|века)\b", normalize_for_matching(candidate)):
        return None
    return candidate if candidate else None


def _first_alpha(text: str) -> str | None:
    match = re.search(r"[A-Za-zА-Яа-яЁё]", text or "")
    return match.group(0) if match else None


def _looks_like_index(text: str) -> bool:
    normalized = normalize_for_matching(text)
    if any(marker in normalized[:120] for marker in ("предметный указатель", "именной указатель", "указатель")):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    comma_dense = sum(1 for line in lines[:20] if line.count(",") >= 3 and re.search(r"\d", line))
    return len(lines) >= 5 and comma_dense >= 4


def _looks_like_bibliography(text: str) -> bool:
    normalized = normalize_for_matching(text)
    if any(marker in normalized[:180] for marker in ("список литературы", "литература", "библиография", "references")):
        return True
    return len(re.findall(r"\b\d{4}\b", normalized)) >= 4 and len(re.findall(r"[А-ЯA-Z]\.", text or "")) >= 4


def _looks_like_caption(text: str) -> bool:
    normalized = normalize_for_matching(text)
    return bool(re.match(r"^(рис\.?|рисунок|табл\.?|таблица|иллюстрация|figure)\s*\d*", normalized))


def _looks_fragmented(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if re.match(r"^[а-яё]{1,4}[а-яё]*[.)]\s+[А-ЯЁ]", normalized):
        return True
    if re.match(r"^[а-яё]+[,.!?;:]\s+", normalized):
        return True
    if normalized.count("...") >= 2:
        return True
    return False


def _flat_toc_entry_count(normalized_text: str) -> int:
    matches = list(re.finditer(r"(?<!\d)([1-9]\d{0,2})(?!\d)", normalized_text or ""))
    if len(matches) < 3:
        return 0
    count = 0
    previous_end = 0
    for match in matches[:80]:
        segment = normalized_text[previous_end : match.start()].strip(" .,:;-/—–\n\t")
        previous_end = match.end()
        words = tokenize(segment)
        if 2 <= len(words) <= 14 and sum(any("а" <= char <= "я" or "a" <= char <= "z" for char in word) for word in words) >= 2:
            count += 1
    return count


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    dot = sum(float(left[idx]) * float(right[idx]) for idx in range(size))
    left_norm = math.sqrt(sum(float(left[idx]) ** 2 for idx in range(size)))
    right_norm = math.sqrt(sum(float(right[idx]) ** 2 for idx in range(size)))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(0.0, min(1.0, (dot / (left_norm * right_norm) + 1.0) / 2.0))


def _hint_score(text: str, hints: Iterable[str]) -> tuple[float, list[str]]:
    score = 0.0
    matched: list[str] = []
    for hint in hints:
        normalized_hint = normalize_for_matching(str(hint))
        if not normalized_hint:
            continue
        if _phrase_match(normalized_hint, text):
            hint_tokens = tokenize(normalized_hint)
            score += 1.5 if len(hint_tokens) > 1 else 1.0
            matched.append(str(hint))
    return score, matched


def _phrase_match(phrase: str, text: str) -> bool:
    phrase_tokens = tokenize(phrase)
    text_tokens = tokenize(text)
    if not phrase_tokens or not text_tokens or len(phrase_tokens) > len(text_tokens):
        return False
    phrase_forms = [_stem(token) for token in phrase_tokens]
    text_forms = [_stem(token) for token in text_tokens]
    for idx in range(0, len(text_forms) - len(phrase_forms) + 1):
        if text_forms[idx : idx + len(phrase_forms)] == phrase_forms:
            return True
    return False


def _query_phrases(text: str) -> list[tuple[str, ...]]:
    tokens = tokenize(text)
    phrases: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for idx, token in enumerate(tokens[:-1]):
        nxt = tokens[idx + 1]
        if token in _STOPWORDS:
            continue
        if _ROMAN_RE.fullmatch(nxt) or nxt.isdigit():
            phrase = (token, nxt)
            if phrase not in seen:
                phrases.append(phrase)
                seen.add(phrase)
    significant = [token for token in tokens if token not in _STOPWORDS and not _is_weak_token(token)]
    for left, right in zip(significant, significant[1:]):
        if len(left) >= 4 and len(right) >= 4:
            phrase = (left, right)
            if phrase not in seen:
                phrases.append(phrase)
                seen.add(phrase)
    return phrases


def _exact_phrases(text: str) -> list[str]:
    tokens = tokenize(text)
    phrases: list[str] = []
    seen: set[tuple[str, ...]] = set()

    def add(phrase_tokens: tuple[str, ...]) -> None:
        if phrase_tokens and phrase_tokens not in seen:
            phrases.append(" ".join(phrase_tokens))
            seen.add(phrase_tokens)

    for idx, token in enumerate(tokens[:-1]):
        nxt = tokens[idx + 1]
        if token in _STOPWORDS or _is_weak_token(token):
            continue
        if not _is_modifier_token(nxt):
            continue
        phrase_tokens = [token, nxt]
        if idx + 2 < len(tokens) and _is_modifier_continuation(nxt, tokens[idx + 2]):
            phrase_tokens.append(tokens[idx + 2])
        add(tuple(phrase_tokens))

    for phrase in _letter_rule_phrases(text):
        add(tuple(tokenize(phrase)))

    for acronym in _strong_abbreviation_tokens(text):
        if any(acronym in phrase.split() and len(phrase.split()) > 1 for phrase in phrases):
            continue
        add((acronym,))

    return phrases


def _is_modifier_token(token: str) -> bool:
    token = (token or "").lower()
    if not token or token in _STOPWORDS:
        return False
    if token.isdigit() or _ROMAN_RE.fullmatch(token):
        return True
    if any(char.isdigit() for char in token):
        return True
    return len(token) == 1 and bool(re.fullmatch(r"[a-z]", token))


def _is_modifier_continuation(previous: str, token: str) -> bool:
    token = (token or "").lower()
    if not token or token in _STOPWORDS:
        return False
    if previous.isdigit() and token.isalpha() and 2 <= len(token) <= 4:
        return True
    return any(char.isdigit() for char in previous) and token.isalpha() and 1 <= len(token) <= 4


def _strong_abbreviation_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"(?<![0-9A-Za-zА-Яа-яЁё])(?:[A-ZА-ЯЁ]{2,}[0-9]*|[A-ZА-ЯЁ]+[0-9]{2,})(?![0-9A-Za-zА-Яа-яЁё])")
    for match in pattern.finditer(text or ""):
        if match.start() >= 2 and (text or "")[match.start() - 1] == "-" and (text or "")[match.start() - 2].isdigit():
            continue
        normalized = tokenize(match.group(0))
        if len(normalized) != 1:
            continue
        token = normalized[0]
        if _ROMAN_RE.fullmatch(token) or token in seen:
            continue
        tokens.append(token)
        seen.add(token)
    return tokens


def _letter_rule_phrases(text: str) -> list[str]:
    pattern = re.compile(r"(?<![0-9A-Za-zА-Яа-яЁё])[A-ZА-ЯЁ]\s+(?:и|and)\s+[A-ZА-ЯЁ]{2,}(?![0-9A-Za-zА-Яа-яЁё])")
    return [match.group(0) for match in pattern.finditer(text or "")]


def _contains_name_with_roman_modifier(text: str) -> bool:
    return bool(re.search(r"\b[А-ЯЁ][а-яё]{3,}\s+[IVXLCDM]+\b", text or ""))


def _significant_tokens(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokenize(text):
        if token in _STOPWORDS or _is_weak_token(token):
            continue
        if len(token) < 2 and not token.isdigit():
            continue
        if token in seen:
            continue
        terms.append(token)
        seen.add(token)
    return terms


def _is_weak_token(token: str) -> bool:
    return len(token) == 1 and (token.isalpha() or bool(_ROMAN_RE.fullmatch(token)))


def _stem(token: str) -> str:
    token = normalize_for_matching(token)
    if any("а" <= char <= "я" for char in token) and len(token) >= 5:
        for suffix in (
            "иями",
            "ями",
            "ами",
            "ого",
            "ему",
            "ыми",
            "ими",
            "ая",
            "яя",
            "ое",
            "ее",
            "ые",
            "ие",
            "ый",
            "ий",
            "ой",
            "ах",
            "ях",
            "ам",
            "ям",
            "ом",
            "ем",
            "ов",
            "ев",
            "ия",
            "ии",
            "ей",
            "ы",
            "и",
            "а",
            "я",
            "е",
            "о",
            "у",
            "ю",
        ):
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                return token[: -len(suffix)]
    return token


def _append_title(value: str, titles: list[str], seen: set[str]) -> None:
    cleaned = re.sub(r"\s+", " ", value.strip())
    if not _looks_like_section_title(cleaned):
        return
    key = normalize_for_matching(cleaned)
    if key in seen:
        return
    titles.append(cleaned[:180])
    seen.add(key)


def _looks_like_section_title(text: str) -> bool:
    if not text:
        return False
    cleaned = re.sub(r"\s+", " ", text.strip())
    normalized = normalize_for_matching(cleaned)
    if re.match(r"^(?:[ivxlcdm]+|[1-9]\d*)\s*[.)]\s+", cleaned, flags=re.IGNORECASE):
        return False
    if re.match(r"^\d+(?:\s*[,;]\s*\d+)*\s*\.\s+", cleaned):
        return False
    if re.match(r"^[АAВB]\d{1,2}\s*[.)]\s+", cleaned):
        return False
    if _looks_like_navigation_reference(cleaned):
        return False
    if normalized.startswith("см ") or "см разбор" in normalized or re.search(r"\bп\.\s", cleaned.lower()):
        return False
    if cleaned.count("(") > cleaned.count(")"):
        return False
    if re.fullmatch(r"[\d\s.,;:()]+", cleaned):
        return False
    if len(text) > 180:
        return False
    if len(text.split()) > 7 and not _looks_like_known_heading(cleaned):
        return False
    if re.fullmatch(r"[\d\s.:-]+", text):
        return False
    if _looks_like_sentence_instruction(cleaned):
        return False
    if text.endswith((".", ",", ";")) and len(text.split()) > 5:
        return False
    return True


def _normalize_subject(subject: str) -> str:
    normalized = (subject or "unknown").strip().lower()
    aliases = {
        "russian": "russian_language",
        "русский": "russian_language",
        "русский язык": "russian_language",
        "обж": "safety",
        "civil_defense": "safety",
        "история": "history",
        "математика": "math",
        "биология": "biology",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in SUPPORTED_SUBJECTS else "unknown"
