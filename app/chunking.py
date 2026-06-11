import re
from dataclasses import dataclass

from app.config import settings
from app.utils import normalize_text


@dataclass
class StructuredChunk:
    text: str
    heading_path: list[str]


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_TABLE_LINE_RE = re.compile(r"\|.+\||\t+| {3,}")
_FAQ_LINE_RE = re.compile(r"^(q:|question:|вопрос:|a:|answer:|ответ:)", re.IGNORECASE)


def split_to_subchunks(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def split_structured_chunks(
    text: str,
    *,
    min_size: int | None = None,
    max_size: int | None = None,
) -> list[StructuredChunk]:
    min_size = min_size or settings.adaptive_chunk_min_chars
    max_size = max_size or settings.adaptive_chunk_max_chars
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text or "") if b.strip()]
    if not blocks:
        return []

    chunks: list[StructuredChunk] = []
    heading_path: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        merged = normalize_text("\n\n".join(buffer))
        if not merged:
            buffer.clear()
            return
        local_min_size, local_max_size = _semantic_chunk_bounds(merged, min_size=min_size, max_size=max_size)
        for part in _split_without_breaking_sentences(merged, min_size=local_min_size, max_size=local_max_size):
            chunks.append(StructuredChunk(text=_attach_heading_context(part, heading_path), heading_path=list(heading_path)))
        buffer.clear()

    for block_idx, block in enumerate(blocks):
        heading_match = _HEADING_RE.match(block)
        if heading_match:
            flush_buffer()
            level = len(heading_match.group(1))
            heading = normalize_text(heading_match.group(2))
            if heading:
                heading_path[:] = heading_path[: level - 1]
                heading_path.append(heading)
            continue

        block_lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(block_lines) == 1 and block_idx < len(blocks) - 1 and _looks_like_heading_line(block_lines[0]):
            flush_buffer()
            heading = normalize_text(block_lines[0])
            if heading:
                if heading_path:
                    heading_path[-1] = heading
                else:
                    heading_path.append(heading)
            continue
        if len(block_lines) > 1 and _looks_like_heading_line(block_lines[0]):
            flush_buffer()
            heading = normalize_text(block_lines[0])
            if heading:
                if heading_path:
                    heading_path[-1] = heading
                else:
                    heading_path.append(heading)
            block = "\n".join(block_lines[1:]).strip()
            if not block:
                continue

        candidate = normalize_text("\n\n".join(buffer + [block]))
        if buffer and len(candidate) > max_size:
            flush_buffer()

        buffer.append(block)
        current_len = len(normalize_text("\n\n".join(buffer)))
        if current_len >= min_size:
            flush_buffer()

    flush_buffer()
    return _postprocess_chunk_boundaries(chunks, max_size=max_size)


def _split_without_breaking_sentences(text: str, *, min_size: int, max_size: int) -> list[str]:
    if len(text) <= max_size:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return _split_on_word_boundaries(text, max_size=max_size)

    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if buf and len(candidate) > max_size:
            chunks.append(buf)
            buf = sentence
            continue
        buf = candidate
        if len(buf) >= min_size:
            chunks.append(buf)
            buf = ""

    if buf:
        if chunks and len(buf) < min_size:
            chunks[-1] = normalize_text(f"{chunks[-1]} {buf}")
        else:
            chunks.append(buf)
    return [normalize_text(c) for c in chunks if c.strip()]


def _split_on_word_boundaries(text: str, *, max_size: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    buf = ""
    for word in words:
        candidate = f"{buf} {word}".strip() if buf else word
        if buf and len(candidate) > max_size:
            chunks.append(buf)
            buf = word
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks or [text[:max_size]]


def _postprocess_chunk_boundaries(chunks: list[StructuredChunk], *, max_size: int) -> list[StructuredChunk]:
    processed: list[StructuredChunk] = []
    for chunk in chunks:
        text = normalize_text(chunk.text)
        if not text:
            continue
        if processed and chunk.heading_path == processed[-1].heading_path and _should_merge_with_previous(text, max_size=max_size):
            merged_text = normalize_text(f"{processed[-1].text} {text}")
            if len(merged_text) <= max_size + settings.chunk_overlap:
                processed[-1] = StructuredChunk(text=merged_text, heading_path=processed[-1].heading_path)
                continue
        processed.append(StructuredChunk(text=text, heading_path=chunk.heading_path))
    return processed


def _should_merge_with_previous(text: str, *, max_size: int) -> bool:
    if len(text) < max(220, settings.chunk_overlap):
        return True
    first_alpha = re.search(r"[A-Za-zА-Яа-яЁё]", text)
    if first_alpha and first_alpha.group(0).islower():
        return True
    return len(text) < max_size // 3 and not re.search(r"^[A-ZА-ЯЁ0-9#§]", text)


def _looks_like_heading_line(text: str) -> bool:
    cleaned = normalize_text(re.sub(r"^[§\d.\s-]+", "", text or ""))
    if not cleaned or len(cleaned) > 140:
        return False
    words = cleaned.split()
    if not (1 <= len(words) <= 10):
        return False
    if re.search(r"[.!?]\s*$", cleaned):
        return False
    if re.fullmatch(r"[\d\s.:-]+", cleaned):
        return False
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", cleaned)
    uppercase = re.findall(r"[A-ZА-ЯЁ]", cleaned)
    if letters and len(uppercase) / len(letters) >= 0.55:
        return True
    return len(words) <= 6 and bool(re.search(r"[A-ZА-ЯЁ]", cleaned[:1]))


def _semantic_chunk_bounds(text: str, *, min_size: int, max_size: int) -> tuple[int, int]:
    if not settings.semantic_chunking_enabled:
        return min_size, max_size

    semantic_max = max_size
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(_TABLE_LINE_RE.search(line) for line in lines):
        semantic_max = min(semantic_max, settings.semantic_table_chunk_max_chars)
    if any(_FAQ_LINE_RE.search(line) for line in lines):
        semantic_max = min(semantic_max, settings.semantic_faq_chunk_max_chars)

    semantic_min = max(300, min_size // 2) if semantic_max < max_size else min_size
    return min(semantic_min, semantic_max), semantic_max


def _attach_heading_context(text: str, heading_path: list[str]) -> str:
    if not heading_path:
        return text
    heading = normalize_text(" / ".join(heading_path))
    if not heading or text.lower().startswith(heading.lower()):
        return text
    return normalize_text(f"{heading}\n\n{text}")
