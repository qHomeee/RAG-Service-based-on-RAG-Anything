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
        for part in _split_without_breaking_sentences(merged, min_size=min_size, max_size=max_size):
            chunks.append(StructuredChunk(text=part, heading_path=list(heading_path)))
        buffer.clear()

    for block in blocks:
        heading_match = _HEADING_RE.match(block)
        if heading_match:
            flush_buffer()
            level = len(heading_match.group(1))
            heading = normalize_text(heading_match.group(2))
            if heading:
                heading_path[:] = heading_path[: level - 1]
                heading_path.append(heading)
            continue

        candidate = normalize_text("\n\n".join(buffer + [block]))
        if buffer and len(candidate) > max_size:
            flush_buffer()

        buffer.append(block)
        current_len = len(normalize_text("\n\n".join(buffer)))
        if current_len >= min_size:
            flush_buffer()

    flush_buffer()
    return chunks


def _split_without_breaking_sentences(text: str, *, min_size: int, max_size: int) -> list[str]:
    if len(text) <= max_size:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text[i : i + max_size] for i in range(0, len(text), max_size)]

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
