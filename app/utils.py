import hashlib
import re


def normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    return compact


def snippet_from_text(text: str, min_len: int = 200, max_len: int = 400) -> str:
    norm = normalize_text(text)
    if len(norm) <= max_len:
        return norm
    clipped = norm[:max_len]
    if len(clipped) < min_len:
        return norm[:min_len]
    return clipped


def stable_fragment_id(source_uri: str, element_index: int, content: str) -> str:
    prefix = normalize_text(content)[:512]
    payload = f"{source_uri}|{element_index}|{prefix}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = sum(a * a for a in vec_a) ** 0.5
    nb = sum(b * b for b in vec_b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
