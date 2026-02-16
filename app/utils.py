import hashlib
import math
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
    if vec_a is None or vec_b is None:
        return 0.0
    if len(vec_a) == 0 or len(vec_b) == 0 or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    nb = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
