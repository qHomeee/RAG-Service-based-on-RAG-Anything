import pytest

from app.utils import cosine_similarity


class AmbiguousVector(list):
    def __bool__(self):  # pragma: no cover
        raise ValueError("ambiguous truth value")


def test_cosine_similarity_handles_vectors_with_ambiguous_truthiness():
    vec_a = AmbiguousVector([1.0, 0.0, 1.0])
    vec_b = AmbiguousVector([1.0, 0.0, 1.0])

    score = cosine_similarity(vec_a, vec_b)

    assert score == pytest.approx(1.0)
