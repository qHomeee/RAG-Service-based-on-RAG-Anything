import hashlib
import random

from app.config import settings


class EmbeddingProvider:
    def __init__(self) -> None:
        self.dim = settings.embed_dim
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embed_model)
        except Exception:
            self._model = None

    def embed(self, text: str) -> list[float]:
        if self._model is not None:
            values = self._model.encode(text, normalize_embeddings=True).tolist()
            if len(values) != self.dim:
                if len(values) > self.dim:
                    return values[: self.dim]
                return values + [0.0] * (self.dim - len(values))
            return values
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rnd = random.Random(seed)
        return [rnd.uniform(-1, 1) for _ in range(self.dim)]
