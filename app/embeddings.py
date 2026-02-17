import hashlib
import logging
import random
import threading

from app.config import settings


logger = logging.getLogger("rag_service")


class EmbeddingProvider:
    def __init__(self) -> None:
        self.dim = settings.embed_dim
        self._model = None
        self._lock = threading.Lock()
        self.using_fallback = False
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embed_model)
        except Exception:
            self._model = None
            self.using_fallback = True
            logger.warning("embedding_model_unavailable", extra={"embed_model": settings.embed_model})

        if self.using_fallback and settings.fail_on_embedding_fallback:
            raise RuntimeError(
                "Embedding model failed to load. Install sentence-transformers model/dependencies "
                "or set FAIL_ON_EMBEDDING_FALLBACK=false for non-production fallback mode."
            )

    def embed(self, text: str) -> list[float]:
        if self._model is not None:
            with self._lock:
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
