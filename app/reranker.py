import logging
import threading

from app.config import settings

logger = logging.getLogger("rag_service")


class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._load_error: str | None = None
        self.model_name = settings.reranker_model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(settings.reranker_model)
            logger.info("cross_encoder_loaded", extra={"reranker_model": settings.reranker_model})
        except Exception as exc:
            self._model = None
            self._load_error = str(exc)
            logger.warning(
                "cross_encoder_unavailable",
                extra={"reranker_model": settings.reranker_model, "error": self._load_error},
            )

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self._model is None:
            return [0.0 for _ in passages]
        pairs = [[query, p] for p in passages]
        with self._lock:
            scores = self._model.predict(pairs)
        return [float(s) for s in scores]
