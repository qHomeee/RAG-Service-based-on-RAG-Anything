import logging

from app.config import settings

logger = logging.getLogger("rag_service")


class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(settings.reranker_model)
        except Exception:
            self._model = None
            logger.warning("cross_encoder_unavailable", extra={"reranker_model": settings.reranker_model})

    @property
    def available(self) -> bool:
        return self._model is not None

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self._model is None:
            return [0.0 for _ in passages]
        pairs = [[query, p] for p in passages]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]
