import hashlib
import logging
import random
import threading
from importlib import metadata

from app.config import settings


logger = logging.getLogger("rag_service")


def _safe_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except Exception:
        return None


class EmbeddingProvider:
    def __init__(self) -> None:
        self.dim = settings.embed_dim
        self._model = None
        self._lock = threading.Lock()
        self.using_fallback = False

        versions = {
            "transformers": _safe_version("transformers"),
            "accelerate": _safe_version("accelerate"),
            "huggingface_hub": _safe_version("huggingface_hub"),
            "sentence_transformers": _safe_version("sentence-transformers"),
            "torch": _safe_version("torch"),
        }

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embed_model)
        except Exception as exc:
            self._model = None
            self.using_fallback = True
            error_text = _extract_error_text(exc)
            logger.warning(
                "embedding_model_unavailable",
                extra={"embed_model": settings.embed_model, "error_type": type(exc).__name__, "error": str(exc)},
            )
            if _is_embedding_dependency_mismatch(error_text):
                logger.error(
                    "embedding_dependency_mismatch",
                    extra={
                        **versions,
                        "symptom": "huggingface_hub/accelerate/transformers import incompatibility",
                        "recommendation": 'pip install "transformers==4.35.0" "accelerate>=0.24,<0.26" "huggingface_hub>=0.19.4"',
                    },
                )

            if settings.fail_on_embedding_fallback:
                raise RuntimeError(
                    "Embedding provider failed to initialize in strict mode. "
                    f"versions={versions}. "
                    "Recommended fix: pip uninstall -y accelerate huggingface_hub transformers && "
                    "pip install \"transformers==4.35.0\" \"accelerate>=0.24,<0.26\" \"huggingface_hub>=0.19.4\" "
                    "\"sentence-transformers>=2.2,<3\" safetensors"
                ) from exc

            logger.warning(
                "embedding_provider_degraded",
                extra={
                    **versions,
                    "embed_model": settings.embed_model,
                    "reason": type(exc).__name__,
                    "mode": "fallback_hash_embeddings",
                },
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


def _extract_error_text(exc: Exception) -> str:
    parts = [str(exc)]
    parts.extend(str(a) for a in getattr(exc, "args", ()))
    return "\n".join(parts).lower()


def _is_embedding_dependency_mismatch(error_text: str) -> bool:
    checks = (
        "split_torch_state_dict_into_shards",
        "huggingface_hub",
        "accelerate",
        "cannot import name",
    )
    return any(token in error_text for token in checks)
