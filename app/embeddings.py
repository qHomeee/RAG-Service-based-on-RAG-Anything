import hashlib
import logging
import random
import re
import threading
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from app.config import settings


logger = logging.getLogger("rag_service")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EmbeddingModelReference:
    configured: str
    model_arg: str
    source_type: str
    local_path: Path | None
    local_files_only: bool


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
            "huggingface_hub": _safe_version("huggingface-hub") or _safe_version("huggingface_hub"),
            "sentence_transformers": _safe_version("sentence-transformers"),
            "torch": _safe_version("torch"),
        }

        try:
            model_ref = _embedding_model_reference(settings.embed_model, embed_offline=settings.embed_offline)
            _validate_embedding_model_reference(model_ref)

            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_ref.model_arg, local_files_only=model_ref.local_files_only)
        except Exception as exc:
            self._model = None
            self.using_fallback = True
            error_text = _extract_error_text(exc)
            model_ref = _safe_embedding_model_reference(settings.embed_model, embed_offline=settings.embed_offline)
            logger.warning(
                "embedding_model_unavailable",
                extra={
                    "embed_model": settings.embed_model,
                    "embed_offline": settings.embed_offline,
                    "model_source": model_ref.source_type,
                    "resolved_local_path": str(model_ref.local_path) if model_ref.local_path else None,
                    "local_files_only": model_ref.local_files_only,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if _is_embedding_dependency_mismatch(error_text):
                logger.error(
                    "embedding_dependency_mismatch",
                    extra={
                        **versions,
                        "symptom": "huggingface_hub/accelerate/transformers import incompatibility",
                        "recommendation": [
                            "pip uninstall -y accelerate",
                            'pip install --no-cache-dir "huggingface-hub>=0.16.4,<0.18" "tokenizers==0.14.1"',
                            "pip install --no-cache-dir -r requirements.txt",
                        ],
                    },
                )

            if settings.fail_on_embedding_fallback:
                raise RuntimeError(_embedding_model_error_message(exc, model_ref, versions=versions)) from exc

            logger.warning(
                "embedding_provider_degraded",
                extra={
                    **versions,
                    "embed_model": settings.embed_model,
                    "embed_offline": settings.embed_offline,
                    "model_source": model_ref.source_type,
                    "local_files_only": model_ref.local_files_only,
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


def _embedding_model_reference(model_name_or_path: str, *, embed_offline: bool) -> EmbeddingModelReference:
    configured = str(model_name_or_path or "").strip().strip('"').strip("'")
    if not configured:
        configured = "all-MiniLM-L6-v2"

    if _looks_like_local_model_path(configured):
        local_path = _resolve_local_model_path(configured)
        return EmbeddingModelReference(
            configured=configured,
            model_arg=str(local_path),
            source_type="local_path",
            local_path=local_path,
            local_files_only=True,
        )

    return EmbeddingModelReference(
        configured=configured,
        model_arg=configured,
        source_type="huggingface_model_id",
        local_path=None,
        local_files_only=bool(embed_offline),
    )


def _safe_embedding_model_reference(model_name_or_path: str, *, embed_offline: bool) -> EmbeddingModelReference:
    try:
        return _embedding_model_reference(model_name_or_path, embed_offline=embed_offline)
    except Exception:
        configured = str(model_name_or_path or "").strip()
        return EmbeddingModelReference(
            configured=configured,
            model_arg=configured,
            source_type="unknown",
            local_path=None,
            local_files_only=bool(embed_offline),
        )


def _looks_like_local_model_path(value: str) -> bool:
    raw = value.strip()
    if not raw:
        return False
    path = Path(raw).expanduser()
    normalized = raw.replace("\\", "/")
    local_prefixes = ("./", "../", "~/", "/", "storage/", "models/")
    if path.exists() or path.is_absolute():
        return True
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        return True
    return normalized.startswith(local_prefixes)


def _resolve_local_model_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value):
        return path.resolve()

    project_candidate = PROJECT_ROOT / path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists() and not project_candidate.exists():
        return cwd_candidate.resolve()
    return project_candidate.resolve()


def _validate_embedding_model_reference(model_ref: EmbeddingModelReference) -> None:
    if model_ref.source_type != "local_path":
        return
    path = model_ref.local_path
    if path is None:
        raise ValueError("EMBED_MODEL was detected as a local path but could not be resolved.")
    if not path.exists():
        raise FileNotFoundError(f"Local embedding model path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Local embedding model path is not a directory: {path}")
    modules_json = path / "modules.json"
    if not modules_json.is_file():
        raise FileNotFoundError(
            f"Local embedding model path is missing modules.json: {path}. "
            "Use SentenceTransformer(...).save(...) to create a valid local SentenceTransformer model."
        )


def _embedding_model_error_message(exc: Exception, model_ref: EmbeddingModelReference, *, versions: dict[str, str | None]) -> str:
    offline = "true" if settings.embed_offline else "false"
    source_label = "local path" if model_ref.source_type == "local_path" else "HuggingFace model id"
    path_line = f"\nResolved local path: {model_ref.local_path}" if model_ref.local_path else ""
    dependency_hint = ""
    if _is_embedding_dependency_mismatch(_extract_error_text(exc)):
        dependency_hint = (
            "\n\nDependency repair, if this is a package mismatch:\n"
            "pip uninstall -y accelerate huggingface-hub transformers tokenizers\n"
            "pip install --no-cache-dir -r requirements.txt\n"
            "Optional accelerate: pip install -r requirements-accelerate.txt"
        )
    return (
        "Embedding model is not available.\n"
        f"Configured EMBED_MODEL={model_ref.configured}\n"
        f"Model source: {source_label}{path_line}\n"
        f"EMBED_OFFLINE={offline}\n"
        f"local_files_only={model_ref.local_files_only}\n"
        "Strict mode is enabled by FAIL_ON_EMBEDDING_FALLBACK=true, so startup was stopped.\n\n"
        "If you want offline/local startup, download the model once:\n"
        "Windows PowerShell:\n"
        'python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(\'sentence-transformers/all-MiniLM-L6-v2\').save(\'storage/models/all-MiniLM-L6-v2\')"\n\n'
        "Linux/bash:\n"
        "python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer(\"sentence-transformers/all-MiniLM-L6-v2\").save(\"storage/models/all-MiniLM-L6-v2\")'\n\n"
        "Then set in .env:\n"
        "EMBED_MODEL=storage/models/all-MiniLM-L6-v2\n"
        "EMBED_OFFLINE=true\n\n"
        "For Docker/Linux production, copy or mount storage/models/all-MiniLM-L6-v2 into the container and keep EMBED_OFFLINE=true.\n"
        f"Installed ML package versions: {versions}\n"
        f"Original error: {type(exc).__name__}: {exc}"
        f"{dependency_hint}"
    )
