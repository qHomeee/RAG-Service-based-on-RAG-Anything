from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from app.schemas import ParsedElement
from app.utils import normalize_text


logger = logging.getLogger("rag_service")


def log_dependency_compatibility() -> None:
    versions = {
        "transformers": _safe_version("transformers"),
        "torch": _safe_version("torch"),
        "mineru": _safe_version("mineru"),
        "raganything": _safe_version("raganything"),
    }
    logger.info("dependency_versions", extra=versions)

    transformers_ver = versions["transformers"]
    mineru_ver = versions["mineru"]
    if mineru_ver and transformers_ver and _is_transformers_likely_incompatible(transformers_ver):
        logger.error(
            "dependency_mismatch",
            extra={
                "component": "mineru",
                "transformers_version": transformers_ver,
                "mineru_version": mineru_ver,
                "symptom": "cache_position / UnimerMBartForCausalLM incompatibility",
                "recommendation": 'pip install "transformers==4.35.0"',
            },
        )


class RAGAnythingParser:
    """Adapter over HKUDS/RAG-Anything parsing pipeline.

    If the package is unavailable in runtime, it falls back to lightweight local parsers
    to keep the service operational in constrained environments.
    """

    def parse_file(self, source_uri: str, path: Path) -> list[ParsedElement]:
        elements, _ = self.parse_file_with_mode(source_uri, path)
        return elements

    def parse_file_with_mode(self, source_uri: str, path: Path) -> tuple[list[ParsedElement], str]:
        rag_elements, reason = self._parse_with_rag_anything(path)
        if rag_elements:
            return self._normalize_elements(rag_elements), "rag_anything"

        logger.info(
            "parser_fallback_used",
            extra={"source_uri": source_uri, "path": str(path), "reason": reason},
        )
        return self._fallback_parse(path), "fallback"

    def _parse_with_rag_anything(self, path: Path) -> tuple[list[dict] | None, str]:
        try:
            parse_callable = self._load_rag_parse_callable(text_only=False)
            result = parse_callable(str(path))
            if inspect.isawaitable(result):
                result = _run_awaitable(result)
            elements, reason = _extract_elements(result)
            if not elements:
                return None, reason
            return elements, "ok"
        except Exception as exc:
            if _is_mineru_transformers_mismatch(exc):
                logger.error(
                    "dependency_mismatch",
                    extra={
                        "component": "mineru",
                        "path": str(path),
                        "symptom": "cache_position / UnimerMBartForCausalLM incompatibility",
                        "recommendation": 'pip install "transformers==4.35.0"',
                    },
                )
                logger.warning(
                    "rag_anything_retry_text_only",
                    extra={"path": str(path), "reason": "mineru_transformers_incompatible"},
                )
                try:
                    parse_callable = self._load_rag_parse_callable(text_only=True)
                    result = parse_callable(str(path))
                    if inspect.isawaitable(result):
                        result = _run_awaitable(result)
                    elements, reason = _extract_elements(result)
                    if elements:
                        return elements, "ok_text_only_retry"
                    return None, f"text_only_retry:{reason}"
                except Exception as retry_exc:
                    logger.warning(
                        "rag_anything_text_only_retry_failed",
                        extra={
                            "path": str(path),
                            "error_type": type(retry_exc).__name__,
                            "error": str(retry_exc),
                        },
                    )
                    return None, f"dependency_mismatch:{type(retry_exc).__name__}"

            logger.warning(
                "rag_anything_parse_failed",
                extra={"path": str(path), "error_type": type(exc).__name__, "error": str(exc)},
            )
            return None, f"exception:{type(exc).__name__}"

    @staticmethod
    def _load_rag_parse_callable(*, text_only: bool) -> Callable[[str], Any]:
        module_names = (
            "raganything",
            "raganything.parser",
        )
        class_names = (
            "RAGAnything",
            "RagAnything",
            "RAGAnythingParser",
            "DocumentParser",
            "BatchParser",
            "ParsingPipeline",
            "Parser",
        )
        method_names = ("parse_document", "process_folder_complete", "parse", "parse_file", "run", "process")
        function_names = ("parse_document", "process_folder_complete", "parse", "parse_file", "run", "process")

        init_kwargs = {
            "enable_image": not text_only,
            "enable_table": not text_only,
            "enable_equation": not text_only,
            "parser": "mineru",
            "parse_method": "auto",
            "max_concurrent_files": 1,
        }

        last_exc: Exception | None = None
        attempted: list[str] = []

        for module_name in module_names:
            attempted.append(module_name)
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - tested via failure path
                last_exc = exc
                continue

            parse_callable = _resolve_parse_callable(
                module,
                class_names,
                method_names,
                function_names,
                init_kwargs=init_kwargs,
            )
            if parse_callable is not None:
                return parse_callable

        if last_exc is not None:
            raise ModuleNotFoundError(
                f"RAG-Anything parser entrypoint not found. attempted={attempted}; last_error={type(last_exc).__name__}: {last_exc}"
            ) from last_exc
        raise ModuleNotFoundError(f"RAG-Anything parser entrypoint not found. attempted={attempted}")

    def _normalize_elements(self, elements: list[dict]) -> list[ParsedElement]:
        normalized: list[ParsedElement] = []
        for idx, item in enumerate(elements):
            elem_type = str(item.get("type", "text")).lower()
            content = item.get("content") or item.get("text") or ""
            page = item.get("page")
            meta = {k: v for k, v in item.items() if k not in {"content", "text", "type", "page"}}
            normalized.append(
                ParsedElement(
                    element_index=idx,
                    type=elem_type if elem_type in {"text", "table", "image", "equation"} else "text",
                    content=normalize_text(str(content)),
                    page=page if isinstance(page, int) else None,
                    meta=meta,
                )
            )
        return [x for x in normalized if x.content]

    def _fallback_parse(self, path: Path) -> list[ParsedElement]:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return [ParsedElement(element_index=0, type="text", content=normalize_text(text), meta={"fallback": True})]
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                elems: list[ParsedElement] = []
                for i, page in enumerate(reader.pages):
                    text = normalize_text(page.extract_text() or "")
                    if text:
                        elems.append(ParsedElement(element_index=i, type="text", content=text, page=i + 1, meta={"fallback": True}))
                return elems
            except Exception:
                return []
        if suffix in {".docx"}:
            try:
                import docx

                document = docx.Document(str(path))
                text = normalize_text("\n".join(p.text for p in document.paragraphs))
                if text:
                    return [ParsedElement(element_index=0, type="text", content=text, meta={"fallback": True})]
            except Exception:
                return []
        return []


def _safe_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _is_transformers_likely_incompatible(version_str: str) -> bool:
    try:
        major, minor, *_ = [int(p) for p in version_str.split(".")]
    except Exception:
        return False
    return (major, minor) >= (4, 36)


def _extract_error_text(exc: Exception) -> str:
    parts = [str(exc)]
    args = getattr(exc, "args", ())
    parts.extend(str(a) for a in args)
    for attr in ("stdout", "stderr", "output", "message"):
        val = getattr(exc, attr, None)
        if val:
            parts.append(str(val))
    return "\n".join(parts)


def _is_mineru_transformers_mismatch(exc: Exception) -> bool:
    text = _extract_error_text(exc).lower()
    return "cache_position" in text or "unimermbartforcausallm" in text


def _extract_elements(result: Any) -> tuple[list[dict] | None, str]:
    if isinstance(result, dict):
        elements = result.get("elements", [])
        if isinstance(elements, list) and elements:
            return elements, "ok"
        return None, "empty_elements"

    if isinstance(result, list):
        if result and isinstance(result[0], dict):
            return result, "ok"
        return None, "unexpected_list_payload"

    return None, f"unexpected_result_type:{type(result).__name__}"


def _resolve_parse_callable(
    module: Any,
    class_names: tuple[str, ...],
    method_names: tuple[str, ...],
    function_names: tuple[str, ...],
    *,
    init_kwargs: dict[str, Any],
) -> Callable[[str], Any] | None:
    search_spaces = [module]
    for nested_name in ("pipeline", "parser", "processor", "raganything"):
        nested = getattr(module, nested_name, None)
        if nested is not None:
            search_spaces.append(nested)

    for space in search_spaces:
        for class_name in class_names:
            cls = getattr(space, class_name, None)
            callable_ = _build_callable_from_class(cls, method_names, init_kwargs)
            if callable_ is not None:
                return callable_

    for space in search_spaces:
        for function_name in function_names:
            fn = getattr(space, function_name, None)
            if callable(fn):
                return fn

    for space in search_spaces:
        for _, member in inspect.getmembers(space, inspect.isclass):
            callable_ = _build_callable_from_class(member, method_names, init_kwargs)
            if callable_ is not None:
                return callable_

    return None


def _build_callable_from_class(
    cls: Any,
    method_names: tuple[str, ...],
    init_kwargs: dict[str, Any],
) -> Callable[[str], Any] | None:
    if not isinstance(cls, type):
        return None

    instance = None
    try:
        instance = cls(**init_kwargs)
    except Exception:
        try:
            instance = cls()
        except Exception:
            return None

    for method_name in method_names:
        method = getattr(instance, method_name, None)
        if callable(method):
            return method
    return None


def _run_awaitable(awaitable: Any) -> Any:
    """Execute awaitable from sync context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["value"] = asyncio.run(awaitable)
        except BaseException as exc:  # pragma: no cover
            error_box["error"] = exc

    import threading

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")
