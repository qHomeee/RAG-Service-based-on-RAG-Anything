from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Callable

from app.schemas import ParsedElement
from app.utils import normalize_text


logger = logging.getLogger("rag_service")


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
            parse_callable = self._load_rag_parse_callable()
            result = parse_callable(str(path))
            elements, reason = _extract_elements(result)
            if not elements:
                return None, reason
            return elements, "ok"
        except Exception as exc:
            logger.warning(
                "rag_anything_parse_failed",
                extra={"path": str(path), "error_type": type(exc).__name__, "error": str(exc)},
            )
            return None, f"exception:{type(exc).__name__}"

    @staticmethod
    def _load_rag_parse_callable() -> Callable[[str], Any]:
        module_names = (
            "rag_anything.pipeline",
            "raganything.pipeline",
            "rag_anything.parser",
            "raganything.parser",
            "rag_anything.raganything",
            "raganything.raganything",
            "rag_anything",
            "raganything",
        )
        class_names = (
            "ParsingPipeline",
            "RAGAnything",
            "RagAnything",
            "RAGAnythingParser",
            "Parser",
            "DocumentParser",
            "BatchParser",
        )
        method_names = ("parse", "parse_file", "run", "process")
        function_names = ("parse", "parse_file", "run", "process")

        last_exc: Exception | None = None
        attempted: list[str] = []

        for module_name in module_names:
            attempted.append(module_name)
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - tested via failure path
                last_exc = exc
                continue

            parse_callable = _resolve_parse_callable(module, class_names, method_names, function_names)
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
) -> Callable[[str], Any] | None:
    search_spaces = [module]
    for nested_name in ("pipeline", "parser", "processor", "raganything"):
        nested = getattr(module, nested_name, None)
        if nested is not None:
            search_spaces.append(nested)

    # Preferred explicit class names first.
    for space in search_spaces:
        for class_name in class_names:
            cls = getattr(space, class_name, None)
            callable_ = _build_callable_from_class(cls, method_names)
            if callable_ is not None:
                return callable_

    # Then module-level functions.
    for space in search_spaces:
        for function_name in function_names:
            fn = getattr(space, function_name, None)
            if callable(fn):
                return fn

    # Finally scan any class exposing parse-like methods.
    for space in search_spaces:
        for _, member in inspect.getmembers(space, inspect.isclass):
            callable_ = _build_callable_from_class(member, method_names)
            if callable_ is not None:
                return callable_

    return None


def _build_callable_from_class(cls: Any, method_names: tuple[str, ...]) -> Callable[[str], Any] | None:
    if not isinstance(cls, type):
        return None

    try:
        instance = cls()
    except Exception:
        return None

    for method_name in method_names:
        method = getattr(instance, method_name, None)
        if callable(method):
            return method
    return None
