from __future__ import annotations

import importlib
import logging
from pathlib import Path

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
            # API shape may vary between revisions; defensive adapter.
            rag_pipeline = self._load_rag_pipeline_module()

            parser = rag_pipeline.ParsingPipeline()
            result = parser.parse(str(path))
            if not isinstance(result, dict):
                return None, f"unexpected_result_type:{type(result).__name__}"

            elements = result.get("elements", [])
            if not elements:
                return None, "empty_elements"
            return elements, "ok"
        except Exception as exc:
            logger.warning(
                "rag_anything_parse_failed",
                extra={"path": str(path), "error_type": type(exc).__name__, "error": str(exc)},
            )
            return None, f"exception:{type(exc).__name__}"


    @staticmethod
    def _load_rag_pipeline_module():
        module_names = ("rag_anything.pipeline", "raganything.pipeline")
        last_exc: Exception | None = None
        for module_name in module_names:
            try:
                return importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - tested via failure path
                last_exc = exc

        if last_exc is None:
            raise ModuleNotFoundError("RAG-Anything pipeline module is unavailable")
        raise last_exc

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
