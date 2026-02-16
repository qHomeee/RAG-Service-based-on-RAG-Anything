from __future__ import annotations

from pathlib import Path

from app.schemas import ParsedElement
from app.utils import normalize_text


class RAGAnythingParser:
    """Adapter over HKUDS/RAG-Anything parsing pipeline.

    If the package is unavailable in runtime, it falls back to lightweight local parsers
    to keep the service operational in constrained environments.
    """

    def parse_file(self, source_uri: str, path: Path) -> list[ParsedElement]:
        rag_elements = self._parse_with_rag_anything(path)
        if rag_elements:
            return self._normalize_elements(rag_elements)
        return self._fallback_parse(path)

    def _parse_with_rag_anything(self, path: Path) -> list[dict] | None:
        try:
            # API shape may vary between revisions; defensive adapter.
            from rag_anything import pipeline as rag_pipeline  # type: ignore

            parser = rag_pipeline.ParsingPipeline()
            result = parser.parse(str(path))
            return result.get("elements", []) if isinstance(result, dict) else []
        except Exception:
            return None

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
