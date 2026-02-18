from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from app.config import settings
from app.schemas import ParsedElement
from app.utils import normalize_text


logger = logging.getLogger("rag_service")


def log_dependency_compatibility() -> None:
    versions = {
        "transformers": _safe_version("transformers"),
        "torch": _safe_version("torch"),
        "mineru": _safe_version("mineru"),
        "raganything": _safe_version("raganything"),
        "sentence_transformers": _safe_version("sentence-transformers"),
        "huggingface_hub": _safe_version("huggingface-hub") or _safe_version("huggingface_hub"),
        "accelerate": _safe_version("accelerate"),
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
                "recommendation": "Use separate mineru venv via MINERU_PYTHON; core keeps transformers==4.35.0",
            },
        )


@dataclass(frozen=True)
class MineruCliCaps:
    supports_json: bool
    output_dir_flag: str | None
    disable_image_flag: str | None
    disable_table_flag: str | None
    disable_equation_flag: str | None


class RAGAnythingParser:
    """Parser adapter that executes MinerU in a separate python environment via subprocess."""

    def parse_file(self, source_uri: str, path: Path) -> list[ParsedElement]:
        elements, _ = self.parse_file_with_mode(source_uri, path)
        return elements

    def parse_file_with_mode(self, source_uri: str, path: Path) -> tuple[list[ParsedElement], str]:
        rag_elements, reason = self._parse_with_mineru(path)
        if rag_elements:
            return self._normalize_elements(rag_elements), "rag_anything"

        logger.info(
            "parser_fallback_used",
            extra={"source_uri": source_uri, "path": str(path), "reason": reason},
        )
        return self._fallback_parse(path), "fallback"

    def _parse_with_mineru(self, path: Path) -> tuple[list[dict] | None, str]:
        try:
            result = self._run_mineru_subprocess(path, text_only=False)
            elements, reason = _extract_elements(result)
            if elements:
                return elements, "ok"
            return None, reason
        except Exception as exc:
            if _is_mineru_transformers_mismatch(exc):
                logger.error(
                    "dependency_mismatch",
                    extra={
                        "component": "mineru",
                        "path": str(path),
                        "symptom": "cache_position / UnimerMBartForCausalLM incompatibility",
                        "recommendation": 'pip install "transformers==4.35.0" in core and keep MinerU in separate venv',
                    },
                )
                logger.warning(
                    "parser_degraded_mode_used",
                    extra={"path": str(path), "reason": "mineru_transformers_incompatible", "mode": "text_only_retry"},
                )
                try:
                    result = self._run_mineru_subprocess(path, text_only=True)
                    elements, reason = _extract_elements(result)
                    if elements:
                        return elements, "ok_text_only_retry"
                    return None, f"text_only_retry:{reason}"
                except Exception as retry_exc:
                    logger.warning(
                        "mineru_execution_error",
                        extra={"path": str(path), "error_type": type(retry_exc).__name__, "error": str(retry_exc)},
                    )
                    return None, f"dependency_mismatch:{type(retry_exc).__name__}"

            logger.warning(
                "mineru_execution_error",
                extra={"path": str(path), "error_type": type(exc).__name__, "error": str(exc)},
            )
            return None, f"exception:{type(exc).__name__}"

    def _run_mineru_subprocess(self, path: Path, *, text_only: bool) -> Any:
        caps = self._detect_mineru_cli_caps(settings.mineru_python)
        with tempfile.TemporaryDirectory(prefix="mineru_out_") as tmpdir:
            out_dir = Path(tmpdir)
            cmd = self._build_mineru_command(path=path, text_only=text_only, output_dir=out_dir, caps=caps)

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=settings.mineru_timeout_seconds,
                check=False,
            )

            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            if proc.returncode != 0:
                raise RuntimeError(f"mineru_returncode={proc.returncode}\ncmd={' '.join(cmd)}\nstdout={stdout}\nstderr={stderr}")

            if caps.supports_json and stdout:
                try:
                    return json.loads(stdout)
                except json.JSONDecodeError:
                    logger.info("mineru_stdout_not_json", extra={"path": str(path)})

            return _read_mineru_output_dir(out_dir)

    @staticmethod
    @lru_cache(maxsize=4)
    def _detect_mineru_cli_caps(mineru_python: str) -> MineruCliCaps:
        parse_help = _run_help_command([mineru_python, "-m", "mineru.cli.client", "parse_doc", "--help"])
        full_help = _run_help_command([mineru_python, "-m", "mineru.cli.client", "--help"])
        combined = f"{parse_help}\n{full_help}".lower()

        output_flag = None
        if "--output-dir" in combined:
            output_flag = "--output-dir"
        elif "--output_dir" in combined:
            output_flag = "--output_dir"

        return MineruCliCaps(
            supports_json="--json" in combined,
            output_dir_flag=output_flag,
            disable_image_flag="--disable-image" if "--disable-image" in combined else None,
            disable_table_flag="--disable-table" if "--disable-table" in combined else None,
            disable_equation_flag="--disable-equation" if "--disable-equation" in combined else None,
        )

    @staticmethod
    def _build_mineru_command(*, path: Path, text_only: bool, output_dir: Path, caps: MineruCliCaps) -> list[str]:
        cmd = [
            settings.mineru_python,
            "-m",
            "mineru.cli.client",
            "parse_doc",
            str(path),
        ]

        if caps.output_dir_flag:
            cmd.extend([caps.output_dir_flag, str(output_dir)])

        if caps.supports_json:
            cmd.append("--json")

        if text_only:
            if caps.disable_image_flag:
                cmd.append(caps.disable_image_flag)
            if caps.disable_table_flag:
                cmd.append(caps.disable_table_flag)
            if caps.disable_equation_flag:
                cmd.append(caps.disable_equation_flag)

        return cmd

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


def _run_help_command(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except Exception:
        return ""
    return f"{proc.stdout or ''}\n{proc.stderr or ''}"


def _read_mineru_output_dir(output_dir: Path) -> Any:
    json_candidates = sorted(output_dir.rglob("*.json"))
    for candidate in json_candidates:
        try:
            return json.loads(candidate.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

    text_candidates = sorted([*output_dir.rglob("*.md"), *output_dir.rglob("*.txt")])
    for candidate in text_candidates:
        text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return {"elements": [{"type": "text", "text": text}]}

    raise RuntimeError(f"mineru_output_not_found:{output_dir}")


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
    has_cache_position = "cache_position" in text
    has_unimer = "unimer" in text or "unimermbartforcausallm" in text
    return has_cache_position and has_unimer


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
