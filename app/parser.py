from __future__ import annotations

import json
import logging
import re
import subprocess
import shutil
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from app.config import settings
from app.schemas import ParsedElement
from app.utils import normalize_text


logger = logging.getLogger("rag_service")

MINERU_MODULE_TO_PACKAGE = {
    "fast_langdetect": "fast-langdetect",
    "doclayout_yolo": "doclayout-yolo",
    "huggingface_hub": "huggingface-hub",
    "ultralytics": "ultralytics",
    "torch": "torch",
}


def mineru_doctor(python_exe: str) -> dict[str, Any]:
    check_cmd = [
        python_exe,
        "-c",
        "import torch, ultralytics, doclayout_yolo, transformers, huggingface_hub; "
        "from fast_langdetect import detect_language; print('ok')",
    ]
    versions_cmd = [
        python_exe,
        "-c",
        "import mineru, torch, ultralytics, doclayout_yolo, transformers, huggingface_hub, fast_langdetect; "
        "print(mineru.__version__); print(torch.__version__); print(transformers.__version__); "
        "print(huggingface_hub.__version__); print(getattr(fast_langdetect, '__version__', 'unknown')); "
        "print(getattr(doclayout_yolo, '__version__', 'unknown')); print(getattr(ultralytics, '__version__', 'unknown'))",
    ]

    missing: list[dict[str, str]] = []
    versions: dict[str, str] = {
        "mineru": "unknown",
        "torch": "unknown",
        "transformers": "unknown",
        "huggingface_hub": "unknown",
        "fast_langdetect": "unknown",
        "doclayout_yolo": "unknown",
        "ultralytics": "unknown",
    }

    try:
        proc = subprocess.run(check_cmd, capture_output=True, text=True, check=False)
    except Exception as exc:
        module = _extract_missing_module(str(exc))
        if not module:
            module = "mineru_python"
        result = {
            "ok": False,
            "message": "mineru_missing_dependency",
            "missing": [{"module": module, "pip": _module_to_package(module) or module}],
            "versions": versions,
            "error": str(exc),
            "how_to_fix": "pip install -r requirements-mineru.txt",
        }
        logger.error("mineru_missing_dependency", extra={"missing_module": module, "suggested_package": _module_to_package(module), "how_to_fix": "pip install -r requirements-mineru.txt"})
        logger.warning("mineru_doctor", extra={"ok": False, "missing": result["missing"], "versions": versions})
        return result

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        module = _extract_missing_module(stderr)
        if module:
            missing.append({"module": module, "pip": _module_to_package(module) or module})
            logger.error(
                "mineru_missing_dependency",
                extra={
                    "missing_module": module,
                    "suggested_package": _module_to_package(module),
                    "how_to_fix": "pip install -r requirements-mineru.txt",
                },
            )
        result = {
            "ok": False,
            "message": "mineru_missing_dependency",
            "missing": missing,
            "versions": versions,
            "error": stderr or (proc.stdout or "").strip(),
            "how_to_fix": "pip install -r requirements-mineru.txt",
        }
        logger.warning("mineru_doctor", extra={"ok": False, "missing": missing, "versions": versions})
        return result

    vproc = subprocess.run(versions_cmd, capture_output=True, text=True, check=False)
    lines = [line.strip() for line in (vproc.stdout or "").splitlines() if line.strip()]
    versions = {
        "mineru": lines[0] if len(lines) > 0 else "unknown",
        "torch": lines[1] if len(lines) > 1 else "unknown",
        "transformers": lines[2] if len(lines) > 2 else "unknown",
        "huggingface_hub": lines[3] if len(lines) > 3 else "unknown",
        "fast_langdetect": lines[4] if len(lines) > 4 else "unknown",
        "doclayout_yolo": lines[5] if len(lines) > 5 else "unknown",
        "ultralytics": lines[6] if len(lines) > 6 else "unknown",
    }
    result = {"ok": True, "missing": [], "versions": versions}
    logger.info("mineru_doctor", extra={"ok": True, "missing": [], "versions": versions})
    return result


def check_mineru_runtime(mineru_python: str) -> dict[str, Any]:
    """Backward-compatible alias."""
    return mineru_doctor(mineru_python)


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
    path_flag: str
    output_dir_flag: str
    disable_image_flag: str | None
    disable_table_flag: str | None
    disable_equation_flag: str | None
    mode_flag: str | None
    formula_flag: str | None
    table_flag: str | None
    backend_flag: str | None
    device_flag: str | None


class RAGAnythingParser:
    """Parser adapter that executes MinerU in a separate python environment via subprocess."""

    def parse_file(self, source_uri: str, path: Path, reindex: bool = False) -> list[ParsedElement]:
        elements, _ = self.parse_file_with_mode(source_uri, path, reindex=reindex)
        return elements

    def parse_file_with_mode(self, source_uri: str, path: Path, reindex: bool = False) -> tuple[list[ParsedElement], str]:
        rag_elements, reason = self._parse_with_mineru(path, reindex=reindex)
        if rag_elements:
            return self._normalize_elements(rag_elements), "rag_anything"

        logger.info(
            "parser_fallback_used",
            extra={"source_uri": source_uri, "path": str(path), "reason": reason},
        )
        return self._fallback_parse(path), "fallback"

    def _parse_with_mineru(self, path: Path, reindex: bool = False) -> tuple[list[dict] | None, str]:
        doctor = mineru_doctor(settings.mineru_python)
        if not doctor.get("ok", False):
            missing = doctor.get("missing", [])
            logger.warning(
                "parser_degraded_mode_used",
                extra={"path": str(path), "mode": "fallback_text_parser", "missing": missing},
            )
            return None, "mineru_doctor_missing_deps"

        try:
            result = self._run_mineru_subprocess(path, text_only=False, reindex=reindex)
            elements, reason = _extract_elements(result)
            if elements:
                return elements, "ok"
            return None, reason
        except Exception as exc:
            if _should_retry_degraded(exc):
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
                    extra={"path": str(path), "reason": "mineru_error", "mode": "text_only_retry"},
                )
                try:
                    result = self._run_mineru_subprocess(path, text_only=True, reindex=reindex)
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

    def _run_mineru_subprocess(self, path: Path, *, text_only: bool, reindex: bool = False) -> Any:
        caps = self._detect_mineru_cli_caps(settings.mineru_python)
        output_dir, use_cached = self._prepare_output_dir(path, reindex=reindex)
        if use_cached:
            files = _collect_output_files(output_dir)
            logger.info(
                "mineru_output_files",
                extra={
                    "path": str(path),
                    "output_dir": str(output_dir),
                    "count": len(files),
                    "files": [str(file.relative_to(output_dir)) for file in files[:30]],
                    "cached": True,
                },
            )
            return _read_mineru_output_dir(output_dir, files)

        cmd = self._build_mineru_command(path=path, text_only=text_only, output_dir=output_dir, caps=caps)
        return self._execute_mineru_command(path=path, output_dir=output_dir, cmd=cmd)

    def _prepare_output_dir(self, path: Path, *, reindex: bool) -> tuple[Path, bool]:
        output_dir = Path(settings.storage_parsed).resolve() / _doc_dir_name(path)
        has_artifacts = output_dir.exists() and bool(_collect_output_files(output_dir))

        if reindex and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
            has_artifacts = False

        output_dir.mkdir(parents=True, exist_ok=True)

        if has_artifacts and not reindex:
            logger.info(
                "mineru_cached_output_used",
                extra={"path": str(path), "output_dir": str(output_dir)},
            )
            return output_dir, True
        return output_dir, False

    def _execute_mineru_command(self, *, path: Path, output_dir: Path, cmd: list[str]) -> Any:
        output_files = mineru_run_and_validate(cmd=cmd, output_dir=output_dir, source_path=path)
        return _read_mineru_output_dir(output_dir, output_files)

    @staticmethod
    @lru_cache(maxsize=4)
    def _detect_mineru_cli_caps(mineru_python: str) -> MineruCliCaps:
        full_help = _run_help_command([mineru_python, "-m", "mineru.cli.client", "--help"])
        combined = full_help.lower()

        path_flag = "--path" if "--path" in combined else "-p"

        if "--output-dir" in combined:
            output_flag = "--output-dir"
        elif "--output" in combined:
            output_flag = "--output"
        else:
            output_flag = "-o"

        return MineruCliCaps(
            path_flag=path_flag,
            output_dir_flag=output_flag,
            disable_image_flag="--disable-image" if "--disable-image" in combined else None,
            disable_table_flag="--disable-table" if "--disable-table" in combined else None,
            disable_equation_flag="--disable-equation" if "--disable-equation" in combined else None,
            mode_flag="-m" if "-m" in combined else ("--mode" if "--mode" in combined else None),
            formula_flag="-f" if "-f" in combined else ("--formula" if "--formula" in combined else None),
            table_flag="-t" if "-t" in combined else ("--table" if "--table" in combined else None),
            backend_flag="-b" if "-b" in combined else ("--backend" if "--backend" in combined else None),
            device_flag="-d" if "-d" in combined else ("--device" if "--device" in combined else None),
        )

    @staticmethod
    def _build_mineru_command(*, path: Path, text_only: bool, output_dir: Path, caps: MineruCliCaps) -> list[str]:
        path_abs = path.resolve()
        output_abs = output_dir.resolve()
        cmd = [
            settings.mineru_python,
            "-m",
            "mineru.cli.client",
            caps.path_flag,
            str(path_abs),
            caps.output_dir_flag,
            str(output_abs),
        ]

        if caps.backend_flag:
            cmd.extend([caps.backend_flag, "pipeline"])
        if caps.device_flag:
            cmd.extend([caps.device_flag, "cpu"])

        if text_only:
            if caps.mode_flag:
                cmd.extend([caps.mode_flag, "txt"])
            if caps.formula_flag:
                cmd.extend([caps.formula_flag, "false"])
            if caps.table_flag:
                cmd.extend([caps.table_flag, "false"])
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



def _doc_dir_name(path: Path) -> str:
    return path.stem

def _run_help_command(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except Exception:
        return ""
    return f"{proc.stdout or ''}\n{proc.stderr or ''}"


def _collect_output_files(output_dir: Path) -> list[Path]:
    return sorted([p for p in output_dir.rglob("*") if p.is_file()])


def _read_mineru_output_dir(output_dir: Path, files: list[Path] | None = None) -> Any:
    candidates = files if files is not None else _collect_output_files(output_dir)

    def _by_suffix(suffix: str) -> list[Path]:
        return [p for p in candidates if p.suffix.lower() == suffix]

    for candidate in _by_suffix(".json"):
        try:
            return json.loads(candidate.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

    for candidate in _by_suffix(".md") + _by_suffix(".txt"):
        text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return {"elements": [{"type": "text", "text": text}]}

    raise RuntimeError(f"mineru_output_artifact_not_found:{output_dir}")


def mineru_run_and_validate(*, cmd: list[str], output_dir: Path, source_path: Path) -> list[Path]:
    logger.info("mineru_cmd", extra={"cmd": cmd})
    logger.info("mineru_output_dir", extra={"path": str(output_dir)})

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=settings.mineru_timeout_seconds,
        check=False,
    )

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    logger.info("mineru_returncode", extra={"path": str(source_path), "returncode": proc.returncode})
    logger.info("mineru_stdout_tail", extra={"path": str(source_path), "stdout_tail": stdout[-4000:]})
    logger.info("mineru_stderr_tail", extra={"path": str(source_path), "stderr_tail": stderr[-4000:]})

    files = _collect_output_files(output_dir)
    if files:
        logger.info(
            "mineru_output_files",
            extra={
                "path": str(source_path),
                "output_dir": str(output_dir),
                "count": len(files),
                "files": [str(file.relative_to(output_dir)) for file in files[:30]],
            },
        )

    stderr_fail = _stderr_indicates_failure(stderr)
    missing_module = _extract_missing_module(stderr)
    if missing_module:
        logger.error(
            "mineru_missing_dependency_detected",
            extra={
                "missing_module": missing_module,
                "suggested_package": _module_to_package(missing_module),
                "how_to_fix": "add dependency to requirements-mineru.txt and reinstall .venv-mineru",
            },
        )

    if (proc.returncode != 0 or stderr_fail) and missing_module and settings.auto_install_mineru_deps and settings.app_env.lower() not in {"prod", "production"}:
        package = _module_to_package(missing_module)
        if package:
            install_cmd = [settings.mineru_python, "-m", "pip", "install", package]
            install = subprocess.run(install_cmd, capture_output=True, text=True, check=False)
            logger.warning(
                "mineru_dependency_auto_install_attempt",
                extra={
                    "missing_module": missing_module,
                    "package": package,
                    "returncode": install.returncode,
                    "stdout_tail": (install.stdout or "")[-1000:],
                    "stderr_tail": (install.stderr or "")[-1000:],
                },
            )
            if install.returncode == 0:
                retry = subprocess.run(cmd, capture_output=True, text=True, timeout=settings.mineru_timeout_seconds, check=False)
                rstdout = (retry.stdout or "").strip()
                rstderr = (retry.stderr or "").strip()
                logger.warning(
                    "mineru_dependency_auto_install_retry",
                    extra={
                        "missing_module": missing_module,
                        "returncode": retry.returncode,
                        "stdout_tail": rstdout[-1000:],
                        "stderr_tail": rstderr[-1000:],
                    },
                )
                if retry.returncode == 0 and not _stderr_indicates_failure(rstderr):
                    retry_files = _collect_output_files(output_dir)
                    if retry_files:
                        logger.info(
                            "mineru_output_files",
                            extra={
                                "path": str(source_path),
                                "output_dir": str(output_dir),
                                "count": len(retry_files),
                                "files": [str(file.relative_to(output_dir)) for file in retry_files[:30]],
                                "auto_install_retry": True,
                            },
                        )
                        return retry_files

    if proc.returncode != 0 or stderr_fail:
        logger.error(
            "mineru_execution_error",
            extra={
                "path": str(source_path),
                "returncode": proc.returncode,
                "cmd": cmd,
                "stderr_excerpt": stderr[-4000:],
                "missing_module": missing_module,
                "suggested_package": _module_to_package(missing_module) if missing_module else None,
            },
        )
        raise RuntimeError(
            f"mineru_returncode={proc.returncode}\ncmd={' '.join(cmd)}\nstdout={stdout}\nstderr={stderr}"
        )

    if not files:
        logger.warning(
            "mineru_output_empty",
            extra={
                "path": str(source_path),
                "output_dir": str(output_dir),
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            },
        )
        raise RuntimeError(f"mineru_output_empty:{output_dir}")

    return files


def _stderr_indicates_failure(stderr: str) -> bool:
    lowered = (stderr or "").lower()
    return "traceback" in lowered or "modulenotfounderror" in lowered or "error" in lowered


def _extract_missing_module(stderr: str) -> str | None:
    match = re.search(r'No module named ["\']([^"\']+)["\']', stderr or "")
    if match:
        return match.group(1)
    return None


def _module_to_package(module_name: str | None) -> str | None:
    if not module_name:
        return None
    return MINERU_MODULE_TO_PACKAGE.get(module_name, module_name.replace("_", "-"))


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


def _should_retry_degraded(exc: Exception) -> bool:
    text = _extract_error_text(exc).lower()
    return (
        "mineru_returncode=" in text
        or _is_mineru_transformers_mismatch(exc)
        or "cache_position" in text
        or "unimermbartforcausallm" in text
        or "missing option" in text
    )


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
