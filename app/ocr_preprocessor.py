from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.config import settings


logger = logging.getLogger("rag_service")


class OcrPreprocessError(RuntimeError):
    pass


@dataclass(frozen=True)
class PdfTextProfile:
    pages: int
    sampled_pages: int
    sampled_text_pages: int
    sampled_chars: int

    @property
    def average_chars(self) -> float:
        if self.sampled_pages == 0:
            return 0.0
        return self.sampled_chars / self.sampled_pages


def inspect_pdf_text(path: Path, *, sample_pages: int) -> PdfTextProfile:
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    indexes = _sample_indexes(page_count, sample_pages)
    sampled_chars = 0
    sampled_text_pages = 0
    for index in indexes:
        text = (reader.pages[index].extract_text() or "").strip()
        char_count = len(text)
        sampled_chars += char_count
        if char_count >= 20:
            sampled_text_pages += 1
    return PdfTextProfile(
        pages=page_count,
        sampled_pages=len(indexes),
        sampled_text_pages=sampled_text_pages,
        sampled_chars=sampled_chars,
    )


def pdf_needs_ocr(
    path: Path,
    *,
    sample_pages: int,
    min_chars_per_sample_page: int,
) -> tuple[bool, PdfTextProfile]:
    profile = inspect_pdf_text(path, sample_pages=sample_pages)
    enough_text_pages = profile.sampled_text_pages >= max(1, profile.sampled_pages // 2)
    enough_text = profile.average_chars >= min_chars_per_sample_page
    return not (enough_text_pages and enough_text), profile


def preprocess_pdf(path: Path, *, reindex: bool) -> Path:
    binary = shutil.which(settings.ocrmypdf_binary)
    if not binary:
        raise OcrPreprocessError(
            f"OCRmyPDF binary not found: {settings.ocrmypdf_binary}. "
            "Install ocrmypdf and tesseract-ocr-rus."
        )

    document_dir = Path(settings.storage_parsed).resolve() / path.stem
    temp_dir = Path(settings.ocr_temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(temp_dir).free
    required_bytes = settings.ocr_min_temp_free_mb * 1024 * 1024
    if free_bytes < required_bytes:
        raise OcrPreprocessError(
            "Insufficient OCR temporary storage: "
            f"{free_bytes // (1024 * 1024)} MiB available, "
            f"{settings.ocr_min_temp_free_mb} MiB required"
        )
    output_dir = document_dir / "ocrmypdf"
    output_path = output_dir / f"{path.stem}.ocr.pdf"
    partial_path = output_dir / f"{path.stem}.ocr.partial.pdf"

    if output_path.exists() and output_path.stat().st_size > 0 and not reindex:
        logger.info("ocrmypdf_cached_output_used", extra={"path": str(path), "output_path": str(output_path)})
        return output_path

    if reindex and document_dir.exists():
        shutil.rmtree(document_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path.unlink(missing_ok=True)

    command = [
        binary,
        "--language",
        settings.ocr_languages,
        "--jobs",
        str(settings.ocr_jobs),
        "--output-type",
        "pdf",
        "--optimize",
        "0",
        "--rotate-pages",
        "--deskew",
        "--skip-text",
        "--quiet",
        str(path.resolve()),
        str(partial_path),
    ]
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["TMPDIR"] = str(temp_dir)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.ocr_timeout_seconds,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        partial_path.unlink(missing_ok=True)
        raise OcrPreprocessError(
            f"OCRmyPDF timed out after {settings.ocr_timeout_seconds}s"
        ) from exc

    duration_s = round(time.perf_counter() - started, 3)
    stderr = (result.stderr or "").strip()
    logger.info(
        "ocrmypdf_exec_stats",
        extra={
            "path": str(path),
            "returncode": result.returncode,
            "duration_s": duration_s,
            "languages": settings.ocr_languages,
            "jobs": settings.ocr_jobs,
            "temp_dir": str(temp_dir),
            "stderr_tail": stderr[-2000:],
        },
    )
    if result.returncode != 0 or not partial_path.exists() or partial_path.stat().st_size == 0:
        partial_path.unlink(missing_ok=True)
        raise OcrPreprocessError(
            f"OCRmyPDF failed with return code {result.returncode}: {stderr[-2000:]}"
        )

    partial_path.replace(output_path)
    return output_path


def _sample_indexes(page_count: int, sample_pages: int) -> list[int]:
    if page_count <= 0 or sample_pages <= 0:
        return []
    count = min(page_count, sample_pages)
    if count == 1:
        return [0]
    return sorted({round(index * (page_count - 1) / (count - 1)) for index in range(count)})
