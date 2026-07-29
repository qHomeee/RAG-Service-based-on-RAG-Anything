from pathlib import Path
from types import SimpleNamespace

from app.config import settings
from app.ocr_preprocessor import (
    PdfTextProfile,
    _sample_indexes,
    pdf_needs_ocr,
    preprocess_pdf,
)


def test_sample_indexes_cover_first_and_last_page():
    assert _sample_indexes(100, 5) == [0, 25, 50, 74, 99]


def test_pdf_needs_ocr_when_sampled_text_is_sparse(monkeypatch):
    profile = PdfTextProfile(
        pages=244,
        sampled_pages=12,
        sampled_text_pages=1,
        sampled_chars=120,
    )
    monkeypatch.setattr("app.ocr_preprocessor.inspect_pdf_text", lambda *_args, **_kwargs: profile)

    needs_ocr, returned = pdf_needs_ocr(
        Path("scan.pdf"),
        sample_pages=12,
        min_chars_per_sample_page=80,
    )

    assert needs_ocr is True
    assert returned is profile


def test_preprocess_pdf_builds_russian_cpu_command_and_reuses_cache(tmp_path, monkeypatch):
    source = tmp_path / "raw" / "book.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.4")
    parsed = tmp_path / "parsed"
    calls = []

    monkeypatch.setattr(settings, "storage_parsed", str(parsed))
    monkeypatch.setattr(settings, "ocrmypdf_binary", "ocrmypdf")
    monkeypatch.setattr(settings, "ocr_languages", "rus+eng")
    monkeypatch.setattr(settings, "ocr_jobs", 4)
    monkeypatch.setattr(settings, "ocr_timeout_seconds", 7200)
    monkeypatch.setattr("app.ocr_preprocessor.shutil.which", lambda _binary: "/usr/bin/ocrmypdf")

    def fake_run(command, *, capture_output, text, check, timeout):
        calls.append(command)
        Path(command[-1]).write_bytes(b"%PDF-1.7 OCR")
        assert timeout == 7200
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("app.ocr_preprocessor.subprocess.run", fake_run)

    output = preprocess_pdf(source, reindex=True)
    cached = preprocess_pdf(source, reindex=False)

    assert output == cached
    assert output.read_bytes() == b"%PDF-1.7 OCR"
    assert len(calls) == 1
    assert "--language" in calls[0]
    assert "rus+eng" in calls[0]
    assert "--jobs" in calls[0]
    assert "4" in calls[0]
