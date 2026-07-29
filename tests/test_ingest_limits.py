from app.config import Settings


def test_default_ingest_profile_accepts_two_gigabyte_textbooks():
    config = Settings(_env_file=None)

    assert config.max_file_size_mb == 2048
    assert config.max_ingest_batch_mb == 4096
    assert config.mineru_timeout_seconds == 21_600
    assert config.ocr_timeout_seconds == 21_600
