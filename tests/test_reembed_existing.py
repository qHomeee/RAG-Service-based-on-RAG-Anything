import argparse

import pytest

from scripts.reembed_existing import (
    _document_metadata_with_embedding,
    _validate_backup_prefix,
)


def test_backup_prefix_accepts_safe_postgres_identifier():
    assert _validate_backup_prefix("Reembed_Backup_20260724") == "reembed_backup_20260724"


@pytest.mark.parametrize(
    "value",
    [
        "has-dash",
        "contains space",
        "1starts_with_number",
        "x" * 41,
        "backup;drop table embeddings",
    ],
)
def test_backup_prefix_rejects_unsafe_identifier(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _validate_backup_prefix(value)


def test_document_metadata_update_preserves_existing_fields():
    original = {
        "subject": "history",
        "document_profile": {
            "profile_text": "Учебник истории",
            "summary_embedding": [0.0],
        },
    }

    updated = _document_metadata_with_embedding(
        original,
        summary_embedding=[0.1, 0.2],
        fingerprint="sha256:new",
    )

    assert updated["subject"] == "history"
    assert updated["embedding_fingerprint"] == "sha256:new"
    assert updated["document_profile"]["profile_text"] == "Учебник истории"
    assert updated["document_profile"]["summary_embedding"] == [0.1, 0.2]
    assert original["document_profile"]["summary_embedding"] == [0.0]
