"""Security tests for module5.document_upload."""
from __future__ import annotations

import pytest

from module5.document_upload import (
    DocumentUploader, UploadValidationError, _safe_basename,
)


class TestSafeBasename:
    def test_strips_directory_components(self):
        assert _safe_basename("../../etc/passwd") == "passwd"
        assert _safe_basename("C:\\Users\\bob\\report.pdf") == "report.pdf"

    def test_drops_control_characters(self):
        assert _safe_basename("file\x00name.txt") == "filename.txt"

    def test_empty_input_yields_default(self):
        assert _safe_basename("") == "uploaded"


class TestUploadValidation:
    def test_rejects_oversized_upload(self):
        uploader = DocumentUploader(max_bytes=100)
        with pytest.raises(UploadValidationError, match="too large"):
            uploader.parse_file(b"x" * 200, "file.txt")

    def test_rejects_empty_upload(self):
        uploader = DocumentUploader()
        with pytest.raises(UploadValidationError, match="Empty"):
            uploader.parse_file(b"", "file.txt")

    def test_rejects_unsupported_extension(self):
        uploader = DocumentUploader()
        with pytest.raises(UploadValidationError, match="Unsupported"):
            uploader.parse_file(b"contents", "malicious.exe")

    def test_rejects_pdf_with_bad_magic_bytes(self):
        uploader = DocumentUploader()
        # Looks like a PDF by extension but the bytes are not a PDF.
        with pytest.raises(UploadValidationError, match="not a valid PDF"):
            uploader.parse_file(b"this is not a pdf", "fake.pdf")

    def test_rejects_docx_with_bad_magic_bytes(self):
        uploader = DocumentUploader()
        with pytest.raises(UploadValidationError, match="not a valid DOCX"):
            uploader.parse_file(b"this is not a zip", "fake.docx")

    def test_accepts_valid_text_upload(self):
        uploader = DocumentUploader()
        docs = uploader.parse_file(
            b"This is a paragraph of more than twenty characters in length.\n\nAnother paragraph here too, also long enough.",
            "notes.txt",
        )
        assert len(docs) >= 1
        assert all(d.metadata["source"] == "notes.txt" for d in docs)
