# module5/document_upload.py
"""
Document Upload — Let users upload their own PDF/TXT/DOCX/CSV/MD and query it.
Creates a temporary in-memory vector store from the parsed content.

Security guards:
  • file size capped by configs.config.MAX_UPLOAD_BYTES
  • magic-byte sniffing + extension allow-list (defence in depth)
  • path-traversal-safe filename storage
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import PurePosixPath
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from configs.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_UPLOAD_BYTES

logger = logging.getLogger(__name__)


# Magic-byte signatures for the file types we accept.
_MAGIC_BYTES = {
    "pdf":  (b"%PDF",),
    "docx": (b"PK\x03\x04",),  # DOCX is a ZIP container
}


def _safe_basename(name: str) -> str:
    """
    Reduce an uploaded filename to a safe basename.
    Strips any directory components and drops control characters.
    """
    base = PurePosixPath(name.replace("\\", "/")).name
    return "".join(c for c in base if 32 <= ord(c) < 127) or "uploaded"


class UploadValidationError(ValueError):
    """Raised when an uploaded file fails security validation."""


class DocumentUploader:
    """
    Handles document upload, parsing, chunking, and indexing into
    a temporary vector store for querying.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md", ".csv", ".docx"}

    def __init__(self, embedding_model=None, max_bytes: int = MAX_UPLOAD_BYTES):
        self.embedding_model = embedding_model
        self.max_bytes = max_bytes
        self._vectorstore: Optional[Chroma] = None
        self._chunks: List[Document] = []
        self._filename: str = ""

    @property
    def is_ready(self) -> bool:
        return self._vectorstore is not None and len(self._chunks) > 0

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def vectorstore(self) -> Optional[Chroma]:
        """The in-memory vector store for the active document (None until indexed)."""
        return self._vectorstore

    # ── Validation ──────────────────────────────────────────────────────────
    def _validate(self, file_content: bytes, filename: str) -> str:
        if not file_content:
            raise UploadValidationError("Empty upload (0 bytes).")
        if len(file_content) > self.max_bytes:
            raise UploadValidationError(
                f"File too large: {len(file_content):,} bytes "
                f"(max allowed: {self.max_bytes:,} bytes)."
            )
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UploadValidationError(
                f"Unsupported file type '{ext}'. Supported formats: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        # Magic-byte sniff for binary types within first 1KB (tolerates BOM/leading whitespace)
        header_sample = file_content[:1024]
        if ext == ".pdf" and b"%PDF" not in header_sample:
            raise UploadValidationError("File is not a valid PDF (missing %PDF signature header).")
        if ext == ".docx" and b"PK\x03\x04" not in header_sample:
            raise UploadValidationError("File is not a valid DOCX (missing ZIP/DOCX header).")
        return ext

    # ── Parsers ─────────────────────────────────────────────────────────────
    def parse_file(self, file_content: bytes, filename: str) -> List[Document]:
        ext = self._validate(file_content, filename)
        self._filename = _safe_basename(filename)

        if ext == ".pdf":
            return self._parse_pdf(file_content)
        if ext == ".docx":
            return self._parse_docx(file_content)
        return self._parse_text(file_content)

    def _parse_text(self, content: bytes) -> List[Document]:
        text = ""
        for encoding in ("utf-8", "latin-1", "cp1252", "utf-16"):
            try:
                text = content.decode(encoding)
                break
            except Exception:
                continue
        if not text:
            text = content.decode("utf-8", errors="replace")

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        docs = [
            Document(
                page_content=para,
                metadata={"source": self._filename, "paragraph": i},
            )
            for i, para in enumerate(paragraphs)
            if len(para) > 10
        ]
        if not docs and text.strip():
            docs.append(Document(
                page_content=text.strip()[:10000],
                metadata={"source": self._filename, "paragraph": 0},
            ))
        if not docs:
            raise UploadValidationError("File contains no readable text.")
        return docs

    def _parse_pdf(self, content: bytes) -> List[Document]:
        docs: List[Document] = []
        parse_errors: List[str] = []

        # 1. Primary engine: PyMuPDF (fast, handles complex layouts, fonts, encodings)
        try:
            import pymupdf
            doc = pymupdf.open(stream=content, filetype="pdf")
            if doc.is_encrypted:
                try:
                    doc.authenticate("")
                except Exception:
                    pass

            for i, page in enumerate(doc):
                try:
                    text = page.get_text("text").strip()
                    if text:
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": self._filename, "page": i + 1},
                        ))
                except Exception as page_exc:
                    logger.warning("PyMuPDF extraction failed on page %d: %s", i + 1, page_exc)
            doc.close()
        except Exception as exc:
            parse_errors.append(f"PyMuPDF error: {exc}")
            logger.warning("PyMuPDF parser failed, trying pypdf fallback: %s", exc)

        # 2. Secondary engine fallback: pypdf
        if not docs:
            try:
                import io
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(content))
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception:
                        pass

                for i, page in enumerate(reader.pages):
                    try:
                        text = (page.extract_text() or "").strip()
                        if text:
                            docs.append(Document(
                                page_content=text,
                                metadata={"source": self._filename, "page": i + 1},
                            ))
                    except Exception as page_exc:
                        logger.warning("pypdf page %d extraction failed: %s", i + 1, page_exc)
            except Exception as exc:
                parse_errors.append(f"pypdf error: {exc}")
                logger.warning("pypdf fallback failed: %s", exc)

        if not docs:
            err_msg = (
                "PDF contained no extractable text layer. "
                "This typically happens if the PDF consists of scanned images/photos without OCR. "
                "Please upload a text-searchable PDF or a document in .txt, .md, or .docx format."
            )
            if parse_errors:
                err_msg += f" (Diagnostics: {'; '.join(parse_errors)})"
            raise UploadValidationError(err_msg)

        return docs

    def _parse_docx(self, content: bytes) -> List[Document]:
        import io
        from docx import Document as DocxDocument
        try:
            doc = DocxDocument(io.BytesIO(content))
        except Exception as exc:
            raise UploadValidationError(f"Could not parse DOCX: {exc}") from exc

        parts = []
        for para in doc.paragraphs:
            t = para.text.strip()
            if t:
                parts.append(t)
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    parts.append(row_text)

        docs = [
            Document(
                page_content=p,
                metadata={"source": self._filename, "paragraph": i},
            )
            for i, p in enumerate(parts)
            if len(p) > 10
        ]
        if not docs:
            raise UploadValidationError("DOCX document contained no text content.")
        return docs

    # ── Indexing ────────────────────────────────────────────────────────────
    def index_documents(
        self,
        file_content: bytes,
        filename: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> int:
        docs = self.parse_file(file_content, filename)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._chunks = splitter.split_documents(docs)

        if self.embedding_model is None:
            from module1.data_preprocessing import get_embedding_model
            self.embedding_model = get_embedding_model()

        # Replace any prior in-memory store. We do not pass persist_directory
        # so the collection lives only in memory and is GC'd when this object dies.
        self._vectorstore = Chroma.from_documents(
            documents=self._chunks,
            embedding=self.embedding_model,
        )
        logger.info("Indexed %d chunks from %s", len(self._chunks), self._filename)
        return len(self._chunks)

    # ── Retrieval ───────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5) -> List[Document]:
        if not self.is_ready:
            return []
        return self._vectorstore.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        if not self.is_ready:
            return []
        return self._vectorstore.similarity_search_with_score(query, k=k)

    def clear(self) -> None:
        self._vectorstore = None
        self._chunks = []
        self._filename = ""
