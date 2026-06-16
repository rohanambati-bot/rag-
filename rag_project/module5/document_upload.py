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

    # ── Validation ──────────────────────────────────────────────────────────
    def _validate(self, file_content: bytes, filename: str) -> str:
        if not file_content:
            raise UploadValidationError("Empty upload.")
        if len(file_content) > self.max_bytes:
            raise UploadValidationError(
                f"File too large: {len(file_content):,} bytes "
                f"(max {self.max_bytes:,} bytes)."
            )
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UploadValidationError(
                f"Unsupported file type: {ext}. Supported: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        # Magic-byte sniff for binary types — text formats are checked by decode.
        if ext == ".pdf" and not file_content.startswith(_MAGIC_BYTES["pdf"]):
            raise UploadValidationError("File is not a valid PDF (bad header).")
        if ext == ".docx" and not file_content.startswith(_MAGIC_BYTES["docx"]):
            raise UploadValidationError("File is not a valid DOCX (bad header).")
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
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="replace")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        docs = [
            Document(
                page_content=para,
                metadata={"source": self._filename, "paragraph": i},
            )
            for i, para in enumerate(paragraphs)
            if len(para) > 20
        ]
        if not docs:
            docs.append(Document(
                page_content=text[:5000],
                metadata={"source": self._filename, "paragraph": 0},
            ))
        return docs

    def _parse_pdf(self, content: bytes) -> List[Document]:
        import io

        from pypdf import PdfReader  # maintained successor of PyPDF2
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as exc:
            raise UploadValidationError(f"Could not parse PDF: {exc}") from exc

        docs: List[Document] = []
        for i, page in enumerate(reader.pages):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as exc:  # noqa: BLE001 — keep going on bad pages
                logger.warning("PDF page %d extraction failed: %s", i + 1, exc)
                continue
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": self._filename, "page": i + 1},
                ))
        if not docs:
            raise UploadValidationError("PDF contained no extractable text.")
        return docs

    def _parse_docx(self, content: bytes) -> List[Document]:
        import io

        from docx import Document as DocxDocument
        try:
            doc = DocxDocument(io.BytesIO(content))
        except Exception as exc:
            raise UploadValidationError(f"Could not parse DOCX: {exc}") from exc
        docs = [
            Document(
                page_content=para.text.strip(),
                metadata={"source": self._filename, "paragraph": i},
            )
            for i, para in enumerate(doc.paragraphs)
            if len(para.text.strip()) > 20
        ]
        if not docs:
            raise UploadValidationError("DOCX contained no extractable text.")
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
