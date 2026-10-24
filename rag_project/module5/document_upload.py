"""Validated uploads, isolated parsing, and transactional ephemeral indexing."""

from __future__ import annotations

import logging
import math
import os
from copy import copy
from typing import List, Optional
from uuid import uuid4

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from configs.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_UPLOAD_BYTES
from .parser_isolation import (
    DEFAULT_PARSER_CPU_SECONDS,
    DEFAULT_PARSER_MEMORY_BYTES,
    DEFAULT_PARSER_OUTPUT_BYTES,
    DEFAULT_PARSER_WALL_SECONDS,
    IsolationLimits,
    parse_isolated,
)
from .parser_worker import (
    DEFAULT_MAX_DOCX_COMPRESSION_RATIO as DEFAULT_MAX_DOCX_COMPRESSION_RATIO,
    DEFAULT_MAX_DOCX_ENTRIES as DEFAULT_MAX_DOCX_ENTRIES,
    DEFAULT_MAX_DOCX_MEMBER_BYTES as DEFAULT_MAX_DOCX_MEMBER_BYTES,
    DEFAULT_MAX_DOCX_UNCOMPRESSED_BYTES as DEFAULT_MAX_DOCX_UNCOMPRESSED_BYTES,
    DEFAULT_MAX_EXTRACTED_TEXT_BYTES_PER_DOCUMENT as DEFAULT_MAX_EXTRACTED_TEXT_BYTES_PER_DOCUMENT,
    DEFAULT_MAX_EXTRACTED_TEXT_BYTES_PER_UPLOAD as DEFAULT_MAX_EXTRACTED_TEXT_BYTES_PER_UPLOAD,
    DEFAULT_MAX_EXTRACTED_TEXT_CHARS_PER_DOCUMENT as DEFAULT_MAX_EXTRACTED_TEXT_CHARS_PER_DOCUMENT,
    DEFAULT_MAX_EXTRACTED_TEXT_CHARS_PER_UPLOAD as DEFAULT_MAX_EXTRACTED_TEXT_CHARS_PER_UPLOAD,
    DEFAULT_MAX_PARSED_DOCUMENTS as DEFAULT_MAX_PARSED_DOCUMENTS,
    DEFAULT_MAX_PDF_PAGES as DEFAULT_MAX_PDF_PAGES,
    POLICY_DEFAULTS,
    BoundedParser,
    UploadValidationError as UploadValidationError,
    _safe_basename as _safe_basename,
)

logger = logging.getLogger(__name__)


class DocumentUploader(BoundedParser):
    """Upload parser/index owner. All public parsing runs in bounded workers.

    The inherited private helpers remain directly testable; parse_file never
    calls them in the application process, even for small text uploads.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md", ".csv", ".docx"}

    def __init__(
        self,
        embedding_model=None,
        max_bytes: int = MAX_UPLOAD_BYTES,
        *,
        max_extracted_text_bytes_per_upload: int = DEFAULT_MAX_EXTRACTED_TEXT_BYTES_PER_UPLOAD,
        max_extracted_text_chars_per_upload: int = DEFAULT_MAX_EXTRACTED_TEXT_CHARS_PER_UPLOAD,
        max_extracted_text_bytes_per_document: int = DEFAULT_MAX_EXTRACTED_TEXT_BYTES_PER_DOCUMENT,
        max_extracted_text_chars_per_document: int = DEFAULT_MAX_EXTRACTED_TEXT_CHARS_PER_DOCUMENT,
        max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
        max_parsed_documents: int = DEFAULT_MAX_PARSED_DOCUMENTS,
        max_docx_entries: int = DEFAULT_MAX_DOCX_ENTRIES,
        max_docx_uncompressed_bytes: int = DEFAULT_MAX_DOCX_UNCOMPRESSED_BYTES,
        max_docx_member_bytes: int = DEFAULT_MAX_DOCX_MEMBER_BYTES,
        max_docx_compression_ratio: float = DEFAULT_MAX_DOCX_COMPRESSION_RATIO,
        parser_wall_seconds: float = DEFAULT_PARSER_WALL_SECONDS,
        parser_cpu_seconds: int = DEFAULT_PARSER_CPU_SECONDS,
        parser_memory_bytes: int = DEFAULT_PARSER_MEMORY_BYTES,
        parser_output_bytes: int = DEFAULT_PARSER_OUTPUT_BYTES,
    ):
        self.embedding_model = embedding_model
        values = locals()
        for name in POLICY_DEFAULTS:
            value = self._positive_limit(name, values[name])
            if name != "max_docx_compression_ratio" and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"{name} must be an integer")
            setattr(self, name, value)
        self._parser_limits = IsolationLimits(
            wall_seconds=parser_wall_seconds, cpu_seconds=parser_cpu_seconds,
            memory_bytes=parser_memory_bytes, output_bytes=parser_output_bytes,
        )
        self._last_parser_receipt = None
        self._vectorstore: Optional[Chroma] = None
        self._owned_vectorstore: Optional[Chroma] = None
        self._chunks: List[Document] = []
        self._filename = ""

    @staticmethod
    def _positive_limit(name, value):
        try:
            valid = not isinstance(value, bool) and math.isfinite(value) and value > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"{name} must be positive")
        return value

    @property
    def is_ready(self) -> bool:
        return self._vectorstore is not None and bool(self._chunks)

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def vectorstore(self) -> Optional[Chroma]:
        return self._vectorstore

    @property
    def chunks(self) -> List[Document]:
        return list(self._chunks)

    def _validate(self, file_content: bytes, filename: str) -> str:
        if not file_content:
            raise UploadValidationError("Empty upload (0 bytes).")
        if len(file_content) > self.max_bytes:
            raise UploadValidationError("File too large for the encoded upload limit.")
        ext = os.path.splitext(_safe_basename(filename))[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UploadValidationError("Unsupported file type. Supported formats: PDF, DOCX, TXT, CSV, MD.")
        header = file_content[:1024]
        if ext == ".pdf" and b"%PDF" not in header:
            raise UploadValidationError("File is not a valid PDF (missing signature header).")
        if ext == ".docx" and b"PK\x03\x04" not in header:
            raise UploadValidationError("File is not a valid DOCX (missing signature header).")
        return ext

    def parse_file(self, file_content: bytes, filename: str) -> List[Document]:
        extension = self._validate(file_content, filename)
        safe_name = _safe_basename(filename)
        policy = {name: getattr(self, name) for name in POLICY_DEFAULTS}
        records, receipt = parse_isolated(file_content, safe_name, extension, policy, self._parser_limits)
        documents = [Document(page_content=record["page_content"], metadata=record["metadata"]) for record in records]
        # Publication follows parsing, protocol validation, and conversion.
        self._filename = safe_name
        self._last_parser_receipt = receipt
        return documents

    def index_documents(
        self,
        file_content: bytes,
        filename: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> int:
        staged_parser = copy(self)
        docs = staged_parser.parse_file(file_content, filename)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        embedding_model = self.embedding_model
        if embedding_model is None:
            from module1.data_preprocessing import get_embedding_model
            embedding_model = get_embedding_model()
        vectorstore = Chroma(
            collection_name=f"upload-{uuid4().hex}", embedding_function=embedding_model,
        )
        try:
            vectorstore.add_documents(chunks)
        except Exception:
            self._delete_owned_collection(vectorstore)
            raise
        previous_owned_store = self._owned_vectorstore
        self._vectorstore = vectorstore
        self._owned_vectorstore = vectorstore
        self._chunks = chunks
        self._filename = staged_parser.filename
        self._last_parser_receipt = staged_parser._last_parser_receipt
        self.embedding_model = embedding_model
        self._delete_owned_collection(previous_owned_store)
        logger.info("Indexed %d chunks from %s", len(self._chunks), self._filename)
        return len(self._chunks)

    @staticmethod
    def _delete_owned_collection(vectorstore: Optional[Chroma]) -> None:
        if vectorstore is not None:
            try:
                vectorstore.delete_collection()
            except Exception:
                logger.warning("Could not delete an owned upload collection", exc_info=True)

    def search(self, query: str, k: int = 5) -> List[Document]:
        return self._vectorstore.similarity_search(query, k=k) if self.is_ready else []

    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        return self._vectorstore.similarity_search_with_score(query, k=k) if self.is_ready else []

    def clear(self) -> None:
        owned_store = self._owned_vectorstore
        self._vectorstore = None
        self._owned_vectorstore = None
        self._chunks = []
        self._filename = ""
        self._last_parser_receipt = None
        self._delete_owned_collection(owned_store)
