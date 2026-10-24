# module1/data_preprocessing.py
"""
Module 1 — Data Preprocessing & Synthetic Data Generation
─────────────────────────────────────────────────────────
Steps:
  1. Load MS MARCO v2.1 (HuggingFace datasets) — passage ranking QA
  2. Chunk documents with RecursiveCharacterTextSplitter
  3. Generate dense embeddings (all-MiniLM-L6-v2)
  4. Index into Chroma vector store
  5. Generate synthetic Q&A queries via LLM (simple / paraphrased / ambiguous)
  6. Persist everything for downstream modules

Dataset: MS MARCO v2.1
  - Real Bing search queries
  - ~10 passages per query with is_selected labels
  - Human-written answers
  - 800K+ samples (we use a configurable subset)
"""

import os
import json
import logging
import time
import shutil
import hashlib
import tempfile
import uuid
import atexit
import threading
from contextlib import contextmanager
from typing import Iterable, List, Dict, Any, Optional
from pathlib import Path

import re
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT, DATASET_REVISION,
    DATASET_SUBSET_SIZE, INDEX_REVISION,
    SQUAD_SPLIT, SQUAD_SUBSET_SIZE,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_DIM, VECTOR_STORE_DIR,
    SYNTHETIC_DATA_PATH, SYNTHETIC_BATCH_SIZE,
    NUM_SYNTHETIC_SAMPLES, MISTRAL_API_KEY, LLM_MODEL,
    SYNTHETIC_QUERY_TYPES, DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Build/persistence lifecycle primitives ─────────────────────────────────────

STAGING_METADATA_FILE = ".build-staging.json"
DEFAULT_BUILD_LOCK_TIMEOUT_SECONDS = float(
    os.getenv("INDEX_BUILD_LOCK_TIMEOUT_SECONDS", "30")
)
DEFAULT_BUILD_LOCK_POLL_SECONDS = float(
    os.getenv("INDEX_BUILD_LOCK_POLL_SECONDS", "0.05")
)


class BuildLockError(RuntimeError):
    """Raised when another process owns an index-build lock."""


_READER_LEASES = {}
_LEASE_LOCK = threading.RLock()
_LEASE_PID = os.getpid()
_NATIVE_PATHS = set()
_FAILED_NATIVE_PATHS = set()
_PUBLICATION_GUARDS = {}


def _check_index_process() -> None:
    """Native Chroma handles/locks cannot safely be inherited across fork."""
    if os.getpid() != _LEASE_PID:
        raise BuildLockError("Inherited index state after fork; use a fresh process (spawn)")


def _lease_file(target_path: str):
    # A sibling inode survives directory renames. Never unlink this lock file:
    # replacing it would let later readers lock an unrelated inode.
    path = f"{os.path.realpath(target_path)}.readers.lock"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "a+b")


def _acquire_reader_lease(target_path: str) -> bool:
    """Pin an embedded Chroma pathname until process exit.

    Chroma retains native clients beyond a Python wrapper's lifetime, so dropping
    a wrapper/shutdown reference is not sufficient to safely release this lease.
    Stop all participating reader processes before publishing a replacement.
    """
    import fcntl
    _check_index_process()
    path = os.path.realpath(target_path)
    with _LEASE_LOCK:
        if path in _FAILED_NATIVE_PATHS:
            raise BuildLockError("Failed index writer still owns this path; use a fresh process")
        if path in _READER_LEASES:
            return False
        if path in _PUBLICATION_GUARDS:
            raise BuildLockError("Reader refused inside an index publication guard")
        lease = _lease_file(path)
        try:
            fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            lease.close()
            raise BuildLockError("Index publication in progress; reader refused") from exc
        _READER_LEASES[path] = lease
        return True


@atexit.register
def _close_reader_leases() -> None:
    for lease in _READER_LEASES.values():
        lease.close()
    _READER_LEASES.clear()


@contextmanager
def index_publication_guard(target_path: str, *, pin_reader: bool = False):
    """Fail closed instead of renaming files beneath active embedded readers.

    This is a participating-process POSIX/local-filesystem contract, not hot swap
    support for direct Chroma clients or network filesystems.
    """
    import fcntl
    _check_index_process()
    path = os.path.realpath(target_path)
    with _LEASE_LOCK:
        if path in _READER_LEASES or path in _NATIVE_PATHS:
            raise BuildLockError("Stop this Chroma reader; rebuild from a fresh process")
        if path in _PUBLICATION_GUARDS:
            raise BuildLockError("Nested index publication guard refused")
        _PUBLICATION_GUARDS[path] = threading.get_ident()
    lease = None
    try:
        lease = _lease_file(target_path)
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BuildLockError(
                "Stop existing Chroma readers before publishing; restart after publication"
            ) from exc
        try:
            yield
        except BaseException:
            # Even a failed native constructor can cache the path. Keep EX until
            # exit: no reader may treat a partially written index as complete.
            if path in _NATIVE_PATHS:
                with _LEASE_LOCK:
                    _FAILED_NATIVE_PATHS.add(path)
                    _READER_LEASES[path] = lease
                    lease = None
            raise
        else:
            if pin_reader and path in _NATIVE_PATHS:
                with _LEASE_LOCK:
                    fcntl.flock(lease, fcntl.LOCK_SH)
                    _READER_LEASES[path] = lease
                    lease = None
    finally:
        with _LEASE_LOCK:
            _PUBLICATION_GUARDS.pop(path, None)
        if lease is not None:
            lease.close()


def copy_vector_store(source: str, destination: str) -> str:
    """Copy a quiescent index, including SQLite WAL, under the same barrier.

    A filesystem copy is not a transactional SQLite backup. Stop participating
    readers/writers before snapshotting; never silently copy a changing corpus.
    Destination must be new. No native source client is opened or cached.
    """
    source, destination = os.path.realpath(source), os.path.realpath(destination)
    with build_lock_for(source), index_publication_guard(source):
        if not os.path.isfile(os.path.join(source, "chroma.sqlite3")):
            raise FileNotFoundError(f"Vector store directory not found: {source}")
        with build_lock_for(destination), index_publication_guard(destination):
            shutil.copytree(source, destination)
    return destination


def run_isolated_index_builder(builder, *args):
    """Run a top-level build function in spawn; join before any publication.

    Chroma retains path-keyed native clients. A staging writer must exit before
    its directory can be renamed. Never try to clear Chroma's private cache.
    """
    import multiprocessing
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_isolated_index_worker, args=(child, builder, args))
    try:
        process.start()
        child.close()
        while not parent.poll(0.1):
            if not process.is_alive() and not parent.poll():
                raise RuntimeError(f"Index builder exited without receipt ({process.exitcode})")
        receipt = parent.recv()
        process.join()
        if process.exitcode != 0 or not receipt[0]:
            raise RuntimeError(f"Isolated index build failed: {receipt[1]}")
        return receipt[1]
    finally:
        child.close()
        parent.close()
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join()


def _isolated_index_worker(connection, builder, args):
    try:
        connection.send((True, builder(*args)))
    except BaseException as exc:
        connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def atomic_write_json(path: str, data: Any) -> None:
    """Write JSON through a same-directory temporary file and atomic replace.

    Checkpoints and index manifests are small, but a process interruption while
    writing either file must not leave a truncated document that a later worker
    mistakes for a valid build state.  ``os.replace`` is atomic when both paths
    are on the same filesystem, so the temporary file deliberately lives beside
    the destination.
    """
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(destination)}.",
            suffix=".tmp",
            dir=parent,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(data, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class BuildLock:
    """Bounded POSIX builder lock with a stable inode and ownership receipt.

    The never-unlinked .builders.lock inode releases automatically on process
    death. The historical .build.lock JSON is an ownership receipt, not the lock
    primitive. A dead owner's receipt is archived, never silently deleted.
    """

    def __init__(
        self,
        target_path: str,
        timeout: Optional[float] = None,
        poll_interval: float = DEFAULT_BUILD_LOCK_POLL_SECONDS,
    ):
        self.target_path = os.path.realpath(target_path)
        self.lock_path = f"{self.target_path}.build.lock"
        self.timeout = (
            DEFAULT_BUILD_LOCK_TIMEOUT_SECONDS if timeout is None else max(0.0, timeout)
        )
        self.poll_interval = max(0.001, poll_interval)
        self._token: Optional[str] = None
        self._owned = False
        self._file = None

    def acquire(self) -> "BuildLock":
        import fcntl
        _check_index_process()
        if self._owned:
            return self
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        deadline = time.monotonic() + self.timeout
        lease = open(f"{self.target_path}.builders.lock", "a+b")
        while True:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                if self.timeout <= 0 or time.monotonic() >= deadline:
                    lease.close()
                    raise BuildLockError(
                        f"Index build lock is held for {self.target_path}: "
                        f"{self.lock_path} (waited {self.timeout:.2f}s)"
                    )
                time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))
                continue

            token = uuid.uuid4().hex
            metadata = {
                "target": self.target_path,
                "pid": os.getpid(),
                "acquired_at": time.time(),
                "token": token,
            }
            try:
                if os.path.exists(self.lock_path):
                    with open(self.lock_path, encoding="utf-8") as previous:
                        abandoned = json.load(previous)
                    if (not isinstance(abandoned, dict)
                            or not isinstance(abandoned.get("pid"), int)
                            or abandoned["pid"] <= 0
                            or _process_is_alive(abandoned["pid"])):
                        raise BuildLockError("Existing legacy builder receipt has a live/unknown owner")
                    os.replace(self.lock_path, f"{self.lock_path}.abandoned.{uuid.uuid4().hex}")
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                with os.fdopen(fd, "w", encoding="utf-8") as output:
                    json.dump(metadata, output)
                    output.flush()
                    os.fsync(output.fileno())
            except Exception:
                lease.close()
                raise
            self._token = token
            self._owned = True
            self._file = lease
            return self

    def release(self) -> None:
        if not self._owned:
            return
        try:
            self._release_receipt()
        finally:
            self._file.close()
            self._file = None

    def _release_receipt(self) -> None:
        self._owned = False
        token = self._token
        self._token = None
        try:
            with open(self.lock_path, encoding="utf-8") as lock_file:
                metadata = json.load(lock_file)
            if token is not None and metadata.get("token") != token:
                logger.warning("Build lock %s changed owner before release.", self.lock_path)
                return
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not verify build lock ownership: %s", self.lock_path)
            return
        try:
            os.unlink(self.lock_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not release build lock %s: %s", self.lock_path, exc)

    def __enter__(self) -> "BuildLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def build_lock_for(
    target_path: str,
    timeout: Optional[float] = None,
    poll_interval: float = DEFAULT_BUILD_LOCK_POLL_SECONDS,
) -> BuildLock:
    """Return the lock guarding a final index path."""
    return BuildLock(target_path, timeout=timeout, poll_interval=poll_interval)


def create_staging_dir(
    parent_dir: str,
    prefix: str,
    target_path: str,
    lock: Optional[BuildLock] = None,
) -> str:
    """Create a marked staging directory that cleanup can identify safely."""
    os.makedirs(parent_dir, exist_ok=True)
    staged_dir = tempfile.mkdtemp(prefix=prefix, dir=parent_dir)
    try:
        atomic_write_json(
            os.path.join(staged_dir, STAGING_METADATA_FILE),
            {
                "state": "active",
                "target": os.path.abspath(target_path),
                "pid": os.getpid(),
                "created_at": time.time(),
                "lock_token": getattr(lock, "_token", None),
            },
        )
    except Exception:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise
    return staged_dir


def _process_is_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    return True


def cleanup_stale_staging_dirs(
    data_dir: str,
    *,
    prefix: str,
    target_path: str,
    protected_paths: Optional[List[str]] = None,
    active_lock_token: Optional[str] = None,
) -> List[str]:
    """Archive only marked, non-active staging directories for one target.

    Callers must hold ``build_lock_for(target_path)``.  The marker and target
    match are required, and a live owner PID is preserved as a final guard
    against deleting an active build.  Unmarked directories are deliberately
    left alone because their ownership is unknown.
    """
    expected_target = os.path.abspath(target_path)
    protected = {os.path.abspath(path) for path in (protected_paths or [])}
    removed: List[str] = []
    for candidate in Path(data_dir).glob(f"{prefix}*"):
        if not candidate.is_dir() or str(candidate.resolve()) in protected:
            continue
        metadata_path = candidate / STAGING_METADATA_FILE
        try:
            with metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        if metadata.get("state") != "active":
            continue
        if os.path.abspath(str(metadata.get("target", ""))) != expected_target:
            continue
        owner_is_alive = _process_is_alive(metadata.get("pid"))
        marker_token = metadata.get("lock_token")
        if owner_is_alive and (
            marker_token is None
            or not active_lock_token
            or marker_token == active_lock_token
        ):
            continue
        try:
            # Preserve failed/interrupted build artifacts for inspection. Moving
            # them out of the active-stage prefix prevents repeated cleanup.
            with index_publication_guard(str(candidate)):
                candidate.rename(Path(data_dir) / f"preserved-index-staging.{uuid.uuid4().hex}")
            removed.append(str(candidate))
        except (OSError, BuildLockError) as exc:
            logger.warning("Could not archive stale staging directory %s: %s", candidate, exc)
    return removed


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_msmarco_documents(
    split: str = DATASET_SPLIT,
    max_samples: Optional[int] = DATASET_SUBSET_SIZE,
) -> List[Document]:
    """
    Load MS MARCO v2.1 and convert to LangChain Documents.

    MS MARCO structure per row:
      - query (str): the Bing search question
      - passages (dict):
          - passage_text (list[str]): ~10 passages
          - is_selected (list[int]): 1 if passage contains the answer
          - url (list[str]): source URLs
      - answers (list[str]): human-written answers
      - query_type (str): numeric, entity, description, location, person

    We extract:
      - Each selected passage → Document (page_content)
      - Metadata: query, answer, query_type, url
    """
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")
    if max_samples == 0:
        return []

    logger.info("Loading MS MARCO v2.1 (%s split)…", split)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)

    docs: List[Document] = []
    seen_passages: set = set()
    count = 0

    for row in dataset:
        # Skip rows with no answers
        answers = row.get("answers", [])
        if not answers or answers == ["No Answer Present."]:
            continue

        query = row.get("query", "").strip()
        query_type = row.get("query_type", "unknown")
        passages_data = row.get("passages", {})
        passage_texts = passages_data.get("passage_text", [])
        is_selected = passages_data.get("is_selected", [])
        urls = passages_data.get("url", [])

        # Get the best answer
        answer = answers[0] if answers else ""

        # Extract ALL passages for this query (both selected and unselected)
        # This gives us a realistic retrieval corpus
        for idx, passage_text in enumerate(passage_texts):
            passage_text = passage_text.strip()
            if not passage_text or len(passage_text) < 30:
                continue

            # Deduplicate (first 200 chars matches the persisted 11,510-chunk dev store)
            passage_key = passage_text[:200]
            if passage_key in seen_passages:
                continue
            seen_passages.add(passage_key)

            selected = is_selected[idx] if idx < len(is_selected) else 0
            url = urls[idx] if idx < len(urls) else ""

            doc = Document(
                page_content=passage_text,
                metadata={
                    "query": query,
                    "answers": [answer],
                    "query_type": query_type,
                    "is_selected": selected,
                    "url": url,
                    "source": "ms_marco_v2.1",
                },
            )
            docs.append(doc)

        count += 1
        if max_samples is not None and count >= max_samples:
            break

    logger.info(
        "Loaded %d passages from %d MS MARCO queries (selected + unselected).",
        len(docs), count,
    )
    return docs


def load_msmarco_qa_pairs(
    split: str = DATASET_SPLIT,
    max_samples: Optional[int] = DATASET_SUBSET_SIZE,
) -> List[Dict[str, Any]]:
    """
    Load MS MARCO as Q&A pairs for evaluation.
    Returns list of dicts with: question, ground_truth, query_type, context.
    """
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")
    if max_samples == 0:
        return []

    logger.info("Loading MS MARCO Q&A pairs (%s split, max %s)…", split, max_samples)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)

    qa_pairs: List[Dict[str, Any]] = []
    count = 0

    for row in dataset:
        answers = row.get("answers", [])
        if not answers or answers == ["No Answer Present."]:
            continue

        query = row.get("query", "").strip()
        if not query:
            continue

        passages_data = row.get("passages", {})
        passage_texts = passages_data.get("passage_text", [])
        is_selected = passages_data.get("is_selected", [])

        # Get selected passages as ground-truth context
        selected_passages = [
            passage_texts[i]
            for i in range(len(passage_texts))
            if i < len(is_selected) and is_selected[i] == 1
        ]
        context = " ".join(selected_passages) if selected_passages else ""

        qa_pairs.append({
            "id": f"msmarco_{count}",
            "question": query,
            "ground_truth": answers[0],
            "query_type": row.get("query_type", "unknown"),
            "context": context,
            "all_passages": passage_texts,
        })

        count += 1
        if max_samples is not None and count >= max_samples:
            break

    logger.info("Loaded %d Q&A pairs from MS MARCO.", len(qa_pairs))
    return qa_pairs


# Legacy SQuAD loader (kept for backward compatibility)
def load_squad_documents(
    split: str = SQUAD_SPLIT,
    max_samples: Optional[int] = SQUAD_SUBSET_SIZE,
) -> List[Document]:
    """
    Load SQuAD 2.0 (legacy). Use load_msmarco_documents() for the primary dataset.
    """
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None")
    if max_samples == 0:
        return []

    logger.info("Loading SQuAD 2.0 %s split (legacy)…", split)
    dataset = load_dataset("rajpurkar/squad_v2", split=split)

    docs: List[Document] = []
    seen_contexts: set = set()
    count = 0

    for row in dataset:
        if not row["answers"]["text"]:
            continue
        ctx = row["context"].strip()
        if ctx in seen_contexts:
            continue
        seen_contexts.add(ctx)

        doc = Document(
            page_content=ctx,
            metadata={
                "id": row["id"],
                "title": row["title"],
                "question": row["question"],
                "answers": row["answers"]["text"],
            },
        )
        docs.append(doc)
        count += 1
        if max_samples is not None and count >= max_samples:
            break

    logger.info("Loaded %d unique answerable contexts from SQuAD.", len(docs))
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DOCUMENT CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into character-bounded chunks while preserving metadata.
    RecursiveCharacterTextSplitter respects sentence / paragraph boundaries.
    """
    _validate_chunking(chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunked = splitter.split_documents(docs)
    logger.info(
        "Split %d docs into %d chunks (size=%d, overlap=%d).",
        len(docs), len(chunked), chunk_size, chunk_overlap,
    )
    return chunked


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMBEDDING GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return a HuggingFace embedding model on the configured device."""
    logger.info("Loading embedding model: %s (device=%s)", model_name, EMBEDDING_DEVICE)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VECTOR STORE INDEXING
# ═══════════════════════════════════════════════════════════════════════════════

def _index_manifest_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "_index_manifest.json")


class IndexManifestError(RuntimeError):
    """The persisted index cannot be proven compatible with its side data."""


class CorpusIdentityError(IndexManifestError):
    """The dense index and a derived corpus/side-data view disagree."""


INDEX_MANIFEST_VERSION = 2
_SIDE_DATA_KINDS = ("bm25", "graph", "qrels")


def _json_safe_metadata(metadata: Any) -> Dict[str, Any]:
    """Keep only deterministic corpus fields used by sparse/graph/qrels views."""
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "query": str(metadata.get("query", "")),
        "query_type": str(metadata.get("query_type", "")),
        "is_selected": int(bool(metadata.get("is_selected", 0))),
        "source": str(metadata.get("source", "")),
    }


def _corpus_record(doc: Any) -> Dict[str, Any]:
    return {
        "text": str(getattr(doc, "page_content", "")),
        "metadata": _json_safe_metadata(getattr(doc, "metadata", {})),
    }


def corpus_identity(docs: Iterable[Any]) -> Dict[str, Any]:
    """Return a stable identity for the ordered chunk corpus.

    The text and the metadata fields consumed by BM25, graph traversal and qrels
    are included.  Hashing the same canonical records for a dense build and for
    reconstructed side data prevents a full dense index from being paired with
    the historical 1,000-query development subset (or with another split).
    """
    import hashlib

    digest = hashlib.sha256()
    count = 0
    for doc in docs:
        payload = json.dumps(
            _corpus_record(doc), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
    return {"corpus_id": digest.hexdigest(), "chunk_count": count}


class _CorpusIdentityBuilder:
    """Streaming equivalent of :func:`corpus_identity` for full/train builds."""

    def __init__(self) -> None:
        import hashlib
        self._digest = hashlib.sha256()
        self.chunk_count = 0

    def update(self, docs: Iterable[Any]) -> None:
        for doc in docs:
            payload = json.dumps(
                _corpus_record(doc), sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self._digest.update(len(payload).to_bytes(8, "big"))
            self._digest.update(payload)
            self.chunk_count += 1

    def finish(self) -> Dict[str, Any]:
        return {"corpus_id": self._digest.hexdigest(), "chunk_count": self.chunk_count}


def read_index_manifest(persist_dir: str) -> Dict[str, Any]:
    """Read and structurally validate a manifest; never infer a legacy contract."""
    path = _index_manifest_path(os.path.realpath(persist_dir))
    if not os.path.isfile(path):
        raise IndexManifestError(f"Index manifest is missing: {path}")
    try:
        with open(path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndexManifestError(f"Index manifest is malformed: {path}") from exc
    if not isinstance(manifest, dict):
        raise IndexManifestError(f"Index manifest must be a JSON object: {path}")

    required = (
        "manifest_version", "index_revision", "build_id", "embedding_model",
        "embedding_dim", "chunk_size", "chunk_overlap", "num_chunks",
        "dataset_name", "dataset_config", "dataset_revision", "split", "scope",
        "query_count", "document_count", "corpus_id", "corpus_fingerprint",
        "side_data",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise IndexManifestError(
            f"Index manifest is incomplete ({', '.join(missing)}): {path}"
        )
    if manifest.get("manifest_version") != INDEX_MANIFEST_VERSION:
        raise IndexManifestError(f"Unsupported index manifest version: {path}")
    if manifest.get("index_revision") != INDEX_REVISION:
        raise IndexManifestError(
            f"Index revision mismatch: {manifest.get('index_revision')!r} != {INDEX_REVISION!r}"
        )
    for key in ("embedding_model", "dataset_name", "dataset_config", "dataset_revision",
                "split", "scope", "corpus_id", "corpus_fingerprint"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise IndexManifestError(f"Index manifest field {key!r} is invalid: {path}")
    if manifest["corpus_id"] != manifest["corpus_fingerprint"]:
        raise IndexManifestError(f"Index manifest corpus identity aliases disagree: {path}")
    for key in ("embedding_dim", "chunk_size", "chunk_overlap", "num_chunks",
                "query_count", "document_count"):
        value = manifest.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise IndexManifestError(f"Index manifest count/setting {key!r} is invalid: {path}")
    if manifest["num_chunks"] <= 0:
        raise IndexManifestError(f"Index manifest num_chunks must be positive: {path}")
    side_data = manifest.get("side_data")
    if not isinstance(side_data, list) or set(side_data) != set(_SIDE_DATA_KINDS):
        raise IndexManifestError(f"Index manifest side_data is invalid: {path}")
    return manifest


def _manifest_expected_values(
    *,
    split: Optional[str] = None,
    scope: Optional[str] = None,
    index_revision: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    expected = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "split": split,
        "scope": scope,
        "index_revision": index_revision or INDEX_REVISION,
        "chunk_size": CHUNK_SIZE if chunk_size is None else chunk_size,
        "chunk_overlap": CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
    }
    return {key: value for key, value in expected.items() if value is not None}


def verify_index_manifest(
    persist_dir: str,
    *,
    expected_chunks: Optional[int] = None,
    split: Optional[str] = None,
    scope: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """Fail closed on missing, malformed, stale or internally inconsistent metadata."""
    manifest = read_index_manifest(persist_dir)
    expected = _manifest_expected_values(
        split=split, scope=scope, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if expected_chunks is not None and manifest["num_chunks"] != expected_chunks:
        mismatches["num_chunks"] = (manifest["num_chunks"], expected_chunks)
    if mismatches:
        detail = ", ".join(f"{key}={actual!r} (expected {wanted!r})"
                           for key, (actual, wanted) in mismatches.items())
        raise IndexManifestError(f"Index manifest mismatch at {persist_dir}: {detail}")
    return manifest


def verify_corpus_side_data(
    manifest: Dict[str, Any],
    chunks: Iterable[Any],
    *,
    qrels: Optional[Dict[str, Any]] = None,
    required: Iterable[str] = _SIDE_DATA_KINDS,
) -> Dict[str, Any]:
    """Verify that every requested in-memory side view is present and same-corpus."""
    if not isinstance(manifest, dict):
        raise IndexManifestError("Side-data verification requires a manifest object")
    required_set = set(required)
    if required_set - set(_SIDE_DATA_KINDS):
        raise ValueError("Unknown side-data kind")
    if not required_set.issubset(set(manifest.get("side_data", []))):
        raise IndexManifestError("Manifest does not declare all required side data")
    chunk_list = list(chunks)
    identity = corpus_identity(chunk_list)
    if identity["chunk_count"] != manifest["num_chunks"]:
        raise CorpusIdentityError(
            f"Side-data chunk count {identity['chunk_count']} != dense index {manifest['num_chunks']}"
        )
    if identity["corpus_id"] != manifest["corpus_id"]:
        raise CorpusIdentityError("Dense index and sparse/graph/qrels corpus identities differ")
    if "qrels" in required_set and qrels is not None and not isinstance(qrels, dict):
        raise IndexManifestError("Required qrels side data is missing or malformed")
    return identity


def documents_from_vector_store(vectorstore: Any) -> List[Document]:
    """Reconstruct the exact ordered chunk view represented by a dense store."""
    stored = vectorstore.get(include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or [{} for _ in documents]
    if len(metadatas) != len(documents):
        raise CorpusIdentityError("Dense index documents and metadata counts differ")
    return [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(documents, metadatas)
    ]


def write_index_manifest(
    persist_dir: str,
    embedding_model: str = EMBEDDING_MODEL,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    num_chunks: int = 0,
    *,
    dataset_name: str = DATASET_NAME,
    dataset_config: str = DATASET_CONFIG,
    dataset_revision: str = DATASET_REVISION,
    split: str = DATASET_SPLIT,
    scope: str = "dev",
    query_count: int = 0,
    document_count: int = 0,
    corpus_id: Optional[str] = None,
    index_revision: str = INDEX_REVISION,
    documents_path: Optional[str] = None,
) -> None:
    """
    Persist a small JSON manifest beside a newly built Chroma store.

    Chroma itself does not record which embedding model or chunking parameters
    produced an index, so a stale index built with a different embedder or chunk
    config is invisible until retrieval quality mysteriously degrades. The manifest
    makes the compatible build parameters explicit so ``load_vector_store`` can
    detect — and clearly report — a mismatch instead of silently scoring a mixed
    system.
    """
    from datetime import datetime, timezone

    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    # Generic callers (including deterministic lifecycle fixtures) may not have
    # the source documents available at write time. They still get a complete,
    # structurally valid manifest; corpus equality is verified when side data is
    # reconstructed from an actual chunk list.
    identity = corpus_id or f"unavailable:{num_chunks}"
    manifest = {
        "manifest_version": INDEX_MANIFEST_VERSION,
        "index_revision": index_revision,
        "revision": index_revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_id": uuid.uuid4().hex,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "split": split,
        "scope": scope,
        "query_count": query_count,
        "document_count": document_count,
        "embedding_model": embedding_model,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": num_chunks,
        "corpus_id": identity,
        "corpus_fingerprint": identity,
        "counts": {
            "queries": query_count, "documents": document_count, "chunks": num_chunks,
        },
        "settings": {
            "embedding_model": embedding_model, "embedding_dim": EMBEDDING_DIM,
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        },
        "side_data": list(_SIDE_DATA_KINDS),
    }
    if documents_path is not None:
        manifest["documents_path"] = documents_path
    atomic_write_json(_index_manifest_path(persist_dir), manifest)


def _check_index_manifest(persist_dir: str) -> None:
    """
    Warn when a store's manifest disagrees with the current configuration.

    Missing manifest (stores built before this existed) is reported at debug level,
    not interrupted — an absent manifest means "unknown build parameters," not a
    detected incompatibility.
    """
    path = _index_manifest_path(persist_dir)
    if not os.path.exists(path):
        logger.warning("No index manifest at %s — build parameters unknown.", persist_dir)
        return
    try:
        with open(path) as f:
            manifest = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Index manifest at %s is unreadable: %s", path, exc)
        return

    if not isinstance(manifest, dict):
        logger.warning("Index manifest at %s must be a JSON object.", path)
        return

    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        logger.warning(
            "Index at %s was built with embedding model %r but the current config "
            "uses %r — retrieval scores are NOT comparable/reliable. Rebuild with "
            "`python preprocess_full.py` or remove VECTOR_STORE_DIR override.",
            persist_dir, manifest.get("embedding_model"), EMBEDDING_MODEL,
        )
    if (manifest.get("chunk_size"), manifest.get("chunk_overlap")) != (CHUNK_SIZE, CHUNK_OVERLAP):
        logger.warning(
            "Index at %s was built with chunking %s/%s but the current config uses "
            "%s/%s — BM25/qrels corpus may diverge from the dense index.",
            persist_dir, manifest.get("chunk_size"), manifest.get("chunk_overlap"),
            CHUNK_SIZE, CHUNK_OVERLAP,
        )


def _manifest_matches_current(persist_dir: str, num_chunks: int) -> bool:
    """Return whether a known manifest matches the current index contract.

    Missing/invalid manifests are never treated as reusable. Callers must fail
    closed and leave the existing store untouched rather than silently rebuilding.
    """
    try:
        manifest = verify_index_manifest(
            persist_dir, expected_chunks=num_chunks, split=DATASET_SPLIT, scope="dev",
        )
    except (IndexManifestError, OSError):
        return False
    return manifest.get("query_count") == DATASET_SUBSET_SIZE


def build_vector_store(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_dir: str = VECTOR_STORE_DIR,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    lock=None,
) -> Chroma:
    """Create a NEW index under an exclusive writer barrier, then pin its path.

    Existing stores are never appended to or cleared. Use the staged replacement
    helper from a fresh process to rebuild. The returned handle is read-only by
    contract; mutations belong inside ``persistent_index_writer``.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list")
    with persistent_index_writer(persist_dir, embedding_model, create=True, lock=lock) as vectorstore:
        vectorstore.add_documents(chunks)
        write_index_manifest(
            persist_dir, num_chunks=len(chunks),
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            scope="dev", query_count=DATASET_SUBSET_SIZE,
            document_count=len(chunks), corpus_id=corpus_identity(chunks)["corpus_id"],
        )
    return vectorstore


@contextmanager
def persistent_index_writer(persist_dir, embedding_model, *, create=False, lock=None):
    """Exclusive native writer session; opening is guarded even on first build.

    The native path remains pinned for process lifetime after the session. A
    failed write retains EX until exit. Callers must not mutate the returned
    handle after this context exits. Resumable legacy stores are opened only
    explicitly (``create=False``), never by an unguarded raw constructor.
    """
    persist_dir = os.path.realpath(persist_dir)
    if lock is None:
        with build_lock_for(persist_dir) as owned_lock:
            with persistent_index_writer(
                persist_dir, embedding_model, create=create, lock=owned_lock,
            ) as store:
                yield store
        return
    if not lock._owned or lock.target_path != os.path.realpath(persist_dir):
        raise BuildLockError("Writer requires the matching owned build lock")
    with index_publication_guard(persist_dir, pin_reader=True):
        exists = os.path.isfile(os.path.join(persist_dir, "chroma.sqlite3"))
        if create and exists:
            raise FileExistsError("Index already exists; use a staged rebuild from a fresh process")
        if not create and not exists:
            raise FileNotFoundError(f"Vector store directory not found: {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)
        _NATIVE_PATHS.add(os.path.realpath(persist_dir))
        store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
        yield store


class _FrozenEmbeddings:
    """Spawn-safe precomputed vectors; no model/client serialization needed."""
    def __init__(self, texts, vectors):
        self.vectors = dict(zip(texts, vectors))

    def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]


def _build_frozen_index(chunks, vectors, staged_dir, chunk_size, chunk_overlap):
    store = build_vector_store(
        chunks, _FrozenEmbeddings([doc.page_content for doc in chunks], vectors),
        staged_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    from collections import Counter
    if Counter(store.get(include=["documents"])["documents"]) != Counter(
        doc.page_content for doc in chunks
    ):
        raise RuntimeError("Staged vector store did not match replacement corpus")
    return len(chunks)


def rebuild_vector_store_safely(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_dir: str = VECTOR_STORE_DIR,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Chroma:
    """Stop-serving replacement; isolated staging writer exits before rename."""
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list")
    persist_dir = os.path.realpath(persist_dir)
    with build_lock_for(persist_dir) as build_lock, index_publication_guard(persist_dir, pin_reader=True):
        parent_dir = os.path.dirname(os.path.abspath(persist_dir))
        staged_dir = create_staging_dir(
            parent_dir, ".vector_store.", persist_dir, lock=build_lock
        )
        try:
            vectors = embedding_model.embed_documents([doc.page_content for doc in chunks])
            run_isolated_index_builder(
                _build_frozen_index, chunks, vectors, staged_dir, chunk_size, chunk_overlap,
            )
            from preprocess_full import _publish_vector_store_without_readers
            _publish_vector_store_without_readers(staged_dir, persist_dir)
            # This process has never opened this final path: no stale native cache.
            _NATIVE_PATHS.add(os.path.realpath(persist_dir))
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
        except Exception:
            shutil.rmtree(staged_dir, ignore_errors=True)
            raise
    return vectorstore


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    persist_dir: str = VECTOR_STORE_DIR,
    *,
    allow_unmanifested_legacy: bool = False,
) -> Chroma:
    """Load an already-persisted Chroma vector store.

    ``allow_unmanifested_legacy`` is only for explicitly labelled, non-release
    measurements against old stores. Normal application loads remain fail-closed.
    """
    persist_dir = os.path.realpath(persist_dir)
    acquired = _acquire_reader_lease(persist_dir)
    try:
        if not os.path.isfile(os.path.join(persist_dir, "chroma.sqlite3")):
            raise FileNotFoundError(f"Vector store directory not found: {persist_dir}")
        manifest = None if allow_unmanifested_legacy else read_index_manifest(persist_dir)
        _NATIVE_PATHS.add(os.path.realpath(persist_dir))
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )
        count = vectorstore._collection.count()
        if count == 0:
            raise ValueError(f"Vector store is empty: {persist_dir}")
        if manifest is not None and count != manifest["num_chunks"]:
            raise CorpusIdentityError(
                f"Dense index count {count} != manifest {manifest['num_chunks']}"
            )
        return vectorstore
    except BaseException:
        if acquired:
            with _LEASE_LOCK:
                lease = _READER_LEASES.pop(persist_dir, None)
            if lease is not None:
                lease.close()
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

SYNTHETIC_PROMPT_TEMPLATES = {
    "simple": (
        "Given the following passage, generate ONE clear, specific question whose answer "
        "can be found directly in the passage. Return ONLY the question.\n\nPassage:\n{context}"
    ),
    "paraphrased": (
        "Given the following passage, generate ONE paraphrased or rephrased question that "
        "asks the same thing as a direct question but in a different way. The answer must "
        "be in the passage. Return ONLY the question.\n\nPassage:\n{context}"
    ),
    "ambiguous": (
        "Given the following passage, generate ONE ambiguous or multi-hop question that "
        "requires understanding the passage deeply. Return ONLY the question.\n\nPassage:\n{context}"
    ),
}


def generate_synthetic_queries(
    docs: List[Document] = None,
    qa_pairs: List[Dict[str, Any]] = None,
    num_samples: int = NUM_SYNTHETIC_SAMPLES,
    batch_size: int = SYNTHETIC_BATCH_SIZE,
    output_path: str = SYNTHETIC_DATA_PATH,
    api_key: str = MISTRAL_API_KEY,
    model: str = LLM_MODEL,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic Q&A dataset.

    Strategy:
      - If qa_pairs provided (from MS MARCO), use real queries + augment with LLM rewrites
      - If docs provided, generate questions from passages using LLM
      - Falls back to using MS MARCO's own queries as-is (free, no API needed)

    Returns list of dicts:
      { "id", "context", "question", "query_type", "ground_truth" }
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Resume if checkpoint exists
    existing: List[Dict] = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        logger.info("Resuming from %d existing samples.", len(existing))
        if len(existing) >= num_samples:
            return existing[:num_samples]

    samples = existing[:]

    # Strategy 1: Use MS MARCO Q&A pairs directly (FREE — no API needed)
    if qa_pairs:
        logger.info("Using MS MARCO Q&A pairs as synthetic dataset (free mode).")
        for i, pair in enumerate(qa_pairs[:num_samples]):
            if i < len(existing):
                continue
            samples.append({
                "id": pair.get("id", f"msmarco_{i}"),
                "context": pair.get("context", ""),
                "question": pair["question"],
                "query_type": pair.get("query_type", "unknown"),
                "ground_truth": pair.get("ground_truth", ""),
            })
        _save_json(samples, output_path)
        logger.info("Synthetic dataset saved (%d samples, from MS MARCO queries).", len(samples))
        return samples

    # Strategy 2: Generate from docs using LLM (requires API key)
    if docs and api_key:
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        docs_subset = docs[:num_samples]

        for i, doc in enumerate(docs_subset):
            if i < len(existing):
                continue

            ctx = doc.page_content
            ground_truth = (
                doc.metadata.get("answers", [""])[0]
                if doc.metadata.get("answers")
                else ""
            )
            query_type = SYNTHETIC_QUERY_TYPES[i % len(SYNTHETIC_QUERY_TYPES)]
            prompt = SYNTHETIC_PROMPT_TEMPLATES[query_type].format(context=ctx)

            try:
                resp = client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                question = resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("LLM call failed for doc %d: %s", i, exc)
                question = doc.metadata.get("query", "What does this passage describe?")

            samples.append({
                "id": doc.metadata.get("id", str(i)),
                "context": ctx,
                "question": question,
                "query_type": query_type,
                "ground_truth": ground_truth,
            })

            if (i + 1) % batch_size == 0:
                _save_json(samples, output_path)
                logger.info("Checkpoint: %d / %d samples saved.", len(samples), num_samples)
                time.sleep(0.5)

    # Strategy 3: Use docs metadata queries as-is (free fallback)
    elif docs:
        logger.info("No API key — using passage metadata queries as synthetic dataset.")
        for i, doc in enumerate(docs[:num_samples]):
            if i < len(existing):
                continue
            samples.append({
                "id": doc.metadata.get("id", str(i)),
                "context": doc.page_content,
                "question": doc.metadata.get("query", "What is described in this passage?"),
                "query_type": doc.metadata.get("query_type", "unknown"),
                "ground_truth": (
                    doc.metadata.get("answers", [""])[0]
                    if doc.metadata.get("answers")
                    else ""
                ),
            })

    _save_json(samples, output_path)
    logger.info("Synthetic dataset saved to %s (%d samples).", output_path, len(samples))
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PARENT-CHILD CHUNK BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_parent_child_chunks(
    docs: List[Document],
    parent_chunk_size: int = 1200,
    child_chunk_size: int = 300,
    overlap: int = 50,
) -> Dict[str, Any]:
    """
    For each parent chunk, create smaller child chunks.
    Returns { "parents": [...], "children": [...], "child_to_parent": {child_id: parent_id} }
    """
    _validate_chunking(parent_chunk_size, overlap)
    _validate_chunking(child_chunk_size, overlap)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=overlap
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=overlap
    )

    parents = parent_splitter.split_documents(docs)
    children: List[Document] = []
    child_to_parent: Dict[str, str] = {}

    for p_idx, parent in enumerate(parents):
        parent_id = f"parent_{p_idx}"
        parent.metadata["parent_id"] = parent_id
        child_docs = child_splitter.split_documents([parent])
        for c_idx, child in enumerate(child_docs):
            child_id = f"child_{p_idx}_{c_idx}"
            child.metadata["child_id"] = child_id
            child.metadata["parent_id"] = parent_id
            children.append(child)
            child_to_parent[child_id] = parent_id

    logger.info(
        "Parent-Child split: %d parents → %d children.", len(parents), len(children)
    )
    return {"parents": parents, "children": children, "child_to_parent": child_to_parent}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    """Reject invalid splitter settings before LangChain raises a vague error."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

def _save_json(data: Any, path: str) -> None:
    atomic_write_json(path, data)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_module1(
    generate_synthetic: bool = True,
    dataset: str = "msmarco",
) -> Dict[str, Any]:
    """
    Full Module 1 pipeline. Returns artefacts for downstream modules.

    Args:
        generate_synthetic: Whether to generate/load synthetic Q&A dataset
        dataset: "msmarco" (default) or "squad" (legacy)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load documents based on dataset choice
    if dataset == "msmarco":
        logger.info("═══ Loading MS MARCO v2.1 ═══")
        docs = load_msmarco_documents()
        qa_pairs = load_msmarco_qa_pairs()
    else:
        logger.info("═══ Loading SQuAD 2.0 (legacy) ═══")
        docs = load_squad_documents()
        qa_pairs = None

    # Chunk
    chunks = chunk_documents(docs)

    # Embed + index — reuse an existing persisted store if one is already on disk
    embedding_model = get_embedding_model()
    existing_store = os.path.exists(os.path.join(VECTOR_STORE_DIR, "chroma.sqlite3"))
    manifest_compatible = _manifest_matches_current(VECTOR_STORE_DIR, len(chunks))
    if existing_store and manifest_compatible:
        logger.info("Found existing vector store at %s — loading instead of rebuilding.", VECTOR_STORE_DIR)
        vectorstore = load_vector_store(embedding_model)
    else:
        if existing_store:
            raise IndexManifestError(
                f"Existing vector store at {VECTOR_STORE_DIR} does not match the configured "
                "development corpus. Refusing an implicit rebuild; run the intended "
                "preprocessing workflow into a new staging path and publish only after "
                "its manifest and side data are verified."
            )
        else:
            # A first build is also serialized: two workers starting with no
            # directory must not both write Chroma's persistent files. Recheck
            # after acquiring the lock because another worker may have finished
            # while this one was waiting.
            with build_lock_for(VECTOR_STORE_DIR) as first_build_lock:
                if os.path.exists(os.path.join(VECTOR_STORE_DIR, "chroma.sqlite3")):
                    if _manifest_matches_current(VECTOR_STORE_DIR, len(chunks)):
                        vectorstore = load_vector_store(embedding_model)
                    else:
                        raise IndexManifestError(
                            f"Existing vector store at {VECTOR_STORE_DIR} is incompatible; "
                            "refusing an implicit rebuild."
                        )
                else:
                    vectorstore = build_vector_store(chunks, embedding_model, lock=first_build_lock)

    # Synthetic data
    synthetic_data: List[Dict] = []
    if generate_synthetic:
        synthetic_data = generate_synthetic_queries(
            docs=docs,
            qa_pairs=qa_pairs,
        )

    # Parent-child
    pc = build_parent_child_chunks(docs)
    child_vectorstore = None
    if pc.get("children"):
        from langchain_chroma import Chroma as _Chroma
        logger.info("Building child-chunk vector store (%d children)…", len(pc["children"]))
        child_vectorstore = _Chroma.from_documents(
            pc["children"],
            embedding_model,
            collection_name="parent_child_children",
        )
        logger.info("Child vector store ready — %d chunks indexed.", len(pc["children"]))

    logger.info("═══ Module 1 complete ═══")
    logger.info("  Documents: %d", len(docs))
    logger.info("  Chunks: %d", len(chunks))
    logger.info("  Synthetic Q&A: %d", len(synthetic_data))
    logger.info("  Parent chunks: %d", len(pc["parents"]))
    logger.info("  Child chunks: %d", len(pc["children"]))

    return {
        "docs": docs,
        "chunks": chunks,
        "embedding_model": embedding_model,
        "vectorstore": vectorstore,
        "synthetic_data": synthetic_data,
        "parent_child": pc,
        "child_vectorstore": child_vectorstore,
        "qa_pairs": qa_pairs,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge Graph Builder — for GRACE-Ω architecture
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeGraphBuilder:
    """
    Builds a lightweight knowledge graph from document chunks.
    Uses SpaCy NER for entity extraction and co-occurrence
    for relation edges.

    Used exclusively by GRACEOmegaRAG for:
    1. Graph traversal retrieval (RELATIONAL queries)
    2. Cross-structure verification (pruning RAPTOR chunks)

    This is Shakeel's novelty — formalised here as:
    For query q, extract entity set E_q.
    For each RAPTOR chunk c, keep c if:
        |tokens(c) ∩ E_q| >= theta (default theta=3)
    This is the subgraph-to-tree intersection algorithm.
    """

    def __init__(self):
        self.entity_to_chunks: Dict[str, List[str]] = {}
        self.graph: Dict[str, List[str]] = {}
        self.chunk_texts: Dict[str, str] = {}
        self._nlp = None

    def _load_nlp(self):
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True
                )
                self._nlp = spacy.load("en_core_web_sm")
            return self._nlp
        except Exception as e:
            logger.warning(f"SpaCy unavailable: {e}. KG will use regex fallback.")
            return None

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        nlp = self._load_nlp()
        if nlp:
            doc = nlp(text[:500])
            return list(set([
                ent.text.lower().strip()
                for ent in doc.ents
                if ent.label_ in [
                    "PERSON", "ORG", "GPE", "PRODUCT",
                    "EVENT", "WORK_OF_ART", "LAW", "NORP",
                    "FAC", "LOC"
                ]
            ]))
        else:
            # Regex fallback: capitalised phrases
            return list(set([
                m.group().lower()
                for m in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            ]))

    def build(self, chunks: List[Dict]) -> None:
        """
        Build entity-to-chunk mapping and co-occurrence graph.
        chunks: list of {id, text, pid, chunk_idx}
        """
        logger.info("Building knowledge graph from %d chunks...", len(chunks))
        for chunk in chunks:
            chunk_id = chunk["id"]
            text = chunk["text"]
            self.chunk_texts[chunk_id] = text
            entities = self._extract_entities(text)

            # Map each entity to the chunks containing it
            for ent in entities:
                if ent not in self.entity_to_chunks:
                    self.entity_to_chunks[ent] = []
                if chunk_id not in self.entity_to_chunks[ent]:
                    self.entity_to_chunks[ent].append(chunk_id)

            # Co-occurrence edges between entities in same chunk
            for i, e1 in enumerate(entities):
                if e1 not in self.graph:
                    self.graph[e1] = []
                for e2 in entities[i+1:]:
                    if e2 not in self.graph[e1]:
                        self.graph[e1].append(e2)
                    self.graph.setdefault(e2, [])
                    if e1 not in self.graph[e2]:
                        self.graph[e2].append(e1)

        logger.info(
            "KG built: %d entities, %d edges",
            len(self.entity_to_chunks),
            sum(len(v) for v in self.graph.values())
        )

    def query_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """
        Find chunks linked to entities in the query.
        Performs 1-hop graph traversal.
        Returns list of {id, text, score}
        """
        query_entities = self._extract_entities(query)
        matched_ids: Dict[str, float] = {}

        for ent in query_entities:
            # Direct entity match — high score
            if ent in self.entity_to_chunks:
                for cid in self.entity_to_chunks[ent]:
                    matched_ids[cid] = matched_ids.get(cid, 0) + 1.0

            # 1-hop neighbour — lower score
            if ent in self.graph:
                for neighbour in self.graph[ent][:3]:
                    if neighbour in self.entity_to_chunks:
                        for cid in self.entity_to_chunks[neighbour][:2]:
                            matched_ids[cid] = matched_ids.get(cid, 0) + 0.5

        # Sort by score
        sorted_ids = sorted(
            matched_ids.items(), key=lambda x: x[1], reverse=True
        )[:k]

        return [
            {
                "id": cid,
                "text": self.chunk_texts.get(cid, ""),
                "score": score
            }
            for cid, score in sorted_ids
            if self.chunk_texts.get(cid, "")
        ]

    def get_entity_set(self, query: str) -> set:
        """
        Returns all entity tokens relevant to this query
        including 1-hop neighbours.
        Used for cross-structure verification.
        """
        query_entities = self._extract_entities(query)
        all_entities = set(query_entities)
        for ent in query_entities:
            if ent in self.graph:
                all_entities.update(self.graph[ent][:5])
        # Flatten to tokens
        all_tokens = set()
        for ent in all_entities:
            all_tokens.update(ent.lower().split())
        return all_tokens

if __name__ == "__main__":
    print("=" * 60)
    print("  Module 1 — Data Preprocessing (MS MARCO v2.1)")
    print("=" * 60)
    artifacts = run_module1(generate_synthetic=True, dataset="msmarco")
    print(f"\n  Documents loaded: {len(artifacts['docs'])}")
    print(f"  Chunks created:   {len(artifacts['chunks'])}")
    print(f"  Synthetic Q&A:    {len(artifacts['synthetic_data'])}")
    print(f"  Parent chunks:    {len(artifacts['parent_child']['parents'])}")
    print(f"  Child chunks:     {len(artifacts['parent_child']['children'])}")
    print("\n  ✓ Ready for Module 2 (RAG architectures)")
