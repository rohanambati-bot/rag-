"""Bounded POSIX subprocess supervision for untrusted document parsing.

Linux: hard RLIMIT_AS plus sampled RSS. macOS: libproc RSS
monitoring (AS/RSS rlimits are not reliable there). Sampling may overshoot by a
poll interval. Windows/other platforms fail closed. This is resource isolation,
not a filesystem/network/privilege sandbox against native code execution.
"""

from __future__ import annotations

import json
import logging
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .parser_worker import ERROR_MESSAGES, _TextBudget, UploadValidationError

DEFAULT_PARSER_WALL_SECONDS = 15.0
DEFAULT_PARSER_CPU_SECONDS = 8
DEFAULT_PARSER_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_PARSER_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_CONCURRENT_PARSERS = 2
_POLL_SECONDS = 0.02
_WORKER_PATH = Path(__file__).with_name("parser_worker.py")
_admission = threading.BoundedSemaphore(MAX_CONCURRENT_PARSERS)
logger = logging.getLogger(__name__)


def _reset_admission():
    global _admission
    _admission = threading.BoundedSemaphore(MAX_CONCURRENT_PARSERS)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_admission)


class ParserIsolationError(UploadValidationError):
    """Public isolation rejection; never contains worker diagnostics."""


@dataclass(frozen=True)
class IsolationLimits:
    wall_seconds: float = DEFAULT_PARSER_WALL_SECONDS
    cpu_seconds: int = DEFAULT_PARSER_CPU_SECONDS
    memory_bytes: int = DEFAULT_PARSER_MEMORY_BYTES
    output_bytes: int = DEFAULT_PARSER_OUTPUT_BYTES

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
            if name != "wall_seconds" and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")


def _rss_reader():
    if os.name != "posix" or sys.platform not in {"linux", "darwin"}:
        raise ParserIsolationError("Document parser isolation is unavailable on this platform.")
    if sys.platform == "linux":
        page_size = os.sysconf("SC_PAGE_SIZE")

        def read_linux(pid):
            try:
                with open(f"/proc/{pid}/statm", "rb") as source:
                    fields = source.read(256).split()
                return int(fields[1]) * page_size
            except (OSError, ValueError, IndexError):
                return None

        return read_linux
    try:
        import ctypes

        class TaskInfo(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "virtual_size", "resident_size", "total_user", "total_system", "threads_user", "threads_system",
            )] + [(name, ctypes.c_int32) for name in (
                "policy", "faults", "pageins", "cow_faults", "messages_sent", "messages_received",
                "syscalls_mach", "syscalls_unix", "csw", "threadnum", "numrunning", "priority",
            )]

        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        libproc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        libproc.proc_pidinfo.restype = ctypes.c_int

        def read_darwin(pid):
            info = TaskInfo()
            size = ctypes.sizeof(info)
            result = libproc.proc_pidinfo(pid, 4, 0, ctypes.byref(info), size)  # PROC_PIDTASKINFO
            return int(info.resident_size) if result == size else None

        return read_darwin
    except (OSError, AttributeError):
        raise ParserIsolationError("Document parser memory monitoring is unavailable.") from None


def _kill_and_reap(process):
    # start_new_session gives this invocation its own process group. Kill that
    # group even after normal exit, so inherited-pipe descendants cannot linger.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin can return EPERM for a process group whose leader has exited
        # but has not yet become waitable. Reap that transition; if it is still
        # alive, fall back to killing the exact owned PID.
        try:
            process.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            process.kill()
    finally:
        process.wait()
        if process.stdout is not None:
            process.stdout.close()


def _read_output(process, limits, deadline, rss_reader):
    output = bytearray()
    peak_rss = 0
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        eof = False
        while not eof or process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ParserIsolationError("Document parsing exceeded its time limit.")
            rss = rss_reader(process.pid)
            if rss is None and process.poll() is None:
                # libproc can stop reporting a terminating process just before
                # waitpid observes its exit. A bounded wait distinguishes that
                # race from an unavailable monitor on a live worker.
                try:
                    process.wait(timeout=min(_POLL_SECONDS, remaining))
                except subprocess.TimeoutExpired:
                    raise ParserIsolationError("Document parser memory monitoring failed.") from None
            peak_rss = max(peak_rss, rss or 0)
            if peak_rss > limits.memory_bytes:
                raise ParserIsolationError("Document parsing exceeded its memory limit.")
            for key, _ in selector.select(min(_POLL_SECONDS, remaining)):
                block = os.read(key.fd, min(64 * 1024, limits.output_bytes - len(output) + 1))
                if not block:
                    selector.unregister(key.fileobj)
                    eof = True
                    continue
                output.extend(block)
                if len(output) > limits.output_bytes:
                    raise ParserIsolationError("Document parser output exceeded its limit.")
    if process.returncode != 0:
        raise ParserIsolationError("Document parser stopped before completing safely.")
    return output, peak_rss


def _validate_response(raw, filename, policy):
    try:
        response = json.loads(raw)
        if not isinstance(response, dict) or response.get("version") != 1:
            raise ValueError
        if "error" in response:
            code = response["error"]
            if not isinstance(code, str) or code not in ERROR_MESSAGES:
                raise ValueError
            raise UploadValidationError(ERROR_MESSAGES[code])
        docs, worker = response["documents"], response["worker"]
        if not isinstance(docs, list) or not 0 < len(docs) <= policy["max_parsed_documents"] or not isinstance(worker, dict):
            raise ValueError
        budget = _TextBudget(
            max_upload_bytes=policy["max_extracted_text_bytes_per_upload"],
            max_upload_chars=policy["max_extracted_text_chars_per_upload"],
            max_document_bytes=policy["max_extracted_text_bytes_per_document"],
            max_document_chars=policy["max_extracted_text_chars_per_document"],
        )
        for doc in docs:
            if not isinstance(doc, dict) or set(doc) != {"page_content", "metadata"} or not isinstance(doc["page_content"], str):
                raise ValueError
            metadata = doc["metadata"]
            if not isinstance(metadata, dict) or metadata.get("source") != filename or set(metadata) not in ({"source", "page"}, {"source", "paragraph"}):
                raise ValueError
            key = "page" if "page" in metadata else "paragraph"
            if type(metadata[key]) is not int or metadata[key] < (1 if key == "page" else 0):
                raise ValueError
            if budget.add_text(doc["page_content"], "worker output") != doc["page_content"]:
                raise ValueError
        return docs, worker
    except UploadValidationError:
        raise
    except (ValueError, KeyError, TypeError, RecursionError, UnicodeError):
        raise ParserIsolationError("Document parser returned an invalid result.") from None


def parse_isolated(content: bytes, filename: str, extension: str, policy: dict, limits: IsolationLimits):
    """Return validated document dictionaries and an internal execution receipt.

    Admission is shared by all uploader instances in this process and never
    waits: at most two workers and no retained parser queue. No fallback exists.
    """
    if not _admission.acquire(blocking=False):
        raise ParserIsolationError("Document parser is busy. Please retry later.")
    process = None
    started = time.monotonic()
    try:
        rss_reader = _rss_reader()
        if len(content) > policy["max_bytes"]:
            raise UploadValidationError(ERROR_MESSAGES["upload_bytes"])
        request = json.dumps({"filename": filename, "extension": extension, "policy": policy}).encode("utf-8")
        if len(request) > 16 * 1024:
            raise ParserIsolationError("Document parser request exceeded its limit.")
        with tempfile.TemporaryDirectory(prefix="rag-parser-") as workdir:
            directory = Path(workdir)
            (directory / "upload.bin").write_bytes(content)
            (directory / "request.json").write_bytes(request)
            # Do not inherit provider credentials, PYTHONPATH, dotenv settings or
            # the caller's HOME. Python -I also ignores user-site/startup hooks.
            environment = {"PATH": os.defpath, "HOME": workdir, "TMPDIR": workdir, "LANG": "C.UTF-8",
                           "PYTHON_DOTENV_DISABLED": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            with (directory / "request.json").open("rb") as stdin:
                process = subprocess.Popen(
                    [sys.executable, "-I", str(_WORKER_PATH), str(limits.cpu_seconds), str(limits.memory_bytes), str(limits.output_bytes)],
                    stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    cwd=workdir, env=environment, start_new_session=True, close_fds=True,
                )
                try:
                    raw, peak = _read_output(process, limits, started + limits.wall_seconds, rss_reader)
                finally:
                    _kill_and_reap(process)
            documents, receipt = _validate_response(raw, filename, policy)
            receipt.update({"observed_peak_rss_bytes": peak, "output_bytes": len(raw), "elapsed_seconds": time.monotonic() - started})
            return documents, receipt
    except UploadValidationError:
        raise
    except Exception:
        logger.debug("Parser supervision failed", exc_info=True)
        raise ParserIsolationError("Document parser could not complete safely.") from None
    finally:
        _admission.release()
