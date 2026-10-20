#!/usr/bin/env python3
"""Create-only, offline evidence for the real upload subprocess boundary.

Example (destination must not exist)::

    PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/validate_parser_isolation.py \
        --output outputs/security-v4-parser-new-run.json

All fixtures are finite synthetic local files. Native parsers handle valid tiny
PDF/DOCX only. Fault workers use bounded output/allocation or a CPU limit and
wall deadline. No existing collection, corpus, Redis, model or provider is used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import struct
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

# Set this before application/config imports; never discover/read private dotenv.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
for _name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"):
    os.environ[_name] = ""
for _name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module5.document_upload import DocumentUploader, UploadValidationError  # noqa: E402
import module5.document_upload as uploads  # noqa: E402
import module5.parser_isolation as isolation  # noqa: E402


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _pdf():
    import pymupdf
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "A finite isolated PDF fixture.")
        return document.tobytes()


def _docx():
    from docx import Document
    document = Document()
    document.add_paragraph("A finite isolated DOCX fixture.")
    target = io.BytesIO()
    document.save(target)
    return target.getvalue()


def _zip(entries):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries:
            archive.writestr(name, value)
    return target.getvalue()


class Evidence:
    def __init__(self):
        self.cases = []
        self.workers = []

    def case(self, name, action):
        started = time.monotonic()
        try:
            details = action() or {}
            self.cases.append({"name": name, "passed": True, "elapsed_seconds": time.monotonic() - started, "details": details})
        except Exception as exc:
            # Only runner-authored assertion messages and generic public errors;
            # no untrusted parser diagnostics or upload contents are recorded.
            message = str(exc) if isinstance(exc, (AssertionError, UploadValidationError)) else type(exc).__name__
            self.cases.append({"name": name, "passed": False, "elapsed_seconds": time.monotonic() - started, "error": message[:300]})

    @contextmanager
    def tracking(self):
        original = isolation.subprocess.Popen

        def launch(*args, **kwargs):
            child = original(*args, **kwargs)
            self.workers.append((child, Path(kwargs["cwd"])))
            return child

        with patch.object(isolation.subprocess, "Popen", launch):
            yield

    def cleanup_receipts(self):
        receipts = []
        for child, directory in self.workers:
            _check(child.poll() is not None, "worker still running")
            _check(child.stdout.closed, "worker pipe still open")
            _check(not directory.exists(), "owned temporary directory survived")
            try:
                os.waitpid(child.pid, os.WNOHANG)
            except ChildProcessError:
                reaped = True
            else:
                reaped = False
            _check(reaped, "child was not reaped by supervisor")
            receipts.append({"pid": child.pid, "returncode": child.returncode, "reaped": reaped, "temporary_directory_removed": True, "stdout_closed": True})
        return {"worker_count": len(receipts), "workers": receipts}


@contextmanager
def _fault_worker(directory, source):
    target = directory / "synthetic_worker.py"
    target.write_text(source, encoding="utf-8")
    with patch.object(isolation, "_WORKER_PATH", target):
        yield


def _expect_rejection(action, expected):
    try:
        action()
    except UploadValidationError as exc:
        _check(expected in str(exc), "wrong public rejection category")
        return {"public_error": str(exc)}
    raise AssertionError("unsafe operation was accepted")


def _valid_case(extension, content):
    def forbidden(*args, **kwargs):
        raise AssertionError("application process attempted parsing")

    uploader = DocumentUploader()
    with patch.object(DocumentUploader, "_parse_pdf", forbidden), patch.object(DocumentUploader, "_parse_docx", forbidden), patch.object(DocumentUploader, "_parse_text", forbidden):
        docs = uploader.parse_file(content, f"fixture.{extension}")
    receipt = uploader._last_parser_receipt
    _check(receipt["pid"] != os.getpid() and receipt["parent_pid"] == os.getpid(), "no process boundary")
    _check(receipt["heavy_stack_loaded"] is False, "worker loaded ML/index stack")
    _check(receipt["limits"]["RLIMIT_CPU"] == [8, 8], "CPU limit missing")
    _check(receipt["limits"]["RLIMIT_CORE"] == [0, 0], "core limit missing")
    _check(receipt["limits"]["RLIMIT_FSIZE"] == [16 * 1024 * 1024] * 2, "file limit missing")
    _check(receipt["observed_peak_rss_bytes"] <= 512 * 1024 * 1024, "RSS bound exceeded")
    if sys.platform == "linux":
        _check(receipt["limits"]["RLIMIT_AS"] == [512 * 1024 * 1024] * 2, "address space limit missing")
    return {"documents": len(docs), "input_bytes": len(content), "input_sha256": hashlib.sha256(content).hexdigest(), "worker": receipt}


def _native_import_order(directory, extension, content):
    worker = isolation._WORKER_PATH
    marker = directory / "native-imports.json"
    source = f'''
import builtins, json, os, resource, runpy, sys
worker = runpy.run_path({str(worker)!r})
original = builtins.__import__
seen = []
def guarded(name, *args, **kwargs):
    if name in {{"pymupdf", "docx", "zipfile"}}:
        assert resource.getrlimit(resource.RLIMIT_CPU) == (8, 8)
        assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
        assert resource.getrlimit(resource.RLIMIT_FSIZE) == (16777216, 16777216)
        if sys.platform == "linux":
            assert resource.getrlimit(resource.RLIMIT_AS) == (536870912, 536870912)
        assert not any(key in sys.modules for key in ("torch", "chromadb"))
        assert not any(key in os.environ for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"))
        seen.append(name)
        with open({str(marker)!r}, "w") as handle:
            json.dump(seen, handle)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
raise SystemExit(worker["main"]())
'''
    with _fault_worker(directory, source):
        DocumentUploader().parse_file(content, f"order.{extension}")
    seen = json.loads(marker.read_text())
    _check(("pymupdf" if extension == "pdf" else "docx") in seen, "native import hook not reached")
    return {"guarded_imports": seen, "resource_limits_preceded_imports": True, "provider_environment_absent": True}


def _fault_case(directory, mode):
    actions = {
        "timeout": "time.sleep(5)",
        "crash": "os._exit(19)",
        "output": "os.write(1, b'x' * 8192); time.sleep(5)",
        "memory": "allocation = bytearray(64 * 1024 * 1024); time.sleep(5)",
        "json": "os.write(1, b'{not-json')",
        "pickle": "os.write(1, bytes([128, 4, 99]) + b'os\\nsystem\\n.')",
    }
    categories = {"timeout": "time limit", "crash": "stopped", "output": "output exceeded", "memory": "memory limit", "json": "invalid result", "pickle": "invalid result"}
    with _fault_worker(directory, "import os, time\n" + actions[mode] + "\n"):
        uploader = DocumentUploader(parser_wall_seconds=0.7, parser_memory_bytes=32 * 1024 * 1024, parser_output_bytes=1024)
        result = _expect_rejection(lambda: uploader.parse_file(b"Finite fault probe input.", "probe.txt"), categories[mode])
        _check(uploader.filename == "", "failed parser published filename")
        return result


def _cpu_case(directory):
    worker = isolation._WORKER_PATH
    marker = directory / "cpu-applied"
    source = f'''
import runpy, sys
from pathlib import Path
worker = runpy.run_path({str(worker)!r})
worker["_apply_resource_limits"](int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
Path({str(marker)!r}).write_text("applied")
while True:
    pass
'''
    with _fault_worker(directory, source):
        result = _expect_rejection(lambda: DocumentUploader(parser_cpu_seconds=1, parser_wall_seconds=5).parse_file(b"CPU bounded input.", "cpu.txt"), "stopped")
    _check(marker.exists(), "CPU limit was not installed")
    return result


def _admission_case(directory, evidence):
    release = directory / "release"
    source = f'''
import json, os, sys, time
from pathlib import Path
request = json.load(sys.stdin)
Path({str(directory)!r}, "ready-" + str(os.getpid())).write_text("ready")
deadline = time.monotonic() + 5
while not Path({str(release)!r}).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
print(json.dumps({{"version":1,"documents":[{{"page_content":"Finite admitted worker output.","metadata":{{"source":request["filename"],"paragraph":0}}}}],"worker":{{"pid":os.getpid()}}}}))
'''
    before = len(evidence.workers)
    with _fault_worker(directory, source), ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(DocumentUploader().parse_file, b"bounded", f"admitted-{index}.txt") for index in range(2)]
        try:
            deadline = time.monotonic() + 3
            while len(list(directory.glob("ready-*"))) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            _check(len(list(directory.glob("ready-*"))) == 2, "two workers were not simultaneously admitted")
            started = time.monotonic()
            for _ in range(20):
                _expect_rejection(lambda: DocumentUploader().parse_file(b"flood", "flood.txt"), "busy")
            latency = time.monotonic() - started
            _check(len(evidence.workers) - before == 2, "flood spawned another worker")
            _check(latency < 0.5, "flood was queued")
        finally:
            release.touch()
        _check(all(task.result(timeout=3) for task in tasks), "admitted workers failed")
    return {"concurrent_workers": 2, "rejected_requests": 20, "extra_workers": 0, "rejection_total_seconds": latency}


def _transaction_case(directory):
    # An owned local store double keeps this experiment focused on parse/index
    # publication; tests/test_upload_isolation.py also exercises actual Chroma.
    live = []

    class Store:
        def __init__(self, **kwargs):
            live.append(self)

        def add_documents(self, docs):
            self.docs = list(docs)
            if embeddings.fail:
                raise RuntimeError("synthetic indexing failure")

        def delete_collection(self):
            live.remove(self)

    class Embeddings:
        fail = False

    embeddings = Embeddings()
    uploader = DocumentUploader(embeddings)
    preserved = []
    with patch.object(uploads, "Chroma", Store):
        try:
            uploader.index_documents(b"The previous upload remains active.", "previous.txt")
            previous, chunks = uploader.vectorstore, uploader.chunks
            with _fault_worker(directory, "import time\ntime.sleep(5)\n"):
                uploader._parser_limits = isolation.IsolationLimits(wall_seconds=0.5)
                _expect_rejection(lambda: uploader.index_documents(b"Rejected replacement.", "failed.txt"), "time limit")
            _check(uploader.vectorstore is previous and uploader.chunks == chunks and uploader.filename == "previous.txt" and live == [previous], "parse failure changed active collection")
            preserved.append("parse_timeout")
            uploader._parser_limits = isolation.IsolationLimits()
            embeddings.fail = True
            try:
                uploader.index_documents(b"Index failure replacement.", "failed.txt")
            except RuntimeError:
                pass
            else:
                raise AssertionError("index fault was not reached")
            _check(uploader.vectorstore is previous and uploader.chunks == chunks and uploader.filename == "previous.txt" and live == [previous], "index failure changed active collection")
            preserved.append("index_failure")
        finally:
            uploader.clear()
    _check(not live, "owned temporary collection survived")
    return {"previous_collection_preserved_for": preserved, "store": "owned local double; real Chroma covered by focused tests"}


def run():
    evidence = Evidence()
    with tempfile.TemporaryDirectory(prefix="rag-parser-evidence-") as root, evidence.tracking():
        directory = Path(root)
        fixtures = {"pdf": _pdf(), "docx": _docx(), "txt": b"A finite isolated text fixture.", "md": b"# A finite isolated markdown fixture.", "csv": b"name,description\none,A finite CSV fixture"}
        for extension, content in fixtures.items():
            evidence.case(f"valid_{extension}_actual_child_limits_and_lightweight_imports", lambda extension=extension, content=content: _valid_case(extension, content))
        for extension in ("pdf", "docx"):
            evidence.case(f"{extension}_limits_before_native_import", lambda extension=extension: _native_import_order(directory, extension, fixtures[extension]))
        for mode in ("timeout", "crash", "output", "memory", "json", "pickle"):
            evidence.case(f"finite_fault_{mode}", lambda mode=mode: _fault_case(directory, mode))
        evidence.case("actual_cpu_limit", lambda: _cpu_case(directory))
        evidence.case("bounded_admission_flood", lambda: _admission_case(directory, evidence))
        evidence.case("transactional_parse_and_index_publication", lambda: _transaction_case(directory))
        malformed = {
            "traversal": _zip([("../private-canary.xml", b"finite")]),
            "truncated": b"PK\x03\x04finite-not-a-zip",
            "entries": _zip([("one", b"1"), ("two", b"2")]),
            "expanded": _zip([("one", b"123456"), ("two", b"123456")]),
        }
        crc = bytearray(_zip([("word/document.xml", b"finite fixture")]))
        struct.pack_into("<I", crc, crc.index(b"PK\x01\x02") + 16, 0)
        malformed["crc"] = bytes(crc)
        for name, content in malformed.items():
            options = {"max_docx_entries": 1} if name == "entries" else {"max_docx_uncompressed_bytes": 10} if name == "expanded" else {}
            evidence.case(f"archive_{name}", lambda content=content, options=options: _expect_rejection(lambda: DocumentUploader(**options).parse_file(content, "archive.docx"), "DOCX"))
        evidence.case("encoded_input_bound", lambda: _expect_rejection(lambda: DocumentUploader(max_bytes=4).parse_file(b"12345", "small.txt"), "too large"))
        evidence.case("extraction_bound", lambda: _expect_rejection(lambda: DocumentUploader(max_extracted_text_chars_per_document=12).parse_file(b"A bounded but over-limit paragraph.", "small.txt"), "per-document"))

        def monitor_unavailable():
            with patch.object(isolation, "_rss_reader", side_effect=isolation.ParserIsolationError("Document parser memory monitoring is unavailable.")):
                return _expect_rejection(lambda: DocumentUploader().parse_file(b"Must fail closed.", "closed.txt"), "unavailable")

        evidence.case("unavailable_monitor_fail_closed", monitor_unavailable)
        evidence.case("kill_reap_pipe_and_temporary_cleanup", evidence.cleanup_receipts)
    cases = evidence.cases
    return {
        "schema": "security-v4.parser-isolation/v1",
        "platform": platform.platform(), "python": platform.python_version(),
        "parent_pid": os.getpid(), "passed": all(case["passed"] for case in cases),
        "case_count": len(cases), "passed_count": sum(case["passed"] for case in cases),
        "worker_count": len(evidence.workers), "cases": cases,
        "implementation_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            ROOT / "module5/document_upload.py", ROOT / "module5/parser_worker.py", ROOT / "module5/parser_isolation.py", Path(__file__).resolve(),
        )},
        "defaults": {"encoded_upload_bytes": DocumentUploader().max_bytes, "wall_seconds": 15, "cpu_seconds": 8, "memory_bytes": 512 * 1024 * 1024, "output_bytes": 16 * 1024 * 1024, "concurrent_workers_per_process": 2, "rss_poll_seconds": isolation._POLL_SECONDS},
        "residual_gaps": [
            "Only the recorded host platform was executed; Linux AS enforcement needs Linux execution when this receipt is macOS.",
            "Resource isolation is not a privilege, filesystem, network, syscall or native-code-execution sandbox.",
            "macOS RSS is sampled at 20 ms; transient overshoot is possible. Linux additionally enforces RLIMIT_AS.",
            "RSS observes the parser PID, not aggregate descendant memory; CPU/address-space limits are inherited per process. Owned process groups are killed on all outcomes.",
            "Admission is per application process, not cluster-wide; API body buffering and indexing/embedding are outside the parsing admission budget.",
            "Private helper calls can still parse in-process for unit testing; production parse_file/index_documents have no unrestricted fallback.",
            "All fixtures are finite synthetic probes, not comprehensive malformed-document fuzzing or dependency exploit coverage.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.parent != ROOT / "outputs" or not destination.name.startswith("security-v4-parser") or destination.suffix != ".json":
        parser.error("output must be a new outputs/security-v4-parser*.json file")
    # Reserve create-only before running any work; reruns never clobber history.
    try:
        with destination.open("x", encoding="utf-8") as target:
            result = run()
            json.dump(result, target, indent=2, sort_keys=True)
            target.write("\n")
    except FileExistsError:
        parser.error("output already exists; choose a new evidence filename")
    print(json.dumps({"output": str(destination.relative_to(ROOT)), "passed": result["passed"], "cases": result["case_count"], "passed_cases": result["passed_count"], "workers": result["worker_count"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
