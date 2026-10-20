"""Real, finite subprocess boundary tests; no hostile files or provider calls."""

from __future__ import annotations

import io
import json
import os
import struct
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import module5.parser_isolation as isolation
from module5.document_upload import DocumentUploader, UploadValidationError
from module5.parser_worker import POLICY_DEFAULTS


def _pdf():
    import pymupdf
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "A valid isolated PDF paragraph.")
        return document.tobytes()


def _docx():
    from docx import Document
    document = Document()
    document.add_paragraph("A valid isolated DOCX paragraph.")
    stream = io.BytesIO()
    document.save(stream)
    return stream.getvalue()


@pytest.fixture
def tracked_workers(monkeypatch):
    real_popen = isolation.subprocess.Popen
    launched = []

    def launch(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        launched.append((process, Path(kwargs["cwd"]), kwargs))
        return process

    monkeypatch.setattr(isolation.subprocess, "Popen", launch)
    return launched


def _assert_clean(launched):
    assert launched
    for process, directory, _ in launched:
        assert process.poll() is not None
        assert process.stdout.closed
        assert not directory.exists()
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)


def _worker(tmp_path, monkeypatch, source):
    worker = tmp_path / "finite_worker.py"
    worker.write_text(source, encoding="utf-8")
    monkeypatch.setattr(isolation, "_WORKER_PATH", worker)
    return worker


@pytest.mark.parametrize("extension", ["txt", "md", "csv", "pdf", "docx"])
def test_default_public_parser_crosses_real_process_boundary(extension, monkeypatch, tracked_workers):
    payload = _pdf() if extension == "pdf" else _docx() if extension == "docx" else b"A valid isolated text paragraph."

    def forbidden(*args, **kwargs):
        raise AssertionError("the application process must never parse uploads")

    for helper in ("_parse_text", "_parse_pdf", "_parse_docx", "_inspect_docx_archive"):
        monkeypatch.setattr(DocumentUploader, helper, forbidden)
    uploader = DocumentUploader()
    documents = uploader.parse_file(payload, f"valid.{extension}")
    assert documents[0].page_content.startswith("A valid isolated")
    receipt = uploader._last_parser_receipt
    assert receipt["pid"] == tracked_workers[0][0].pid != os.getpid()
    assert receipt["parent_pid"] == os.getpid()
    assert receipt["heavy_stack_loaded"] is False
    assert receipt["limits"]["RLIMIT_CPU"] == [8, 8]
    assert receipt["limits"]["RLIMIT_CORE"] == [0, 0]
    assert receipt["limits"]["RLIMIT_FSIZE"] == [16 * 1024 * 1024] * 2
    assert receipt["observed_peak_rss_bytes"] <= isolation.DEFAULT_PARSER_MEMORY_BYTES
    if isolation.sys.platform == "linux":
        assert receipt["limits"]["RLIMIT_AS"] == [512 * 1024 * 1024] * 2
    _assert_clean(tracked_workers)


@pytest.mark.parametrize("extension", ["pdf", "docx"])
def test_limits_and_clean_environment_precede_native_library_imports(extension, tmp_path, monkeypatch, tracked_workers):
    payload = _pdf() if extension == "pdf" else _docx()
    real_worker = isolation._WORKER_PATH
    marker = tmp_path / "imports.json"
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-do-not-inherit")
    _worker(tmp_path, monkeypatch, f'''
import builtins, json, os, resource, runpy, sys
worker = runpy.run_path({str(real_worker)!r})
original = builtins.__import__
seen = []
def guarded(name, *args, **kwargs):
    if name in {{"pymupdf", "docx", "zipfile"}}:
        assert resource.getrlimit(resource.RLIMIT_CPU) == (8, 8)
        assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
        assert resource.getrlimit(resource.RLIMIT_FSIZE) == (16777216, 16777216)
        if sys.platform == "linux":
            assert resource.getrlimit(resource.RLIMIT_AS) == (536870912, 536870912)
        assert "OPENAI_API_KEY" not in os.environ
        assert not any(key in sys.modules for key in ("torch", "chromadb"))
        seen.append(name)
        with open({str(marker)!r}, "w") as handle:
            json.dump(seen, handle)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
raise SystemExit(worker["main"]())
''')
    assert DocumentUploader().parse_file(payload, f"valid.{extension}")
    seen = json.loads(marker.read_text())
    assert ("pymupdf" if extension == "pdf" else "docx") in seen
    if extension == "docx":
        assert seen.index("zipfile") < seen.index("docx")
    _assert_clean(tracked_workers)


@pytest.mark.parametrize("scenario,expected", [
    ("timeout", "time limit"),
    ("crash", "stopped"),
    ("output", "output exceeded"),
    ("json", "invalid result"),
    ("pickle", "invalid result"),
])
def test_failure_kills_reaps_and_removes_owned_temporary_data(scenario, expected, tmp_path, monkeypatch, tracked_workers):
    marker = tmp_path / "started"
    action = {
        "timeout": "time.sleep(5)",
        "crash": "os._exit(17)",
        "output": "os.write(1, b'x' * 8192); time.sleep(5)",
        "json": "os.write(1, b'{not json')",
        "pickle": "os.write(1, bytes([128, 4, 99]) + b'os\\nsystem\\n.')",
    }[scenario]
    _worker(tmp_path, monkeypatch, f"import os, time\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('ready')\n{action}\n")
    uploader = DocumentUploader(parser_wall_seconds=0.6, parser_output_bytes=1024)
    started = time.monotonic()
    with pytest.raises(UploadValidationError, match=expected):
        uploader.parse_file(b"Finite synthetic worker input.", "private.txt")
    assert marker.exists()
    assert time.monotonic() - started < 3
    assert uploader.filename == ""
    _assert_clean(tracked_workers)


def test_real_finite_memory_allocation_is_killed_by_rss_monitor(tmp_path, monkeypatch, tracked_workers):
    marker = tmp_path / "allocated"
    _worker(tmp_path, monkeypatch, f'''
import time
from pathlib import Path
allocation = bytearray(64 * 1024 * 1024)
Path({str(marker)!r}).write_text("allocated")
time.sleep(5)
''')
    with pytest.raises(UploadValidationError, match="memory limit"):
        DocumentUploader(parser_memory_bytes=32 * 1024 * 1024, parser_wall_seconds=2).parse_file(b"Finite allocation probe.", "memory.txt")
    # Allocation is finite and at most 64 MiB; parent RSS monitoring is real.
    _assert_clean(tracked_workers)


def test_cpu_limit_really_terminates_worker_before_wall_deadline(tmp_path, monkeypatch, tracked_workers):
    real_worker = isolation._WORKER_PATH
    marker = tmp_path / "cpu-limited"
    _worker(tmp_path, monkeypatch, f'''
import runpy, sys
from pathlib import Path
worker = runpy.run_path({str(real_worker)!r})
worker["_apply_resource_limits"](int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
Path({str(marker)!r}).write_text("limits applied")
while True:
    pass
''')
    started = time.monotonic()
    with pytest.raises(UploadValidationError, match="stopped"):
        DocumentUploader(parser_cpu_seconds=1, parser_wall_seconds=5).parse_file(b"CPU bounded probe.", "cpu.txt")
    assert marker.exists()
    assert time.monotonic() - started < 4
    assert tracked_workers[0][0].returncode < 0
    _assert_clean(tracked_workers)


def test_admission_rejects_flood_without_spawning_or_queuing(tmp_path, monkeypatch, tracked_workers):
    release = tmp_path / "release"
    _worker(tmp_path, monkeypatch, f'''
import json, os, time
from pathlib import Path
request = json.load(__import__("sys").stdin)
Path({str(tmp_path)!r}, "ready-" + str(os.getpid())).write_text("ready")
deadline = time.monotonic() + 5
while not Path({str(release)!r}).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
print(json.dumps({{"version":1,"documents":[{{"page_content":"Bounded admitted worker text.","metadata":{{"source":request["filename"],"paragraph":0}}}}],"worker":{{"pid":os.getpid()}}}}))
''')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(DocumentUploader().parse_file, b"admitted", f"{index}.txt") for index in range(2)]
        try:
            deadline = time.monotonic() + 3
            while len(list(tmp_path.glob("ready-*"))) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(list(tmp_path.glob("ready-*"))) == 2
            assert len(tracked_workers) == isolation.MAX_CONCURRENT_PARSERS == 2
            started = time.monotonic()
            for _ in range(20):
                with pytest.raises(UploadValidationError, match="busy"):
                    DocumentUploader().parse_file(b"Rejected flood input.", "flood.txt")
            assert len(tracked_workers) == 2
            assert time.monotonic() - started < 0.5
        finally:
            release.touch()
        assert all(future.result(timeout=3) for future in futures)
    _assert_clean(tracked_workers)


def test_monitor_unavailability_fails_closed_and_releases_admission(monkeypatch, tracked_workers):
    real_reader = isolation._rss_reader

    def unavailable():
        raise isolation.ParserIsolationError("Document parser memory monitoring is unavailable.")

    monkeypatch.setattr(isolation, "_rss_reader", unavailable)
    for _ in range(3):
        with pytest.raises(UploadValidationError, match="unavailable"):
            DocumentUploader().parse_file(b"Must not parse in-process.", "safe.txt")
    assert not tracked_workers
    monkeypatch.setattr(isolation, "_rss_reader", real_reader)
    assert DocumentUploader().parse_file(b"Admission was released after failure.", "safe.txt")
    _assert_clean(tracked_workers)


def test_mid_run_monitor_failure_kills_and_reaps(tmp_path, monkeypatch, tracked_workers):
    _worker(tmp_path, monkeypatch, "import time\ntime.sleep(5)\n")
    monkeypatch.setattr(isolation, "_rss_reader", lambda: lambda pid: None)
    with pytest.raises(UploadValidationError, match="monitoring failed"):
        DocumentUploader().parse_file(b"Monitor failure probe.", "probe.txt")
    _assert_clean(tracked_workers)


@pytest.mark.parametrize("action", ["os._exit(19)", "time.sleep(5)"])
def test_group_signal_permission_race_still_reaps_owned_child(action, tmp_path, monkeypatch, tracked_workers):
    _worker(tmp_path, monkeypatch, f"import os, time\n{action}\n")

    def permission_race(*args):
        raise PermissionError("synthetic Darwin group transition")

    monkeypatch.setattr(isolation.os, "killpg", permission_race)
    with pytest.raises(UploadValidationError, match="stopped|time limit"):
        DocumentUploader(parser_wall_seconds=0.3).parse_file(b"Signal transition probe.", "probe.txt")
    _assert_clean(tracked_workers)


def test_closed_stdout_cannot_bypass_wall_timeout(tmp_path, monkeypatch, tracked_workers):
    _worker(tmp_path, monkeypatch, "import os, time\nos.close(1)\ntime.sleep(5)\n")
    with pytest.raises(UploadValidationError, match="time limit"):
        DocumentUploader(parser_wall_seconds=0.3).parse_file(b"Closed pipe probe.", "probe.txt")
    _assert_clean(tracked_workers)


def test_unsupported_platform_fails_closed(monkeypatch):
    monkeypatch.setattr(isolation.sys, "platform", "win32")
    with pytest.raises(UploadValidationError, match="unavailable on this platform"):
        isolation._rss_reader()


@pytest.mark.parametrize("attack", ["crc", "duplicate", "symlink", "encrypted", "truncated", "lying-size"])
def test_malformed_archive_rejected_in_real_worker(attack, tracked_workers):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("word/document.xml")
        if attack == "symlink":
            info.create_system = 3
            info.external_attr = (0o120777 << 16)
        archive.writestr(info, b"A finite document XML fixture.")
        if attack == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(info, b"duplicate")
    payload = bytearray(stream.getvalue())
    central = payload.index(b"PK\x01\x02")
    if attack == "crc":
        struct.pack_into("<I", payload, central + 16, 0)
    elif attack == "encrypted":
        struct.pack_into("<H", payload, central + 8, 1)
    elif attack == "truncated":
        payload = payload[:-10]
    elif attack == "lying-size":
        struct.pack_into("<I", payload, central + 24, 1)
    with pytest.raises(UploadValidationError, match="DOCX ZIP") as caught:
        DocumentUploader().parse_file(bytes(payload), "private-canary.docx")
    assert "private-canary" not in str(caught.value)
    assert "word/document.xml" not in str(caught.value)
    _assert_clean(tracked_workers)


def test_worker_error_and_invalid_metadata_never_echo_untrusted_diagnostics(tmp_path, monkeypatch):
    for response in (
        {"version": 1, "error": "secret-canary-path"},
        {"version": 1, "documents": [{"page_content": "safe text", "metadata": {"source": "secret-canary-path"}}], "worker": {}},
    ):
        _worker(tmp_path, monkeypatch, f"print({json.dumps(response)!r})\n")
        with pytest.raises(UploadValidationError, match="invalid result") as caught:
            DocumentUploader().parse_file(b"Generic error probe.", "input.txt")
        assert "secret-canary" not in str(caught.value)


def test_parent_rechecks_extraction_budget_on_worker_response(tmp_path, monkeypatch):
    response = {"version": 1, "documents": [{"page_content": "x" * 30, "metadata": {"source": "input.txt", "paragraph": 0}}], "worker": {}}
    _worker(tmp_path, monkeypatch, f"print({json.dumps(response)!r})\n")
    with pytest.raises(UploadValidationError, match="per-document character"):
        DocumentUploader(max_extracted_text_chars_per_document=20).parse_file(b"Bounded protocol probe.", "input.txt")


def test_encoded_input_limit_rejects_before_spawning(tracked_workers):
    with pytest.raises(UploadValidationError, match="too large"):
        DocumentUploader(max_bytes=4).parse_file(b"12345", "input.txt")
    assert not tracked_workers


@pytest.mark.parametrize("kwargs", [{"parser_wall_seconds": 0}, {"parser_cpu_seconds": 0.5}, {"parser_memory_bytes": -1}, {"parser_output_bytes": True}, {"max_pdf_pages": 1.5}])
def test_invalid_isolation_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DocumentUploader(**kwargs)


def test_valid_worker_output_is_still_bounded_by_document_count(tmp_path, monkeypatch):
    policy = dict(POLICY_DEFAULTS, max_parsed_documents=1)
    response = {"version": 1, "documents": [{"page_content": "bounded", "metadata": {"source": "input.txt", "paragraph": 0}}] * 2, "worker": {}}
    _worker(tmp_path, monkeypatch, f"print({json.dumps(response)!r})\n")
    with pytest.raises(UploadValidationError, match="invalid result"):
        isolation.parse_isolated(b"bounded", "input.txt", ".txt", policy, isolation.IsolationLimits())
