"""Opt-in, provider-free process lifecycle verification on temporary resources.

Run from the project root:
  PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/validate_live_lifecycle.py \
      --redis-server .venv/lifecycle-tools-t3-20261019/redis-7.2.16/src/redis-server
  LIVE_LIFECYCLE=1 REDIS_SERVER=<binary> PYTHON_DOTENV_DISABLED=1 \
      .venv/bin/python -m pytest -q tests/test_live_lifecycle.py

No existing Redis instance/store is accepted. The runner starts its own loopback
Redis, verifies its PID, kills only its own Popen child, and uses temporary stores.
No provider credentials, model downloads, FLUSHALL, or service shutdown commands.
Redis absence is an explicit skip; normal offline pytest never launches services.

Locally verified source: Redis 7.2.16, SHA256
960a8ec15e34ff40e57ff16837b26b33bd81f2da6d24497bb63de532a323a18e
Authoritative digest: https://github.com/redis/redis-hashes/blob/master/README
Tarball: https://download.redis.io/releases/redis-7.2.16.tar.gz
Build in a NEW ignored .venv tooling directory after verifying that digest:
  make -C redis-7.2.16 -j4 MALLOC=libc BUILD_TLS=no OPT=-O2 redis-server
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid

# Must precede ANY project import, including in spawned children.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
for _name in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"):
    os.environ[_name] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CTX = mp.get_context("spawn")


class DeterministicEmbeddings:
    """Test vectors, explicitly not evidence of semantic/model quality."""

    def embed_query(self, text):
        return [(byte - 127.5) / 128 for byte in hashlib.sha256(text.encode()).digest()]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def _worker(connection, kind, options):
    try:
        if kind == "cache":
            from phase3.semantic_cache import SemanticCache
            resource = SemanticCache(DeterministicEmbeddings(), redis_host="127.0.0.1",
                                     similarity_threshold=0.99999, **options)
        else:
            resource = None
        connection.send({"pid": os.getpid()})
        while True:
            command, args = connection.recv()
            if command == "exit":
                return
            try:
                if command == "burst":
                    start_at, prefix, count, clear_every = args
                    time.sleep(max(0, start_at - time.monotonic()))
                    for index in range(count):
                        if clear_every and index % clear_every == 0:
                            assert resource.clear()
                        question = f"{prefix}-{index}"
                        resource.store(question, question, [question])
                        resource.lookup(question)
                    result = count
                elif kind == "cache":
                    result = getattr(resource, command)(*args)
                elif command == "build":
                    from langchain_core.documents import Document
                    from module1.data_preprocessing import build_vector_store, build_lock_for
                    path, text = args
                    with build_lock_for(path, timeout=0) as lock:
                        resource = build_vector_store([Document(page_content=text)],
                                                      DeterministicEmbeddings(), path, lock=lock)
                    result = resource.get()["documents"]
                elif command == "read":
                    from module1.data_preprocessing import load_vector_store
                    resource = load_vector_store(DeterministicEmbeddings(), args[0])
                    result = resource.get()["documents"]
                elif command == "rebuild":
                    from langchain_core.documents import Document
                    from module1.data_preprocessing import rebuild_vector_store_safely
                    path, text = args
                    resource = rebuild_vector_store_safely(
                        [Document(page_content=text)], DeterministicEmbeddings(), path,
                    )
                    result = resource.get()["documents"]
                elif command == "recover":
                    from preprocess_full import _recover_previous_vector_store
                    _recover_previous_vector_store(args[0])
                    result = True
                elif command == "snapshot":
                    from module1.data_preprocessing import copy_vector_store
                    result = copy_vector_store(*args)
                else:
                    raise ValueError(command)
                connection.send({"result": result})
            except Exception as exc:  # report to parent; no silent child failures
                connection.send({"error": type(exc).__name__, "detail": str(exc)})
    finally:
        connection.close()


class Worker:
    def __init__(self, kind="cache", **options):
        self.connection, child = CTX.Pipe()
        self.process = CTX.Process(target=_worker, args=(child, kind, options))
        self.process.start()
        child.close()
        try:
            self.pid = self.receive()["pid"]
        except BaseException:
            self.close()
            raise

    def send(self, command, *args):
        self.connection.send((command, args))

    def receive(self):
        if not self.connection.poll(40):
            raise TimeoutError(f"worker {self.process.pid} did not respond")
        return self.connection.recv()

    def call(self, command, *args):
        self.send(command, *args)
        result = self.receive()
        if "error" in result:
            raise RuntimeError(result)
        return result["result"]

    def close(self):
        if self.process.is_alive():
            try:
                self.send("exit")
            except (BrokenPipeError, EOFError):
                pass
            self.process.join(10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(10)
        self.connection.close()
        assert not self.process.is_alive()


class OwnedRedis:
    def __init__(self, binary, directory):
        self.binary, self.directory = str(Path(binary).resolve()), str(directory)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        self.process = None
        self.version = None

    def start(self):
        import redis
        self.process = subprocess.Popen(
            [self.binary, "--bind", "127.0.0.1", "--port", str(self.port),
             "--save", "", "--appendonly", "no", "--dir", self.directory,
             "--protected-mode", "yes"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client = redis.Redis(host="127.0.0.1", port=self.port,
                             socket_timeout=0.2, socket_connect_timeout=0.2)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("owned Redis exited before readiness")
            try:
                info = client.info("server")
                if info["process_id"] != self.process.pid:
                    raise RuntimeError("port is not owned by this runner; refusing access")
                self.version = info["redis_version"]
                return
            except redis.ConnectionError:
                time.sleep(0.03)
        raise TimeoutError("owned Redis startup")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(10)


def validate_redis(binary):
    """Real Redis plus independent Python workers; raises on any failed invariant."""
    workers = []
    with tempfile.TemporaryDirectory(prefix="rag-live-redis-") as directory:
        server = OwnedRedis(binary, directory)
        namespace = "live-lifecycle-" + uuid.uuid4().hex

        def worker(**options):
            scope = options.pop("namespace", namespace)
            item = Worker(redis_port=server.port, namespace=scope, **options)
            workers.append(item)
            return item

        try:
            server.start()
            writer, reader = worker(), worker(redis_failure_policy="miss")
            isolated = worker(namespace=namespace + "-other")
            isolated.call("store", "question", "isolated", [])
            writer.call("store", "question", "old", ["old"])
            assert reader.call("lookup", "question")["answer"] == "old"
            writer.call("store", "question", "new", ["new"])
            assert reader.call("lookup", "question")["answer"] == "new"
            assert writer.call("clear") is True
            assert reader.call("lookup", "question") is None
            assert isolated.call("lookup", "question")["answer"] == "isolated"
            writer.call("store", "question", "revoked", [])
            assert reader.call("lookup", "question")["answer"] == "revoked"
            assert writer.call("lookup", "question")["answer"] == "revoked"
            server.stop()
            assert reader.call("lookup", "question") is None  # fail-closed policy
            assert writer.call("lookup", "question")["answer"] == "revoked"  # opt-in availability
            assert writer.call("clear") is False  # no false revocation receipt
            late = worker(redis_failure_policy="miss")  # initially unavailable
            assert late.call("lookup", "question") is None
            server.start()  # empty, persistence deliberately disabled
            # Recreate the same write count: random epochs must avoid counter ABA.
            fresh = worker()
            fresh.call("store", "question", "after-restart", [])
            assert reader.call("lookup", "question")["answer"] == "after-restart"
            assert late.call("lookup", "question")["answer"] == "after-restart"

            # Separate processes overlap write-through and reader refresh work.
            concurrent = [worker() for _ in range(4)]
            start_at = time.monotonic() + 0.2
            for index, item in enumerate(concurrent):
                item.send("burst", start_at, f"parallel-{index}", 12, 0)
            for item in concurrent:
                assert item.receive()["result"] == 12
            for index in range(4):
                for number in range(12):
                    question = f"parallel-{index}-{number}"
                    assert reader.call("lookup", question)["answer"] == question
            start_at = time.monotonic() + 0.2
            for index, item in enumerate(concurrent):
                item.send("burst", start_at, f"mixed-{index}", 12, 3)
            for _ in range(20):
                result = reader.call("lookup", "mixed-0-11")
                assert result is None or result["answer"] == "mixed-0-11"
            for item in concurrent:
                assert item.receive()["result"] == 12
            assert writer.call("clear") is True
            for item in [reader, *concurrent]:
                assert item.call("lookup", "parallel-0-0") is None
            final = worker(redis_failure_policy="miss")
            assert final.call("stats")["entries"] == 0
            return {"status": "passed", "redis_version": server.version,
                    "worker_count": len(workers), "distinct_worker_pids": len({w.pid for w in workers}),
                    "concurrent_writes": 96, "concurrent_clears": 16,
                    "checks": ["update", "clear", "namespace-isolation", "outage-miss",
                    "explicit-stale-local-policy", "failed-clear-receipt", "initial-outage-reconnect",
                    "restart-epoch", "concurrent-writers-readers", "post-clear-empty"]}
        finally:
            for item in workers:
                item.close()
            server.stop()


def _paused_publisher(connection, staged, final, fail_promotion):
    from preprocess_full import _publish_vector_store
    original = os.replace

    def replace(source, destination):
        source_path = os.path.realpath(os.fspath(source))
        final_path = os.path.realpath(final)
        staged_path = os.path.realpath(staged)
        if source_path == final_path:
            original(source, destination)
            connection.send("rename-gap-held")
            assert connection.poll(30), "publisher release timeout"
            assert connection.recv() == "release"
            return
        if source_path == staged_path and fail_promotion:
            raise OSError("injected promotion failure")
        original(source, destination)

    os.replace = replace
    try:
        _publish_vector_store(staged, final)
        connection.send("published")
    except OSError:
        connection.send("rolled-back")
    except BaseException as exc:
        connection.send(f"publisher-error:{type(exc).__name__}:{exc}")
    finally:
        connection.close()


def validate_index_readers():
    """Real ephemeral Chroma: active-reader refusal and gap/rollback barriers."""
    workers = []

    def worker():
        item = Worker("index")
        workers.append(item)
        return item

    with tempfile.TemporaryDirectory(prefix="rag-live-index-") as directory:
        root = Path(directory)
        final, staged, rollback = [str(root / name) for name in ("live", "stage", "rollback")]
        try:
            for path, text in [(final, "old corpus"), (staged, "new corpus"), (rollback, "bad corpus")]:
                builder = worker()
                assert builder.call("build", path, text) == [text]
                builder.close()  # release native writers before any renames
            from module1.data_preprocessing import BuildLockError
            from preprocess_full import _publish_vector_store
            readers = [worker(), worker()]
            for item in readers:
                assert item.call("read", final) == ["old corpus"]
            try:
                _publish_vector_store(staged, final)
            except BuildLockError:
                pass
            else:
                raise AssertionError("publication was allowed beneath active readers")
            for item in readers:
                assert item.call("read", final) == ["old corpus"]
                item.close()

            for stage, inject, expected in [(staged, False, "published"), (rollback, True, "rolled-back")]:
                parent, child = CTX.Pipe()
                process = CTX.Process(target=_paused_publisher, args=(child, stage, final, inject))
                process.start()
                child.close()
                try:
                    assert parent.poll(40), (
                        f"publisher produced no receipt (alive={process.is_alive()}, "
                        f"exitcode={process.exitcode})"
                    )
                    receipt = parent.recv()
                    assert receipt == "rename-gap-held", f"unexpected publisher receipt: {receipt!r}"
                    # Readers arriving during either rename gap fail BEFORE
                    # touching Chroma; no mixed corpus or missing-path load.
                    arrivals = [worker(), worker()]
                    for item in arrivals:
                        item.send("read", final)
                    for item in arrivals:
                        assert item.receive()["error"] == "BuildLockError"
                        item.close()
                    parent.send("release")
                    assert parent.poll(40) and parent.recv() == expected
                    process.join(10)
                    assert process.exitcode == 0
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join(10)
                    parent.close()
                check = worker()
                assert check.call("read", final) == ["new corpus"]
                check.close()
            return {"status": "passed", "active_readers": 2, "gap_arrivals_refused": 4,
                    "checks": ["active-readers-block-publication", "rename-gap-fail-closed",
                               "post-publication-new-corpus", "rollback-preserves-corpus"],
                    "restriction": "Participating POSIX local readers must exit before publication; no hot swap"}
        finally:
            for item in workers:
                item.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-server", default=os.environ.get("REDIS_SERVER") or shutil.which("redis-server"))
    parser.add_argument("--output", type=Path, help="Optional create-only JSON receipt")
    args = parser.parse_args()
    report = {"index": validate_index_readers()}
    report["redis"] = (validate_redis(args.redis_server) if args.redis_server else
                       {"status": "skipped", "reason": "redis-server unavailable; pass --redis-server PATH"})
    rendered = json.dumps(report, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
