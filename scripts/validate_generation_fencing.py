"""Provider-free real-process generation fencing, using only owned temporary Redis.

PYTHON_DOTENV_DISABLED=1 .venv/bin/python scripts/validate_generation_fencing.py \
  --redis-server .venv/lifecycle-tools-t3-20261019/redis-7.2.16/src/redis-server \
  --output outputs/security-v4-fencing-new.json

Destinations are create-only. Events, barriers and pipe acknowledgments establish
ordering; there are no arbitrary synchronization sleeps. Redis is nonpersistent,
bound to loopback, PID-verified, and only this runner's child is terminated.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import uuid

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
for _name in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "OPENAI_API_KEY"):
    os.environ[_name] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CTX = mp.get_context("spawn")


class Embeddings:
    def embed_query(self, text):
        return [(byte - 127.5) / 128 for byte in hashlib.sha256(text.encode()).digest()]


def _worker(connection, options, release, barrier, epochs, scopes):
    from phase3.semantic_cache import GenerationFencedError, SemanticCache

    def identity():
        return ":".join(str(epochs[ord(key) - ord("a")]) for key in scopes)

    cache = SemanticCache(Embeddings(), redis_host="127.0.0.1", redis_failure_policy="miss",
                          similarity_threshold=0.99999, index_identity=identity(),
                          identity_provider=identity, **options)
    tokens = {}
    connection.send({"pid": os.getpid()})
    try:
        while True:
            command, args = connection.recv()
            if command == "exit":
                return
            try:
                if command == "begin":
                    tokens[args[0]] = cache.begin_generation()
                    result = tokens[args[0]].shared_epoch
                elif command == "publish":
                    label, query = args
                    result = cache.publish_generation(tokens[label], query, query, [query])
                elif command in {"generate_paused", "publish_paused", "publish_lost_reply", "clear_lost_reply"}:
                    label, query = args
                    if command == "generate_paused":
                        tokens[label] = cache.begin_generation()
                        connection.send({"stage": "generation-started"})
                        assert release.wait(15), "generation release timeout"
                        result = cache.publish_generation(tokens[label], query, query, [])
                    else:
                        original = cache._redis.eval

                        def intercept(script, *values):
                            if command == "publish_paused" and "fencing:publish" in script:
                                connection.send({"stage": "conditional-publication-ready"})
                                assert release.wait(15), "publication release timeout"
                            receipt = original(script, *values)
                            marker = "fencing:clear" if command == "clear_lost_reply" else "fencing:publish"
                            if command.endswith("lost_reply") and marker in script:
                                raise ConnectionError("injected lost acknowledgment AFTER Redis committed")
                            return receipt

                        cache._redis.eval = intercept
                        try:
                            result = (cache.clear() if command == "clear_lost_reply" else
                                      cache.publish_generation(tokens[label], query, query, []))
                        finally:
                            cache._redis.eval = original
                elif command == "clear_barrier":
                    barrier.wait(15)
                    result = cache.clear()
                elif command == "misuse":
                    label, field = args
                    cache.validate_generation(replace(tokens[label], **{field: "foreign-scope"}))
                    result = True
                elif command == "lookup_bound":
                    label, query = args
                    result = cache.lookup_generation(tokens[label], query)
                elif command == "local_entries":
                    result = len(cache._key_to_row)
                else:
                    result = getattr(cache, command)(*args)
                connection.send({"result": result})
            except GenerationFencedError:
                connection.send({"fenced": True})
            except Exception as exc:
                connection.send({"error": type(exc).__name__, "detail": str(exc)})
    finally:
        connection.close()


class Worker:
    def __init__(self, port, namespace, barrier, epochs, scopes="a"):
        self.connection, child = CTX.Pipe()
        self.release = CTX.Event()
        self.process = CTX.Process(target=_worker, args=(
            child, {"redis_port": port, "namespace": namespace}, self.release, barrier, epochs, scopes,
        ))
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
        if not self.connection.poll(25):
            raise TimeoutError(f"worker {self.process.pid} acknowledgment timeout")
        result = self.connection.recv()
        if "error" in result:
            raise RuntimeError(result)
        return result

    def call(self, command, *args):
        self.send(command, *args)
        return self.receive()

    def close(self):
        self.release.set()
        if self.process.is_alive():
            try:
                self.send("exit")
            except (BrokenPipeError, EOFError):
                pass
            self.process.join(5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
        self.connection.close()
        assert not self.process.is_alive()


class OwnedRedis:
    def __init__(self, binary, directory):
        self.binary, self.directory = str(Path(binary).resolve()), directory
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        self.process = None

    def start(self):
        import redis
        ready = threading.Event()
        self.process = subprocess.Popen([
            self.binary, "--bind", "127.0.0.1", "--port", str(self.port),
            "--save", "", "--appendonly", "no", "--dir", self.directory,
            "--protected-mode", "yes",
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def read_readiness(process):
            for line in process.stdout:
                if "Ready to accept connections" in line:
                    ready.set()

        self.reader = threading.Thread(target=read_readiness, args=(self.process,), daemon=True)
        self.reader.start()
        if not ready.wait(10):
            raise RuntimeError("owned Redis did not report readiness")
        client = redis.Redis(host="127.0.0.1", port=self.port,
                             socket_connect_timeout=0.5, socket_timeout=0.5)
        info = client.info("server")
        if info["process_id"] != self.process.pid:
            raise RuntimeError("Redis PID is not owned by this runner")
        self.version = info["redis_version"]

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(10)
            self.reader.join(5)
            self.process.stdout.close()


def validate(binary):
    workers, checks = [], []
    assertion_count = 0

    def check(condition, label):
        nonlocal assertion_count
        assertion_count += 1
        if not condition:
            raise AssertionError(label)

    with tempfile.TemporaryDirectory(prefix="rag-generation-fencing-") as directory:
        server = OwnedRedis(binary, directory)
        namespace = "fencing-validation-" + uuid.uuid4().hex
        barrier = CTX.Barrier(4)
        epochs = CTX.Array("i", [1, 1, 1])

        def worker(scope="default", indices="a"):
            item = Worker(server.port, namespace + ":" + scope, barrier, epochs, indices)
            workers.append(item)
            return item

        try:
            server.start()
            stale, clearer = worker(), worker()
            isolated = worker("isolated")
            check("result" in isolated.call("begin", "isolated"), "isolated start")
            check(isolated.call("publish", "isolated", "isolated").get("result") is True, "isolated publication")

            stale.send("generate_paused", "old", "revoked-generation")
            check(stale.receive().get("stage") == "generation-started", "generation captured")
            check(clearer.call("clear").get("result") is True, "clear acknowledged")
            stale.release.set()
            check(stale.receive().get("fenced") is True, "old response rejected")
            check(stale.call("lookup", "revoked-generation").get("result") is None, "old local hit absent")
            check(clearer.call("lookup", "revoked-generation").get("result") is None, "old shared hit absent")
            check(isolated.call("lookup", "isolated")["result"]["answer"] == "isolated", "isolated preserved")
            checks.append("clear-during-generation-and-namespace-isolation")

            check("result" in stale.call("begin", "race"), "race start")
            stale.release.clear()
            stale.send("publish_paused", "race", "read-then-clear-race")
            check(stale.receive().get("stage") == "conditional-publication-ready", "last read passed")
            check(clearer.call("clear").get("result") is True, "racing clear")
            stale.release.set()
            check(stale.receive().get("fenced") is True, "Lua compare refused stale publication")
            check(stale.call("local_entries").get("result") == 0, "no optimistic local insertion")
            checks.append("atomic-clear-between-last-python-read-and-lua-publication")

            check("result" in stale.call("begin", "one"), "normal start one")
            check("result" in clearer.call("begin", "two"), "normal start two")
            check(stale.call("publish", "one", "normal-one").get("result") is True, "normal store one")
            check(clearer.call("publish", "two", "normal-two").get("result") is True, "normal store two")
            check(stale.call("lookup_bound", "one", "normal-two")["result"]["answer"] == "normal-two", "normal stores preserve authority")
            checks.append("snapshot-version-independent-of-revocation-epoch")

            for field in ("instance", "namespace", "index_identity", "local_epoch"):
                check(stale.call("misuse", "one", field).get("fenced") is True, "scope misuse " + field)
            checks.append("instance-namespace-index-token-scope")

            overlaps = [worker(keys, keys) for keys in ("ab", "bc", "c")]
            for item in overlaps:
                check("result" in item.call("begin", "index"), "index capture")
            epochs[1] += 1
            for item in overlaps[:2]:
                check(item.call("publish", "index", "changed-index").get("fenced") is True, "overlap revoked")
            check(overlaps[2].call("publish", "index", "unrelated-index").get("result") is True, "unrelated survives")
            checks.append("overlapping-index-scopes-with-unrelated-work-preserved")

            concurrent = [worker() for _ in range(4)]
            check("result" in stale.call("begin", "concurrent"), "concurrent start")
            for item in concurrent:
                item.send("clear_barrier")
            for item in concurrent:
                check(item.receive().get("result") is True, "concurrent clear acknowledgment")
            check(stale.call("publish", "concurrent", "stale-worker").get("fenced") is True, "concurrent clears fence")
            check(clearer.call("lookup", "normal-one").get("result") is None, "concurrent clear empty")
            checks.append("four-barrier-synchronized-process-clears")

            check("result" in stale.call("begin", "uncertain"), "uncertain start")
            check(stale.call("publish_lost_reply", "uncertain", "committed-uncertain").get("fenced") is True, "uncertain never acknowledged")
            check(stale.call("local_entries").get("result") == 0, "uncertain has no local optimistic rows")
            check(clearer.call("lookup", "committed-uncertain")["result"]["answer"] == "committed-uncertain", "ambiguous may commit")
            check("result" in clearer.call("begin", "remote-uncertain"), "remote token before lost clear acknowledgment")
            check(stale.call("clear_lost_reply", "unused", "unused").get("result") is False, "clear uncertainty not success")
            check(clearer.call("publish", "remote-uncertain", "remote-after-clear").get("fenced") is True, "remote stale worker observes committed clear")
            check(stale.call("publish", "uncertain", "after-clear").get("fenced") is True, "server clear despite lost receipt fences")
            checks.append("lost-publication-and-clear-acknowledgments")

            old_epoch = stale.call("begin", "restart")["result"]
            server.stop()
            check(stale.call("publish", "restart", "outage").get("fenced") is True, "outage publish refused")
            check(clearer.call("begin", "outage").get("fenced") is True, "outage capture refused")
            check(clearer.call("clear").get("result") is False, "outage clear refused")
            check(stale.call("lookup", "committed-uncertain").get("result") is None, "outage never returns current hit")
            server.start()
            fresh = worker()
            new_epoch = fresh.call("begin", "restart")["result"]
            check(old_epoch != new_epoch, "random restart epoch avoids ABA")
            check(stale.call("publish", "restart", "before-restart").get("fenced") is True, "stale worker after restart fenced")
            check(fresh.call("publish", "restart", "after-restart").get("result") is True, "new worker after restart")
            check(stale.call("lookup", "after-restart")["result"]["answer"] == "after-restart", "stale worker reloads current")
            checks.append("outage-and-nonpersistent-restart-with-stale-process")
            return {"schema": "generation-fencing.validation/v1", "status": "passed",
                    "redis_version": server.version, "assertions": assertion_count,
                    "worker_count": len(workers), "distinct_worker_pids": len({w.pid for w in workers}),
                    "concurrent_clears": 4, "checks": checks,
                    "boundary": "Request-start token, Lua conditional publication, final response validation; clears ordered before final validation reject uncached responses too.",
                    "limitations": ["Cannot retract responses linearized before clear or cancel provider work.",
                                    "Legacy direct store and explicit legacy/local availability modes are outside this boundary.",
                                    "Index/config identity providers must cover every relevant scope; serving remains stop-serving/restart, not hot swap.",
                                    "Redis process is standalone; Redis Cluster Lua cross-slot operation is not supported."]}
        finally:
            for item in workers:
                item.close()
            server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-server", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new create-only evidence destination")
    if not args.output.parent.is_dir():
        parser.error("output parent must already exist")
    receipt = validate(args.redis_server)
    receipt["python_version"] = sys.version.split()[0]
    root = Path(__file__).resolve().parents[1]
    receipt["sources_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
            "phase3/semantic_cache.py", "phase3/cicd_pipeline.py", "api.py", "app.py",
            "serving.py", "scripts/validate_generation_fencing.py", "tests/test_generation_fencing.py",
        )
    }
    with args.output.open("x") as destination:
        json.dump(receipt, destination, indent=2)
        destination.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
