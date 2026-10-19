#!/usr/bin/env python3
"""Bounded local HTTP + cache workload. No paid calls, live Redis, or data edits."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
for key in ("GROQ_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
    os.environ[key] = ""
for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import httpx  # noqa: E402
import numpy as np  # noqa: E402


def summary(values) -> dict:
    return {"n": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99))}


def cache_workload() -> list[dict]:
    from phase3.semantic_cache import SemanticCache

    class Embedder:
        def embed_query(self, text):
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            return np.random.default_rng(seed).normal(size=32).tolist()

    rows = []
    for size in (100, 1000, 5000):
        cache = SemanticCache(Embedder(), redis_port=1, warm_start=False, max_entries=size,
                              namespace="validation-local-only", redis_failure_policy="local")
        # This microbenchmark explicitly measures only the local availability policy.
        # HTTP serving below instead measures fail-closed misses while Redis is down.
        cache._redis = None
        cache._use_redis = False
        start = time.perf_counter()
        for i in range(size):
            cache.store(f"fixed-query-{i}", "fixture answer", ["fixture context"])
        insert_seconds = time.perf_counter() - start
        times = []
        for i in range(100):
            start = time.perf_counter()
            cache.lookup(f"fixed-query-{i % size}")
            times.append((time.perf_counter() - start) * 1000)
        rows.append({"entries": size, "dimensions": 32, "synthetic_embedder": True,
                     "insertion_seconds": insert_seconds, **summary(times),
                     "stats": cache.stats()})
    return rows


def profile_stages(store_path: Path) -> dict:
    """Time the actual first-stage and cross-encoder paths separately, n=10."""
    from module1.data_preprocessing import (
        get_embedding_model, load_vector_store, chunk_documents, load_msmarco_documents,
    )
    from module2.rag_architectures import NaiveRAG, RerankRAG, _reranker

    questions = ["What is a corporation?", "What is deep learning?",
                 "What is the capital of France?", "How long does a chicken bake?",
                 "What is a bank account?"] * 2
    with tempfile.TemporaryDirectory(prefix="stage-validation-", dir=ROOT / "data") as temp:
        copied = Path(temp) / "store"
        shutil.copytree(store_path, copied)
        started = time.perf_counter()
        embedder = get_embedding_model()
        store = load_vector_store(embedder, str(copied))
        rerank = RerankRAG(store, chunk_documents(load_msmarco_documents()))
        naive = NaiveRAG(store)
        if not _reranker().is_available():
            raise RuntimeError("Real reranker unavailable; no fallback profile allowed")
        initialization = time.perf_counter() - started
        rows = []
        for question in questions:
            start = time.perf_counter()
            naive.retrieve(question)
            naive_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            candidates = rerank._first_stage(question)
            stage_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            _reranker().rerank(question, candidates, rerank.top_k)
            encoder_ms = (time.perf_counter() - start) * 1000
            rows.append({"question": question, "candidate_count": len(candidates),
                         "dense_top5_ms": naive_ms, "first_stage_ms": stage_ms,
                         "cross_encoder_ms": encoder_ms})
        return {
            "n": len(rows), "initialization_seconds": initialization,
            "device": _reranker().device, "individual": rows,
            "summaries": {field: summary([row[field] for row in rows]) for field in
                          ("dense_top5_ms", "first_stage_ms", "cross_encoder_ms")},
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                1024 ** 2 if sys.platform == "darwin" else 1024
            ), "provider_calls": 0,
            "scope": "One fresh process, cached model files, five fixed queries repeated twice; not production capacity",
        }


async def batch(base, headers, questions, concurrency, use_cache, architecture="naive"):
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base, headers=headers, timeout=90) as client:
        async def one(question):
            async with semaphore:
                start = time.perf_counter()
                response = await client.post("/query", json={
                    "question": question, "architecture": architecture,
                    "use_cache": use_cache,
                })
                elapsed = (time.perf_counter() - start) * 1000
                return {"status": response.status_code, "elapsed_ms": elapsed,
                        "cache_hit": response.json().get("cache_hit", False)}
        start = time.perf_counter()
        results = await asyncio.gather(*(one(q) for q in questions))
        seconds = time.perf_counter() - start
    return {"concurrency": concurrency, "architecture": architecture,
            "use_cache": use_cache, "requests": len(results),
            "successes": sum(r["status"] == 200 for r in results),
            "errors": sum(r["status"] != 200 for r in results),
            "cache_hits": sum(r["cache_hit"] for r in results),
            "wall_seconds": seconds, "requests_per_second": len(results) / seconds,
            **summary([r["elapsed_ms"] for r in results]), "individual": results}


def run(args):
    out = ROOT / args.out
    if out.exists() or out.with_suffix(".log").exists():
        raise FileExistsError("Choose a new --out; existing receipts are never overwritten")
    questions = ["What is a corporation?", "What is deep learning?",
                 "What is the capital of France?", "How long does a chicken bake?",
                 "What is a bank account?"]
    result = {"runtime": platform.python_version(), "platform": platform.platform(),
              "generator": "extractive (credentials blanked)", "provider_calls": 0,
              "cost_per_query_usd": None, "questions": questions,
              "max_concurrent_requests": 2, "rate_limit_for_workload": 100000,
              "http_cache_scope": "Redis unavailable; fail-closed misses, not cached-hit latency",
              "cache_microbenchmark": cache_workload(), "batches": []}
    with tempfile.TemporaryDirectory(prefix="serving-validation-", dir=ROOT / "data") as temp:
        copied = Path(temp) / "store"
        shutil.copytree(ROOT / args.store, copied)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        key = secrets.token_urlsafe(32)
        env = {**os.environ, "API_KEY": key, "ALLOW_UNAUTHENTICATED": "false",
               "VECTOR_STORE_DIR": str(copied), "REDIS_PORT": "1",
               "MAX_CONCURRENT_REQUESTS": "2", "RATE_LIMIT_PER_MINUTE": "100000"}
        with out.with_suffix(".log").open("x") as log:
            started = time.perf_counter()
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1",
                "--port", str(port), "--log-level", "warning",
            ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                first_liveness = None
                with httpx.Client(base_url=base, timeout=5) as client:
                    while time.perf_counter() - started < 120:
                        if process.poll() is not None:
                            raise RuntimeError("API exited; inspect local workload log")
                        try:
                            if client.get("/healthz").status_code == 200:
                                first_liveness = first_liveness or time.perf_counter() - started
                            if client.get("/readyz").status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(.1)
                    else:
                        raise TimeoutError("Readiness deadline exceeded")
                    result["process_start_to_liveness_seconds"] = first_liveness
                    result["process_start_to_ready_seconds"] = time.perf_counter() - started
                    result["rss_after_ready_kib"] = int(subprocess.check_output(
                        ["ps", "-p", str(process.pid), "-o", "rss="], text=True
                    ).strip())
                    result["unauthenticated_query_status"] = client.post(
                        "/query", json={"question": "q"}
                    ).status_code
                    headers = {"X-API-Key": key}
                    for concurrency in (1, 2, 4):
                        result["batches"].append(asyncio.run(batch(
                            base, headers, questions * 4, concurrency, False
                        )))
                    # Request repeats, but record actual hits: serving must miss during
                    # Redis outage. Do not label this as a successful cached-hit test.
                    asyncio.run(batch(base, headers, [questions[0]], 1, True))
                    for concurrency in (1, 2, 4):
                        result["batches"].append(asyncio.run(batch(
                            base, headers, [questions[0]] * 20, concurrency, True
                        )))
                    result["cache_stats"] = client.get("/cache/stats", headers=headers).json()
                    result["batches"].append(asyncio.run(batch(
                        base, headers, questions * 2, 2, False, "rerank"
                    )))
                    result["rss_after_workload_kib"] = int(subprocess.check_output(
                        ["ps", "-p", str(process.pid), "-o", "rss="], text=True
                    ).strip())
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                result["server_returncode_after_termination"] = process.returncode
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in {"batches", "cache_microbenchmark"}}, indent=2))
    for row in result["batches"]:
        print(json.dumps({k: v for k, v in row.items() if k != "individual"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/serving-validation-20261019.json")
    parser.add_argument("--store", default="data/chroma_db_full")
    parser.add_argument("--stages-only", action="store_true")
    args = parser.parse_args()
    if args.stages_only:
        result = profile_stages(ROOT / args.store)
        with (ROOT / args.out).open("x") as output:
            json.dump(result, output, indent=2)
        print(json.dumps(result, indent=2))
    else:
        run(args)
