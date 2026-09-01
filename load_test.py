#!/usr/bin/env python3
"""
Load test for the RAG API.
──────────────────────────
A dependency-free async load generator (httpx is already required) that reports the
numbers a panel or an SRE actually asks for: throughput, latency percentiles, and
error rate under concurrency.

It exercises only **LLM-free** architectures (naive / rerank), so a full run costs
nothing in API fees and the numbers reflect the serving stack — retrieval, the vector
store, the semantic cache and the async request path — rather than upstream LLM
latency.

Two scenarios:
  * cold   — unique queries, every request runs retrieval (worst case for the cache)
  * cached — one repeated query, so every request after the first is a cache hit

Usage:
  python load_test.py --url http://127.0.0.1:8080 --key KEY --concurrency 16 --requests 400
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Dict, List

import httpx

QUESTIONS = [
    "what is a corporation", "how does photosynthesis work",
    "who wrote hamlet", "what causes inflation", "how do vaccines work",
    "what is the capital of france", "define machine learning",
    "how tall is mount everest", "what is a black hole",
    "how does a combustion engine work", "what is compound interest",
    "who painted the mona lisa", "what is DNA", "how do tides work",
    "what is the speed of light", "explain the water cycle",
]


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


async def _worker(client, url, key, arch, use_cache, queries, latencies, errors, idx_counter):
    while True:
        i = idx_counter[0]
        idx_counter[0] += 1
        if i >= len(queries):
            return
        q = queries[i]
        started = time.perf_counter()
        try:
            r = await client.post(
                f"{url}/query",
                headers={"X-API-Key": key, "Content-Type": "application/json"},
                json={"question": q, "architecture": arch, "use_cache": use_cache},
            )
            elapsed = (time.perf_counter() - started) * 1000
            if r.status_code == 200:
                latencies.append(elapsed)
            else:
                errors.append(r.status_code)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc)[:40])


async def _run_scenario(url, key, arch, concurrency, n_requests, use_cache, unique) -> Dict:
    if unique:
        queries = [f"{QUESTIONS[i % len(QUESTIONS)]} variant {i}" for i in range(n_requests)]
    else:
        queries = ["what is a corporation"] * n_requests  # forces cache hits

    latencies: List[float] = []
    errors: List = []
    idx_counter = [0]

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        started = time.perf_counter()
        await asyncio.gather(*[
            _worker(client, url, key, arch, use_cache, queries, latencies, errors, idx_counter)
            for _ in range(concurrency)
        ])
        wall = time.perf_counter() - started

    ok = len(latencies)
    return {
        "arch": arch,
        "concurrency": concurrency,
        "requests": n_requests,
        "succeeded": ok,
        "errors": len(errors),
        "error_sample": errors[:5],
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(ok / wall, 1) if wall > 0 else 0.0,
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 1),
            "p90": round(_percentile(latencies, 0.90), 1),
            "p95": round(_percentile(latencies, 0.95), 1),
            "p99": round(_percentile(latencies, 0.99), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
            "mean": round(statistics.mean(latencies), 1) if latencies else 0.0,
        },
    }


async def _main_async(args) -> None:
    # Wait for readiness so warmup latency does not pollute the measurement.
    async with httpx.AsyncClient(timeout=10) as client:
        for _ in range(120):
            try:
                r = await client.get(f"{args.url}/readyz")
                if r.status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(5)
        else:
            print("Service never became ready; aborting.")
            return

    report = {"url": args.url, "scenarios": []}
    print("=" * 74)
    print(f"LOAD TEST — {args.url} — concurrency={args.concurrency}, requests={args.requests}")
    print("=" * 74)

    scenarios = [
        ("cached (repeated query, cache-hit path)", args.arch, False, True),
        ("cold (unique queries, retrieval path)", args.arch, True, False),
    ]
    for label, arch, unique, use_cache in scenarios:
        # Prime the cache for the cached scenario.
        if not unique:
            async with httpx.AsyncClient(timeout=120) as c:
                await c.post(f"{args.url}/query",
                             headers={"X-API-Key": args.key},
                             json={"question": "what is a corporation",
                                   "architecture": arch, "use_cache": True})
        result = await _run_scenario(
            args.url, args.key, arch, args.concurrency, args.requests,
            use_cache=use_cache, unique=unique,
        )
        result["scenario"] = label
        report["scenarios"].append(result)
        lat = result["latency_ms"]
        print(f"\n[{label}]  arch={arch}")
        print(f"  throughput : {result['throughput_rps']} req/s "
              f"({result['succeeded']}/{result['requests']} ok, {result['errors']} errors)")
        print(f"  latency ms : p50={lat['p50']}  p90={lat['p90']}  p95={lat['p95']}  "
              f"p99={lat['p99']}  max={lat['max']}")
        if result["error_sample"]:
            print(f"  errors     : {result['error_sample']}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default="")
    ap.add_argument("--arch", default="naive", help="LLM-free arch: naive | rerank | parent_child")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--out", default="outputs/load_test.json")
    asyncio.run(_main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
