"""
Tests for the production characteristics of the HTTP API.

These assert the behaviours that separate a service from a demo:

* auth **fails closed** — no configured key means protected routes are unavailable,
  not open (the previous implementation served them with only a log warning);
* readiness is distinct from liveness, so an orchestrator can tell "process is up"
  from "instance can serve";
* every response carries a correlation id;
* metrics are exposed for scraping;
* the architecture registry served over HTTP is the full framework registry.

The retrieval stack is stubbed, so no model, corpus or network is required.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _StubArch:
    """Minimal BaseRAG-shaped stub."""

    def __init__(self, name: str, llm_free: bool = True):
        self.name = name
        self.llm_free_retrieval = llm_free

    def run(self, query: str) -> Dict[str, Any]:
        return {
            "architecture": self.name, "query": query,
            "answer": f"answer to {query}",
            "context_docs": [], "contexts": ["ctx-1", "ctx-2"],
            "latency_ms": 12.5,
        }


class _StubCache:
    def __init__(self):
        self.stored: List[str] = []
        self.hit_next = False

    def lookup(self, query: str):
        if self.hit_next:
            return {
                "answer": "cached answer", "contexts": ["c"],
                "cache_hit": True, "similarity_score": 0.99, "latency_ms": 0.4,
            }
        return None

    def store(self, query, answer, contexts):
        self.stored.append(query)

    def stats(self):
        return {
            "hits": 1, "misses": 2, "total_queries": 3, "hit_rate": 0.33,
            "entries": 5, "capacity": 50000, "avg_lookup_ms": 0.5,
            "backend": "memory", "ann": True, "evictions": 0, "errors": 0,
            "total_latency_saved_ms": 0.0,
        }

    def clear(self):
        self.stored.clear()

    def healthy(self):
        return True


def _make_client(monkeypatch, api_key: str = "", allow_unauth: bool = False,
                 ready: bool = True) -> TestClient:
    """Import api.py with the given security posture and a stubbed stack."""
    monkeypatch.setenv("API_KEY", api_key)
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "true" if allow_unauth else "false")
    monkeypatch.setenv("LOG_FORMAT", "text")

    # Reload config + api so module-level constants pick up the env above.
    import configs.config as cfg
    importlib.reload(cfg)
    import serving
    importlib.reload(serving)
    import api as api_module
    importlib.reload(api_module)

    ctx = api_module.CONTEXT
    archs = {
        "naive": _StubArch("Naive RAG"),
        "rerank": _StubArch("Rerank RAG"),
        "grace_omega": _StubArch("GRACE-Ω", llm_free=False),
    }
    cache = _StubCache()
    monkeypatch.setattr(ctx, "architectures", lambda: archs)
    monkeypatch.setattr(ctx, "cache", lambda: cache)
    monkeypatch.setattr(ctx, "chunks", lambda: [])
    monkeypatch.setattr(ctx, "warmup", lambda: None)
    monkeypatch.setattr(ctx, "is_ready", lambda: ready)
    monkeypatch.setattr(
        ctx, "readiness",
        lambda: {"ready": ready, "reason": None if ready else "warming up",
                 "architectures": len(archs), "vector_store": True,
                 "cache": True, "uptime_seconds": 1.0},
    )
    client = TestClient(api_module.app)
    client._stub_cache = cache  # type: ignore[attr-defined]
    return client


# ── Auth fails closed ──────────────────────────────────────────────────────────

class TestAuthFailsClosed:
    def test_no_key_configured_blocks_protected_route(self, monkeypatch):
        """
        Regression: previously an empty API_KEY served every protected route and only
        logged a warning. It must now refuse.
        """
        with _make_client(monkeypatch, api_key="") as client:
            r = client.post("/query", json={"question": "hi", "architecture": "naive"})
            assert r.status_code == 503
            assert "API_KEY" in r.json()["detail"]

    def test_explicit_dev_optout_allows_access(self, monkeypatch):
        with _make_client(monkeypatch, api_key="", allow_unauth=True) as client:
            r = client.post("/query", json={"question": "hi", "architecture": "naive"})
            assert r.status_code == 200

    def test_wrong_key_is_rejected(self, monkeypatch):
        with _make_client(monkeypatch, api_key="a-sufficiently-long-key") as client:
            r = client.post(
                "/query", json={"question": "hi", "architecture": "naive"},
                headers={"X-API-Key": "wrong"},
            )
            assert r.status_code == 401

    def test_correct_key_is_accepted(self, monkeypatch):
        key = "a-sufficiently-long-key"
        with _make_client(monkeypatch, api_key=key) as client:
            r = client.post(
                "/query", json={"question": "hi", "architecture": "naive"},
                headers={"X-API-Key": key},
            )
            assert r.status_code == 200

    def test_missing_header_is_rejected(self, monkeypatch):
        with _make_client(monkeypatch, api_key="a-sufficiently-long-key") as client:
            r = client.post("/query", json={"question": "hi", "architecture": "naive"})
            assert r.status_code == 401

    def test_health_endpoints_stay_public(self, monkeypatch):
        """Probes must not require credentials or an orchestrator cannot use them."""
        with _make_client(monkeypatch, api_key="a-sufficiently-long-key") as client:
            assert client.get("/healthz").status_code == 200
            assert client.get("/readyz").status_code == 200
            assert client.get("/metrics").status_code == 200


# ── Liveness vs readiness ──────────────────────────────────────────────────────

class TestHealthProbes:
    def test_readyz_returns_503_when_not_ready(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True, ready=False) as client:
            r = client.get("/readyz")
            assert r.status_code == 503
            assert r.json()["ready"] is False

    def test_healthz_is_ok_even_when_not_ready(self, monkeypatch):
        """Liveness must stay green while warming, else orchestrators restart-loop."""
        with _make_client(monkeypatch, allow_unauth=True, ready=False) as client:
            assert client.get("/healthz").status_code == 200

    def test_work_is_refused_when_not_ready(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True, ready=False) as client:
            r = client.post("/query", json={"question": "hi", "architecture": "naive"})
            assert r.status_code == 503

    def test_healthz_reports_service_metadata(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            body = client.get("/healthz").json()
            assert body["status"] == "ok"
            assert body["service"] and body["version"]
            assert "auth_enabled" in body


# ── Correlation ids and metrics ────────────────────────────────────────────────

class TestObservability:
    def test_response_carries_request_id(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            r = client.get("/healthz")
            assert r.headers.get("X-Request-ID")

    def test_inbound_request_id_is_propagated(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            r = client.get("/healthz", headers={"X-Request-ID": "trace-me-123"})
            assert r.headers["X-Request-ID"] == "trace-me-123"

    def test_metrics_endpoint_exposes_counters(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            client.get("/healthz")
            body = client.get("/metrics").text
            assert "rag_http_requests_total" in body

    def test_query_response_includes_request_id(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            r = client.post("/query", json={"question": "hi", "architecture": "naive"})
            assert r.json()["request_id"]


# ── Full architecture registry (no more 4-of-12) ───────────────────────────────

class TestArchitectureRegistry:
    def test_advertises_every_servable_architecture(self, monkeypatch):
        """
        The endpoint reports the declared registry, not just what happens to be
        constructed, so lazily-built architectures are still discoverable — and
        listing them costs no LLM calls.
        """
        from module2.rag_architectures import ARCHITECTURE_KEYS

        with _make_client(monkeypatch, allow_unauth=True) as client:
            body = client.get("/architectures").json()
            assert set(body["architectures"]) == set(ARCHITECTURE_KEYS)
            assert body["count"] == len(ARCHITECTURE_KEYS)
            # Regression: the API used to expose only 4 of the 12 implemented.
            assert body["count"] >= 9

    def test_reports_llm_free_flag(self, monkeypatch):
        """Callers need to know which architectures cost API money."""
        with _make_client(monkeypatch, allow_unauth=True) as client:
            detail = client.get("/architectures").json()["detail"]
            assert detail["naive"]["llm_free_retrieval"] is True
            assert detail["rerank"]["llm_free_retrieval"] is True
            assert detail["grace_omega"]["llm_free_retrieval"] is False

    def test_reports_which_architectures_are_loaded(self, monkeypatch):
        """`loaded` tells a caller whether the first request pays construction cost."""
        with _make_client(monkeypatch, allow_unauth=True) as client:
            detail = client.get("/architectures").json()["detail"]
            assert detail["naive"]["loaded"] is True          # stubbed as eager
            assert detail["hybrid"]["loaded"] is False        # not built yet

    def test_unknown_architecture_is_400_not_500(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            r = client.post("/query", json={"question": "hi", "architecture": "nope"})
            assert r.status_code == 400
            assert "Unknown architecture" in r.json()["detail"]

    def test_any_registered_architecture_is_reachable(self, monkeypatch):
        """Regression: only 4 of 12 architectures used to be exposed."""
        with _make_client(monkeypatch, allow_unauth=True) as client:
            for name in ("naive", "rerank", "grace_omega"):
                r = client.post("/query", json={"question": "q", "architecture": name})
                assert r.status_code == 200, name


# ── Cache integration ──────────────────────────────────────────────────────────

class TestCacheEndpoints:
    def test_cache_hit_short_circuits_the_pipeline(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            client._stub_cache.hit_next = True
            body = client.post(
                "/query", json={"question": "hi", "architecture": "naive"}
            ).json()
            assert body["cache_hit"] is True
            assert body["answer"] == "cached answer"
            assert body["similarity_score"] == 0.99

    def test_miss_populates_the_cache(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            client.post("/query", json={"question": "fresh", "architecture": "naive"})
            assert "fresh" in client._stub_cache.stored

    def test_cache_can_be_bypassed(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            client._stub_cache.hit_next = True
            body = client.post(
                "/query",
                json={"question": "hi", "architecture": "naive", "use_cache": False},
            ).json()
            assert body["cache_hit"] is False

    def test_stats_expose_operational_fields(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            body = client.get("/cache/stats").json()
            for field in ("entries", "capacity", "avg_lookup_ms", "backend", "ann"):
                assert field in body


# ── Input validation ───────────────────────────────────────────────────────────

class TestInputValidation:
    def test_blank_question_rejected(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            r = client.post("/query", json={"question": "   ", "architecture": "naive"})
            assert r.status_code == 422

    def test_overlong_question_rejected(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            r = client.post(
                "/query", json={"question": "x" * 5000, "architecture": "naive"}
            )
            assert r.status_code == 422

    def test_ab_test_returns_both_variants(self, monkeypatch):
        with _make_client(monkeypatch, allow_unauth=True) as client:
            body = client.post(
                "/ab-test",
                json={"question": "q", "arch_a": "naive", "arch_b": "rerank"},
            ).json()
            assert body["variant_a"]["architecture"] == "naive"
            assert body["variant_b"]["architecture"] == "rerank"
            assert "delta_latency_ms" in body


# ── Config validation ──────────────────────────────────────────────────────────

class TestConfigValidation:
    def test_missing_api_key_is_reported(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "")
        monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "false")
        import configs.config as cfg
        importlib.reload(cfg)
        problems = cfg.validate_runtime_config()
        assert any("API_KEY" in p for p in problems)

    def test_short_api_key_is_reported(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "short")
        import configs.config as cfg
        importlib.reload(cfg)
        assert any("16 characters" in p for p in cfg.validate_runtime_config())

    def test_clean_config_has_no_problems(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "a-long-enough-production-key")
        monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com")
        import configs.config as cfg
        importlib.reload(cfg)
        assert cfg.validate_runtime_config() == []


@pytest.fixture(autouse=True)
def _restore_modules():
    """Reload config/serving/api after each test so env mutations don't leak."""
    yield
    for name in ("configs.config", "serving", "api"):
        if name in sys.modules:
            try:
                importlib.reload(sys.modules[name])
            except Exception:  # noqa: BLE001
                pass
