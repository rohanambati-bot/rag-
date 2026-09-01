"""Security tests for the FastAPI surface."""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_auth(monkeypatch):
    """API client where API_KEY auth is enforced."""
    monkeypatch.setenv("API_KEY", "test-secret-key")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:8501")
    # Import after env is set so module-level config picks it up.
    import configs.config
    importlib.reload(configs.config)
    import api as api_mod
    importlib.reload(api_mod)
    return TestClient(api_mod.app)


@pytest.fixture
def client_open(monkeypatch):
    """
    API client in *explicit* dev mode.

    Auth now fails closed, so an unset ``API_KEY`` no longer means "serve everything"
    — it means "refuse protected routes". Opting out is deliberate via
    ``ALLOW_UNAUTHENTICATED=true``, which is what this fixture does so the input
    validation assertions below still reach the handlers.

    The retrieval stack is stubbed because these tests are about request validation,
    not retrieval, and building the real stack needs a persisted vector store.
    """
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "true")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:8501")
    import configs.config
    importlib.reload(configs.config)
    import serving
    importlib.reload(serving)
    import api as api_mod
    importlib.reload(api_mod)

    class _Arch:
        name = "Naive RAG"
        llm_free_retrieval = True

        def run(self, q):
            return {"answer": "a", "contexts": [], "latency_ms": 1.0}

    monkeypatch.setattr(api_mod.CONTEXT, "architectures", lambda: {"naive": _Arch()})
    monkeypatch.setattr(api_mod.CONTEXT, "is_ready", lambda: True)
    return TestClient(api_mod.app)


class TestApiAuth:
    def test_health_is_public(self, client_with_auth):
        r = client_with_auth.get("/health")
        assert r.status_code == 200
        assert r.json()["auth_enabled"] is True

    def test_query_rejects_missing_api_key(self, client_with_auth):
        r = client_with_auth.post(
            "/query",
            json={"question": "What is the capital of France?"},
        )
        assert r.status_code == 401

    def test_query_rejects_wrong_api_key(self, client_with_auth):
        r = client_with_auth.post(
            "/query",
            json={"question": "What is the capital of France?"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401

    def test_open_mode_health_reports_auth_disabled(self, client_open):
        r = client_open.get("/health")
        assert r.status_code == 200
        assert r.json()["auth_enabled"] is False


class TestInputValidation:
    def test_query_rejects_empty_question(self, client_open):
        r = client_open.post("/query", json={"question": ""})
        assert r.status_code == 422

    def test_query_rejects_oversized_question(self, client_open):
        r = client_open.post("/query", json={"question": "x" * 100_000})
        assert r.status_code == 422

    def test_query_rejects_unknown_architecture(self, client_open):
        # 400 rather than 422: the valid set is now the live architecture registry
        # rather than a hardcoded enum, so this is checked in the handler and the
        # error message lists what is actually available.
        r = client_open.post(
            "/query",
            json={"question": "hi", "architecture": "magic"},
        )
        assert r.status_code == 400
        assert "Unknown architecture" in r.json()["detail"]

    def test_evaluate_caps_samples(self, client_open):
        r = client_open.post("/evaluate", json={"samples": 100_000})
        assert r.status_code == 422
