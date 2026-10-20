"""Offline HTTP access-control and private browser-resource regressions."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from security_boundary import session_cache, session_resource


@pytest.fixture
def secured(monkeypatch):
    import api
    monkeypatch.setattr(api, "API_KEY", "fixture-long-api-key")
    monkeypatch.setattr(api, "ALLOW_UNAUTHENTICATED", False)
    monkeypatch.setattr(api, "API_BIND_HOST", "127.0.0.1")
    monkeypatch.setattr(api.CONTEXT, "readiness", lambda: {
        "ready": False, "reason": "private/path credential=fixture-secret",
    })
    return api, TestClient(api.app, client=("127.0.0.1", 50000))


def test_every_nonpublic_route_fails_closed_before_handler(secured):
    api, client = secured
    public = {"/", "/health", "/healthz", "/readyz"}
    for route in api.app.routes:
        if route.path in public:
            continue
        for method in route.methods - {"HEAD", "OPTIONS"}:
            response = client.request(method, route.path)
            assert response.status_code == 401, (method, route.path, response.text)
    api.API_KEY = ""  # fixture monkeypatch restores original value
    for route in api.app.routes:
        if route.path not in public:
            for method in route.methods - {"HEAD", "OPTIONS"}:
                assert client.request(method, route.path).status_code == 503


@pytest.mark.parametrize("credential", [b"", b"wrong", b"\xff", b" \t", b"a\x00b", b"x" * 4097])
def test_malformed_credentials_rejected_at_http_boundary(secured, credential):
    _, client = secured
    response = client.post("/query", content=b"{invalid json", headers=[(b"X-API-Key", credential)])
    assert response.status_code == 401
    assert "fixture" not in response.text


def test_duplicate_credentials_are_not_accepted(secured):
    _, client = secured
    response = client.get("/cache/stats", headers=[
        ("X-API-Key", "fixture-long-api-key"), ("X-API-Key", "fixture-long-api-key"),
    ])
    assert response.status_code == 401


@pytest.mark.parametrize("key", ["é", "\ud800", "a\r\nb", "a b", "x" * 4097])
def test_direct_auth_helper_handles_unicode_and_malformed_input(secured, key):
    api, _ = secured
    with pytest.raises(HTTPException) as caught:
        api.require_api_key(key)
    assert caught.value.status_code == 401


@pytest.mark.parametrize("key", [" ", "fixture\nkey", "é", "\ud800", "x" * 4097])
def test_malformed_server_key_fails_closed_without_echoing_value(secured, monkeypatch, key):
    api, client = secured
    monkeypatch.setattr(api, "API_KEY", key)
    monkeypatch.setattr(api, "ALLOW_UNAUTHENTICATED", True)
    response = client.get("/cache/stats")
    assert response.status_code == 503
    assert response.json() == {"detail": "API_KEY configuration is invalid."}
    assert client.get("/healthz").status_code == 200
    assert client.get("/healthz").json()["auth_enabled"] is False
    assert client.get("/readyz").status_code == 503


@pytest.mark.parametrize("host", ["", "0.0.0.0", "::", "localhost", "127.0.0.1.evil", "127.0.0.1:8000", "127.0.0.1\n"])
def test_dev_override_rejects_unsafe_or_malformed_bind_host(secured, monkeypatch, host):
    api, client = secured
    monkeypatch.setattr(api, "API_KEY", "")
    monkeypatch.setattr(api, "ALLOW_UNAUTHENTICATED", True)
    monkeypatch.setattr(api, "API_BIND_HOST", host)
    assert client.get("/cache/stats").status_code == 503
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 503


@pytest.mark.parametrize("peer,headers", [
    ("203.0.113.1", {}), ("testclient", {}),
    ("127.0.0.1", {"Forwarded": "for=127.0.0.1"}),
    ("127.0.0.1", {"X-Forwarded-For": "127.0.0.1"}),
])
def test_local_dev_override_rejects_remote_or_proxied_requests(secured, monkeypatch, peer, headers):
    api, _ = secured
    monkeypatch.setattr(api, "API_KEY", "")
    monkeypatch.setattr(api, "ALLOW_UNAUTHENTICATED", True)
    client = TestClient(api.app, client=(peer, 50000))
    assert client.get("/cache/stats", headers=headers).status_code == 403


def test_probe_and_not_ready_errors_do_not_disclose_internal_diagnostics(secured, monkeypatch):
    api, client = secured
    assert client.get("/healthz").status_code == 200
    response = client.get("/readyz")
    assert response.status_code == 503
    assert "private" not in response.text and "secret" not in response.text
    monkeypatch.setattr(api.CONTEXT, "is_ready", lambda: False)
    with pytest.raises(HTTPException) as caught:
        api.require_ready()
    assert "private" not in caught.value.detail


def test_unhandled_response_and_request_id_are_sanitized(secured, monkeypatch):
    api, client = secured
    monkeypatch.setattr(api.CONTEXT, "refresh_cache_gauge", lambda: (_ for _ in ()).throw(
        RuntimeError("private/path credential=fixture-secret")))
    response = client.get("/metrics", headers={
        "X-API-Key": "fixture-long-api-key", "X-Request-ID": "x" * 1000,
    })
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert len(response.headers["X-Request-ID"]) <= 128


def _app_functions(*names):
    """Execute real resource functions without starting the UI or model stack."""
    tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    assert not any(node.decorator_list for node in functions)
    return compile(ast.Module(body=functions, type_ignores=[]), "app.py", "exec")


def test_actual_ui_resource_functions_keep_cache_and_uploads_in_session(monkeypatch):
    import chromadb
    from chromadb.config import Settings
    import module5.document_upload as uploads
    from phase3.semantic_cache import SemanticCache
    from langchain_core.embeddings import Embeddings

    class Embedder(Embeddings):
        def embed_query(self, text):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

    # No live Redis; actual semantic cache indexes and actual ephemeral upload stores.
    monkeypatch.setattr(SemanticCache, "_connect_redis", lambda *args: None)
    # Exercise private in-process session separation independently of Redis policy.
    real_cache_init = SemanticCache.__init__
    def local_cache_init(self, *args, **kwargs):
        assert kwargs["redis_failure_policy"] == "miss"
        kwargs["redis_failure_policy"] = "local"
        real_cache_init(self, *args, **kwargs)
    monkeypatch.setattr(SemanticCache, "__init__", local_cache_init)
    from serving import CONTEXT
    monkeypatch.setattr(CONTEXT, "_cache_namespace", lambda key: "fixture-index:" + key)
    # Match the explicit no-telemetry settings of the other real ephemeral tests.
    # Chroma shares one ephemeral system per process, including its settings.
    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
    real_chroma = uploads.Chroma
    monkeypatch.setattr(uploads, "Chroma", lambda **kwargs: real_chroma(client=client, **kwargs))
    st = SimpleNamespace(session_state={})
    env = {"st": st, "session_cache": session_cache, "session_resource": session_resource,
           "_get_embedding_model": Embedder}
    exec(_app_functions("_get_cache", "_get_uploader"), env)
    alice_state = st.session_state
    alice_cache, alice = env["_get_cache"](), env["_get_uploader"]()
    assert isinstance(alice, uploads.DocumentUploader)
    alice.index_documents(b"Alice private note", "same.txt")
    alice_cache.store("same query", "alice secret", ["alice context"])
    assert env["_get_cache"]() is alice_cache
    assert env["_get_uploader"]() is alice

    st.session_state = {}
    bob_cache, bob = env["_get_cache"](), env["_get_uploader"]()
    try:
        assert bob_cache.lookup("same query") is None
        assert bob is not alice and not bob.is_ready
        bob.index_documents(b"Bob separate note", "same.txt")
        assert {doc.page_content for doc in bob.search("same query")} == {"Bob separate note"}
        assert {doc.page_content for doc in alice.search("same query")} == {"Alice private note"}
        assert env["_get_cache"]("rerank").lookup("same query") is None
        bob_cache.store("same query", "bob secret", [])
        bob_cache.clear()
        st.session_state = alice_state
        assert env["_get_cache"]().lookup("same query")["answer"] == "alice secret"
    finally:
        alice.clear()
        bob.clear()


def test_api_does_not_report_failed_shared_clear_as_success(secured, monkeypatch):
    api, client = secured
    monkeypatch.setattr(api.CONTEXT, "available_keys", lambda: ["naive"])
    monkeypatch.setattr(api.CONTEXT, "cache", lambda architecture: SimpleNamespace(clear=lambda: False))
    response = client.delete("/cache", headers={"X-API-Key": "fixture-long-api-key"})
    assert response.status_code == 503
    assert "could not be confirmed" in response.json()["detail"]
