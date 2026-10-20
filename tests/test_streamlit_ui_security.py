"""Focused regressions for Streamlit output trust boundaries and provider UI state."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _load_ui_helpers():
    """Compile the real app helpers without starting Streamlit or model resources."""
    tree = ast.parse(APP_PATH.read_text())
    names = {"_configured_provider", "_llm_available", "_render_untrusted_markdown"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)

    calls = []

    def markdown(body, **kwargs):
        calls.append((body, kwargs))

    env = {"Any": Any, "st": SimpleNamespace(markdown=markdown)}
    exec(compile(module, str(APP_PATH), "exec"), env)
    return env, calls


@pytest.mark.parametrize(
    "payload",
    [
        "<b>HTML</b>",
        "<script>alert('xss')</script>",
        '<div onmouseover="alert(1)">event handler</div>',
        '<img src="https://example.test/x" onerror="alert(1)">',
        "&lt;script&gt; &#x3C; &#x1F600; &amp;",
        "Unicode: café — 日本語 — 👩🏽‍💻",
        "**Markdown** _stays readable_ with `code` and [a link](https://example.test)",
    ],
)
def test_untrusted_render_helper_disables_html_for_all_payload_shapes(payload):
    env, calls = _load_ui_helpers()

    env["_render_untrusted_markdown"](payload)

    assert calls == [(payload, {"unsafe_allow_html": False})]


def test_untrusted_render_helper_uses_the_same_safe_flag_for_placeholders():
    env, _ = _load_ui_helpers()
    calls = []
    placeholder = SimpleNamespace(
        markdown=lambda body, **kwargs: calls.append((body, kwargs)),
    )

    env["_render_untrusted_markdown"]("<script>bad()</script>", placeholder)

    assert calls == [("<script>bad()</script>", {"unsafe_allow_html": False})]


@pytest.mark.parametrize(
    ("openrouter", "groq", "mistral", "label", "available"),
    [
        ("", "", "", "Extractive fallback", False),
        ("or-key", "", "", "OpenRouter", True),
        ("", "groq-key", "", "Groq (Llama 3.1)", True),
        ("", "", "mistral-key", "Mistral", True),
        ("or-key", "groq-key", "", "OpenRouter", True),
        ("or-key", "", "mistral-key", "OpenRouter", True),
        ("", "groq-key", "mistral-key", "Groq (Llama 3.1)", True),
        ("or-key", "groq-key", "mistral-key", "OpenRouter", True),
    ],
)
def test_ui_provider_state_matches_runtime_priority_without_exposing_keys(
    openrouter, groq, mistral, label, available,
):
    env, _ = _load_ui_helpers()

    assert env["_configured_provider"](openrouter, groq, mistral) == label
    assert env["_llm_available"](openrouter, groq, mistral) is available
    assert all(key not in label for key in (openrouter, groq, mistral) if key)


def test_app_surfaces_openrouter_as_presence_only_in_settings_and_controls():
    source = APP_PATH.read_text()

    assert '"OPENROUTER_API_KEY set": bool(OPENROUTER_API_KEY)' in source
    assert "_llm_available(OPENROUTER_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY)" in source
    assert "_configured_provider(OPENROUTER_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY)" in source
