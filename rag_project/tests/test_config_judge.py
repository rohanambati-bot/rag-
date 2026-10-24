"""Example tests for the corrected judge-model defaults (BUG 1).

Task 2.2 of the ``eval-pipeline-fixes`` spec.
Requirements: 2.1 (judge defaults to a live, cross-family Groq model) and 3.1
(the judge stays decoupled from the generator by using a different model family).

Design.md → Testing Strategy → BUG 1 (Example):
    Assert ``config.JUDGE_MODEL == "openai/gpt-oss-20b"``,
    ``config.GROQ_MODEL == "llama-3.1-8b-instant"``, ``JUDGE_MODEL != GROQ_MODEL``,
    and that ``.env.example`` carries ``JUDGE_PROVIDER=groq`` /
    ``JUDGE_MODEL=openai/gpt-oss-20b``.

Determinism note
----------------
The three values under test are resolved as ``os.getenv(NAME, DEFAULT)`` at import
time in ``configs/config.py``, and that module calls ``load_dotenv()`` on the
project-root ``.env`` before reading them. A real shell variable or a real ``.env``
file could therefore override the committed DEFAULTS and make a naive import-and-read
assertion environment-dependent. To keep the check deterministic we verify the
defaults two independent ways:

* ``test_config_defaults_resolve_to_expected_values`` — the runtime path. It pops the
  three variables from ``os.environ``, neutralises ``load_dotenv`` to a no-op (so a
  project ``.env`` cannot inject values), then ``importlib.reload``\\s ``configs.config``
  so the module-level constants re-resolve to their *defaults*. Original environment
  and ``load_dotenv`` are restored afterward.
* ``test_config_source_defaults_are_correct`` — a static AST parse of
  ``configs/config.py`` that reads the DEFAULT string literal of each
  ``os.getenv(NAME, DEFAULT)`` call directly. This never imports anything and holds
  even where ``python-dotenv`` is not installed (the Python 3.9 analysis machine).

Both are pure/offline: ``configs.config`` only imports ``os`` and ``python-dotenv`` at
module level (the ``torch`` probe is guarded inside a function), so no LangChain / Chroma
/ LLM stack and no network access are required.
"""
from __future__ import annotations

import ast
import importlib
import os
import sys
import types

# Project root is the parent of this ``tests/`` directory; resolving from ``__file__``
# keeps the tests independent of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.py")
ENV_EXAMPLE_PATH = os.path.join(PROJECT_ROOT, ".env.example")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Expected committed defaults (design.md → BUG 1).
EXPECTED_JUDGE_MODEL = "openai/gpt-oss-20b"
EXPECTED_GROQ_MODEL = "llama-3.1-8b-instant"
EXPECTED_JUDGE_PROVIDER = "groq"

# Variables that could override the defaults and must be cleared for the reload check.
_OVERRIDE_VARS = ("JUDGE_MODEL", "GROQ_MODEL", "JUDGE_PROVIDER")


def _ensure_dotenv_importable() -> None:
    """``configs.config`` does ``from dotenv import load_dotenv`` at import time.

    Install a minimal no-op stub only when ``python-dotenv`` is genuinely absent, so
    the module can be (re)imported on a runtime that lacks the package. When the real
    package is installed this is a no-op and the real ``load_dotenv`` is used (and then
    neutralised per-test below).
    """
    try:  # pragma: no cover - depends on the runtime
        import dotenv  # noqa: F401
    except Exception:
        mod = types.ModuleType("dotenv")
        mod.load_dotenv = lambda *a, **k: False  # mirrors the "no .env present" case
        sys.modules["dotenv"] = mod


_ensure_dotenv_importable()


def _reload_config():
    """Import (or reload) ``configs.config`` so its constants re-resolve now."""
    if "configs.config" in sys.modules:
        return importlib.reload(sys.modules["configs.config"])
    import configs.config as config  # noqa: WPS433 (local import is intentional)

    return config


def _getenv_defaults(source_path: str) -> dict:
    """Map ENV_NAME -> DEFAULT literal for every ``os.getenv(NAME, DEFAULT)`` call.

    Parses the source with ``ast`` (no import/execution) and keeps only calls whose
    first two arguments are string literals — exactly the module-level config defaults.
    """
    with open(source_path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=source_path)

    defaults: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_getenv = (
            isinstance(func, ast.Attribute) and func.attr == "getenv"
        ) or (isinstance(func, ast.Name) and func.id == "getenv")
        if not is_getenv or len(node.args) < 2:
            continue
        name_node, default_node = node.args[0], node.args[1]
        if (
            isinstance(name_node, ast.Constant)
            and isinstance(name_node.value, str)
            and isinstance(default_node, ast.Constant)
            and isinstance(default_node.value, str)
        ):
            defaults[name_node.value] = default_node.value
    return defaults


def _parse_env_file(path: str) -> dict:
    """Parse ``KEY=VALUE`` lines from a dotenv-style file into a dict.

    Blank lines and full-line comments are skipped; a trailing inline ``#`` comment and
    surrounding whitespace around the key and value are stripped, so the parsed value is
    the bare ``key=value`` token even when the line carries extra whitespace/comments.
    """
    values: dict = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.split("#", 1)[0].strip()  # drop inline comment, if any
            values[key.strip()] = val
    return values


# ─────────────────────────────────────────────────────────────────────────────
# 1. Runtime path — reload the real module with defaults isolated.
# ─────────────────────────────────────────────────────────────────────────────
def test_config_defaults_resolve_to_expected_values():
    """The DEFAULT resolved config values match the corrected judge/generator pair.

    Requirements: 2.1, 3.1.
    """
    import dotenv

    saved_environ = dict(os.environ)
    saved_load_dotenv = dotenv.load_dotenv
    try:
        # Neutralise .env loading so a real project-root .env cannot inject overrides.
        dotenv.load_dotenv = lambda *a, **k: False
        # Remove any shell/session overrides so os.getenv falls through to the defaults.
        for var in _OVERRIDE_VARS:
            os.environ.pop(var, None)

        config = _reload_config()

        assert config.JUDGE_MODEL == EXPECTED_JUDGE_MODEL
        assert config.GROQ_MODEL == EXPECTED_GROQ_MODEL
        # The judge must stay decoupled from the generator (different family, Req 3.1).
        assert config.JUDGE_MODEL != config.GROQ_MODEL
        # Judge provider default is unchanged.
        assert config.JUDGE_PROVIDER == EXPECTED_JUDGE_PROVIDER
    finally:
        dotenv.load_dotenv = saved_load_dotenv
        os.environ.clear()
        os.environ.update(saved_environ)
        _reload_config()  # leave the module reflecting the real environment


# ─────────────────────────────────────────────────────────────────────────────
# 2. Static path — read the DEFAULT literals straight from the source.
# ─────────────────────────────────────────────────────────────────────────────
def test_config_source_defaults_are_correct():
    """The ``os.getenv(..., DEFAULT)`` literals in configs/config.py are correct.

    Environment-independent and importless. Requirements: 2.1, 3.1.
    """
    defaults = _getenv_defaults(CONFIG_PATH)

    assert defaults.get("JUDGE_MODEL") == EXPECTED_JUDGE_MODEL
    assert defaults.get("GROQ_MODEL") == EXPECTED_GROQ_MODEL
    assert defaults.get("JUDGE_PROVIDER") == EXPECTED_JUDGE_PROVIDER
    assert defaults.get("JUDGE_MODEL") != defaults.get("GROQ_MODEL")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Committed template — .env.example carries the same judge defaults.
# ─────────────────────────────────────────────────────────────────────────────
def test_env_example_declares_judge_provider_and_model():
    """``.env.example`` contains ``JUDGE_PROVIDER=groq`` and ``JUDGE_MODEL=openai/gpt-oss-20b``.

    Requirement 2.1 asks for the default to be set consistently in both
    ``configs/config.py`` and the committed ``.env.example``.
    """
    assert os.path.isfile(ENV_EXAMPLE_PATH), (
        ".env.example is missing at project root: %s" % ENV_EXAMPLE_PATH
    )
    env = _parse_env_file(ENV_EXAMPLE_PATH)

    assert env.get("JUDGE_PROVIDER") == EXPECTED_JUDGE_PROVIDER
    assert env.get("JUDGE_MODEL") == EXPECTED_JUDGE_MODEL
    # Generator line stays as the Llama-family instant model (kept different family).
    assert env.get("GROQ_MODEL") == EXPECTED_GROQ_MODEL
    assert env.get("JUDGE_MODEL") != env.get("GROQ_MODEL")


def test_env_example_and_config_defaults_agree():
    """The judge defaults are identical across the source module and the template."""
    source_defaults = _getenv_defaults(CONFIG_PATH)
    env = _parse_env_file(ENV_EXAMPLE_PATH)

    for key in ("JUDGE_PROVIDER", "JUDGE_MODEL", "GROQ_MODEL"):
        assert source_defaults.get(key) == env.get(key), (
            "default for %s differs between configs/config.py (%r) and .env.example (%r)"
            % (key, source_defaults.get(key), env.get(key))
        )
