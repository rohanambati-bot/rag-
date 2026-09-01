"""Property + example tests for `.env` fill-only-missing precedence (BUG 6).

Feature: eval-pipeline-fixes, Property 3: Environment loading fills only missing keys

Property statement (design.md → Correctness Properties → Property 3):
    For any set of key/value pairs, after loading the project `.env` file, every key
    already present in the process environment retains its original (shell-provided)
    value, and every key absent from the environment but present in `.env` resolves to
    its `.env` value.

Validates: Requirements 2.7, 3.9

These tests only exercise `python-dotenv`'s `load_dotenv(override=False)` semantics — the
exact call `configs/config.py` makes at import time — so they need no LangChain/Chroma/LLM
stack and no network access. Every example snapshots and restores `os.environ` and writes
the `.env` into a throwaway temp directory, so nothing leaks between examples.
"""
from __future__ import annotations

import os
import string
import tempfile

from dotenv import load_dotenv
from hypothesis import given, settings
from hypothesis import strategies as st

# ── Distinct, collision-proof prefixes ───────────────────────────────────────────
# Generated keys are namespaced under these prefixes so they can never clobber a real
# environment variable, and the three groups stay mutually disjoint by construction
# (the bytes right after "TEST_ENV_" differ: SHELL_ / ENVFILE_ / BOTH_).
_PREFIX_SHELL = "TEST_ENV_SHELL_"      # present only in the shell (os.environ)
_PREFIX_ENVFILE = "TEST_ENV_ENVFILE_"  # present only in the .env file
_PREFIX_BOTH = "TEST_ENV_BOTH_"        # present in BOTH, with different values

# Env-var key suffix: uppercase ASCII letters, digits, underscore. Prefixed above, so the
# full key always starts with a letter and is a valid environment-variable name.
_KEY_SUFFIX = st.text(
    alphabet=string.ascii_uppercase + string.digits + "_", min_size=1, max_size=12
)

# Env value: non-empty and drawn from an alphabet with no newline / '=' / '#' / '$' /
# quote / backslash / whitespace, so python-dotenv parses it back literally (no comment
# stripping, no POSIX interpolation, no quote handling). Covers real-world shapes like
# "openai/gpt-oss-20b" and "llama-3.1-8b-instant".
_VALUE = st.text(
    alphabet=string.ascii_letters + string.digits + "-_./", min_size=1, max_size=20
)


@st.composite
def _overlap_dict(draw):
    """Keys present in both the shell and .env, with guaranteed-different values.

    Returns {suffix: (shell_value, env_value)} where shell_value != env_value so the
    override=False assertion (shell wins) is meaningful rather than vacuous.
    """
    raw = draw(st.dictionaries(_KEY_SUFFIX, st.tuples(_VALUE, _VALUE), max_size=5))
    out = {}
    for suffix, (shell_v, env_v) in raw.items():
        if shell_v == env_v:
            env_v = env_v + "_ENVDIFF"  # force a difference; suffix uses only safe chars
        out[suffix] = (shell_v, env_v)
    return out


def _write_env_file(directory: str, pairs: dict) -> str:
    """Write KEY=VALUE lines to `<directory>/.env` and return its path."""
    path = os.path.join(directory, ".env")
    lines = [f"{k}={v}" for k, v in pairs.items()]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")
    return path


@settings(max_examples=200, deadline=None)
@given(
    shell_only=st.dictionaries(_KEY_SUFFIX, _VALUE, max_size=5),
    env_only=st.dictionaries(_KEY_SUFFIX, _VALUE, max_size=5),
    overlap=_overlap_dict(),
)
def test_load_dotenv_fills_only_missing_keys(shell_only, env_only, overlap):
    """Feature: eval-pipeline-fixes, Property 3: Environment loading fills only missing keys.

    Validates: Requirements 2.7, 3.9
    """
    # Namespace each group so keys are unique and disjoint across groups.
    shell_only = {_PREFIX_SHELL + k: v for k, v in shell_only.items()}
    env_only = {_PREFIX_ENVFILE + k: v for k, v in env_only.items()}
    overlap = {_PREFIX_BOTH + k: v for k, v in overlap.items()}

    saved_environ = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Seed the shell-present keys into os.environ (shell-only + overlap).
            for k, v in shell_only.items():
                os.environ[k] = v
            for k, (shell_v, _env_v) in overlap.items():
                os.environ[k] = shell_v
            # Ensure .env-only keys are genuinely absent before the load.
            for k in env_only:
                os.environ.pop(k, None)

            # 2. Write the temp .env: the .env-only keys plus the .env side of overlaps.
            env_file_contents = dict(env_only)
            for k, (_shell_v, env_v) in overlap.items():
                env_file_contents[k] = env_v
            env_path = _write_env_file(tmpdir, env_file_contents)

            # 3. Load exactly as configs/config.py does: fill-only-missing.
            load_dotenv(env_path, override=False)

            # 4a. Shell-only keys keep their original shell value (untouched by load).
            for k, v in shell_only.items():
                assert os.environ.get(k) == v

            # 4b. Overlap keys keep the SHELL value — shell wins under override=False.
            for k, (shell_v, _env_v) in overlap.items():
                assert os.environ.get(k) == shell_v

            # 4c. .env-only keys resolve to their .env value (the missing keys get filled).
            for k, v in env_only.items():
                assert os.environ.get(k) == v
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)


# ── Concrete examples (complement the property with named edge cases) ─────────────

def test_env_only_key_is_filled_from_dotenv():
    """A key absent from the shell is populated from .env."""
    key = "TEST_ENV_EXAMPLE_ONLY_IN_FILE"
    saved = dict(os.environ)
    try:
        os.environ.pop(key, None)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_env_file(tmpdir, {key: "from-dotenv"})
            load_dotenv(path, override=False)
            assert os.environ.get(key) == "from-dotenv"
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_shell_value_wins_over_dotenv():
    """A key already exported in the shell keeps its shell value (Req 3.9)."""
    key = "TEST_ENV_EXAMPLE_OVERLAP"
    saved = dict(os.environ)
    try:
        os.environ[key] = "from-shell"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_env_file(tmpdir, {key: "from-dotenv"})
            load_dotenv(path, override=False)
            assert os.environ.get(key) == "from-shell"
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_absent_env_file_is_a_noop():
    """Pointing load_dotenv at a missing file is a silent no-op (adds nothing)."""
    key = "TEST_ENV_EXAMPLE_MISSING_FILE"
    saved = dict(os.environ)
    try:
        os.environ.pop(key, None)
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, ".env")  # never created
            result = load_dotenv(missing, override=False)
            assert result is False
            assert key not in os.environ
    finally:
        os.environ.clear()
        os.environ.update(saved)
