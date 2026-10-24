"""Regression smoke test for version-control hygiene of environment files.

Covers requirement 3.10 of the ``eval-pipeline-fixes`` spec: the repository must
continue to exclude real environment files (``.env`` and local ``.env.*``
variants) from version control while keeping the ``.env.example`` template
tracked, so that no secrets are ever committed.

This is a pure repository-configuration check. It depends only on the standard
library and pytest (no LangChain/Chroma/LLM stack), so it runs on Python 3.9.

Two complementary strategies are used:

* Text-based checks (always run) parse ``.gitignore`` and assert the expected
  ignore rules are present and that the template is not literally excluded.
* ``git check-ignore`` checks (skipped when git is unavailable) authoritatively
  confirm git's actual behavior: ``.env`` and local variants are ignored while
  ``.env.example`` is tracked.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

# Project root is the parent of this ``tests/`` directory. Resolving from
# ``__file__`` keeps the test independent of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GITIGNORE_PATH = os.path.join(PROJECT_ROOT, ".gitignore")

# Env files that MUST be ignored (kept out of version control).
IGNORED_ENV_FILES = [
    ".env",
    ".env.local",
    ".env.production.local",
    ".env.staging.local",
]

# The committed template that MUST stay tracked.
TRACKED_ENV_FILE = ".env.example"

# The ignore rules the design guarantees are present in ``.gitignore``.
REQUIRED_IGNORE_RULES = [".env", ".env.local", ".env.*.local"]


def _gitignore_rules():
    """Return the effective (non-blank, non-comment) rules from ``.gitignore``."""
    with open(GITIGNORE_PATH, encoding="utf-8") as handle:
        rules = []
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rules.append(stripped)
    return rules


def _git_check_ignore_available():
    """True when git is installed and PROJECT_ROOT is inside a work tree."""
    if shutil.which("git") is None:
        return False
    try:
        result = subprocess.run(
            ["git", "-C", PROJECT_ROOT, "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def _is_ignored(name):
    """Return whether git would ignore ``name`` relative to PROJECT_ROOT.

    Uses ``git check-ignore -q`` whose exit status is: 0 = ignored,
    1 = not ignored, other = error.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", name],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            "git check-ignore failed for %r (exit %d)" % (name, result.returncode)
        )
    return result.returncode == 0


GIT_AVAILABLE = _git_check_ignore_available()
_needs_git = pytest.mark.skipif(
    not GIT_AVAILABLE,
    reason="git is not available or the project is not a git work tree",
)


class TestGitignoreText:
    """Text-based assertions over the raw ``.gitignore`` rules (always run)."""

    def test_gitignore_file_exists(self):
        assert os.path.isfile(GITIGNORE_PATH), (
            ".gitignore is missing at project root: %s" % GITIGNORE_PATH
        )

    @pytest.mark.parametrize("rule", REQUIRED_IGNORE_RULES)
    def test_required_ignore_rule_present(self, rule):
        rules = _gitignore_rules()
        assert rule in rules, (
            "expected ignore rule %r to be present in .gitignore; found: %r"
            % (rule, rules)
        )

    def test_template_not_literally_ignored(self):
        # Guards the most likely regression: adding the template as an ignore
        # rule, or introducing an over-broad glob that would sweep it in.
        rules = _gitignore_rules()
        assert TRACKED_ENV_FILE not in rules, (
            "%s must not be listed as an ignore rule" % TRACKED_ENV_FILE
        )
        for over_broad in (".env*", ".env.*", "*.example"):
            assert over_broad not in rules, (
                "over-broad rule %r would also ignore %s"
                % (over_broad, TRACKED_ENV_FILE)
            )


@_needs_git
class TestGitCheckIgnore:
    """Authoritative checks of git's actual ignore behavior."""

    @pytest.mark.parametrize("name", IGNORED_ENV_FILES)
    def test_env_variants_are_ignored(self, name):
        assert _is_ignored(name), "%s should be ignored by git but is not" % name

    def test_env_example_is_tracked(self):
        assert not _is_ignored(TRACKED_ENV_FILE), (
            "%s must remain tracked (must NOT be ignored)" % TRACKED_ENV_FILE
        )
