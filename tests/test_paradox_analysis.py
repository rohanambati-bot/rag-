"""Static + smoke tests for the trimmed ``paradox_analysis.py`` (BUG 5a).

Spec: eval-pipeline-fixes, Task 7.2 (Requirements 2.5, 3.6).

Two independent parts:

1. STATIC checks (always run — pure text/AST, no third-party deps): assert the
   dead summary re-read block and the duplicated ``pandas``/``seaborn``/
   ``matplotlib`` import header were removed from ``paradox_analysis.py``
   (Requirement 2.5). These run on the Python 3.9 analysis machine.

2. SMOKE run (Requirement 3.6): create a tiny ``outputs/raw_eval_results.csv``
   fixture, run the script as a subprocess from a temp working directory, and
   assert exit 0 with an "Overall paradox rate: X%" in ``[0, 100]``. The script
   imports pandas/seaborn/matplotlib; when those are unavailable (the 3.9
   analysis machine) the smoke run is **skipped / deferred to the 3.12
   runtime**. The heatmap PNG assertion is explicitly deferred either way and is
   never required for this test to pass.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys

import pytest

# Absolute path to the script under test: <project root>/paradox_analysis.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "paradox_analysis.py")


def _read_script_source() -> str:
    with open(SCRIPT_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def _deps_available() -> bool:
    """True only when every third-party module the script imports is importable.

    On the Python 3.9 analysis machine these are absent, so the smoke run is
    skipped (deferred to the 3.12 runtime); the static checks still run.
    """
    return all(
        importlib.util.find_spec(mod) is not None
        for mod in ("pandas", "seaborn", "matplotlib")
    )


DEPS_AVAILABLE = _deps_available()


class TestStaticTrim:
    """Pure text / AST assertions — no third-party deps required."""

    def test_no_hardcoded_summary_reread(self):
        """The dead ``pd.read_csv("outputs/eval_results.csv")`` re-read is gone.

        The only permitted read of the summary path is via the ``OLD_PATH``
        fallback variable, i.e. ``pd.read_csv(OLD_PATH)``.
        """
        source = _read_script_source()
        assert 'pd.read_csv("outputs/eval_results.csv")' not in source
        assert "pd.read_csv('outputs/eval_results.csv')" not in source

    def test_read_csv_calls_use_path_variables_not_literals(self):
        """Every ``read_csv`` call reads a path *variable*, never a hard-coded
        summary literal (guards against re-introducing the stale re-read)."""
        source = _read_script_source()
        read_csv_args = re.findall(r"read_csv\(\s*([^)]*?)\s*\)", source)
        assert read_csv_args, "expected at least one read_csv call in the script"
        for arg in read_csv_args:
            assert "eval_results.csv" not in arg, (
                f"read_csv should use a path variable, got literal: {arg!r}"
            )

    def test_third_party_imports_appear_exactly_once(self):
        """No duplicate import header: pandas/seaborn/matplotlib each imported once."""
        tree = ast.parse(_read_script_source())
        counts = {"pandas": 0, "seaborn": 0, "matplotlib": 0}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in counts:
                        counts[root] += 1
        for mod, n in counts.items():
            assert n == 1, f"expected exactly one `import {mod}`, found {n}"

    def test_single_pivot_table_no_dead_eof_block(self):
        """The trimmed script pivots exactly once (inside the guarded heatmap
        block); the dead EOF block re-computed a second pivot on absent columns."""
        source = _read_script_source()
        assert source.count("pivot_table") == 1

    def test_hit_rate_and_query_type_only_derived_or_guarded(self):
        """``hit_rate`` / ``query_type`` are created defensively, not assumed.

        The trimmed script derives both (``df["hit_rate"] = ...`` and
        ``df["query_type"] = ...``) rather than reading them off a summary CSV
        that lacks them. Assert each column is assigned before use.
        """
        source = _read_script_source()
        assert 'df["hit_rate"] =' in source or "df['hit_rate'] =" in source
        assert 'df["query_type"] =' in source or "df['query_type'] =" in source

    def test_script_defines_raw_and_old_path_variables(self):
        """The defensive top half (kept) assigns both RAW_PATH and OLD_PATH."""
        tree = ast.parse(_read_script_source())  # also asserts valid syntax
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "RAW_PATH" in assigned
        assert "OLD_PATH" in assigned


# A handful of rows with a mix of paradox / non-paradox samples and a
# ``query_type`` column so the heatmap pivot path is exercised.
FIXTURE_CSV = """architecture,query_type,precision_omega
naive,factual,0.9
naive,multi_hop,0.2
crag,factual,0.5
crag,multi_hop,0.1
rag_fusion,factual,0.8
rag_fusion,multi_hop,0.35
"""


@pytest.mark.skipif(
    not DEPS_AVAILABLE,
    reason=(
        "pandas/seaborn/matplotlib unavailable (Python 3.9 analysis machine); "
        "smoke run deferred to the 3.12 runtime"
    ),
)
class TestSmokeRun:
    """Run the trimmed script end-to-end against a tiny fixture."""

    def test_exit_zero_and_paradox_rate_in_range(self, tmp_path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "raw_eval_results.csv").write_text(FIXTURE_CSV, encoding="utf-8")

        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"  # headless matplotlib for the heatmap write

        proc = subprocess.run(
            [sys.executable, SCRIPT_PATH],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert proc.returncode == 0, (
            f"script exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

        match = re.search(
            r"Overall paradox rate:\s*([0-9]+(?:\.[0-9]+)?)\s*%", proc.stdout
        )
        assert match is not None, (
            f"'Overall paradox rate: X%' not found in stdout:\n{proc.stdout}"
        )
        rate = float(match.group(1))
        assert 0.0 <= rate <= 100.0, f"paradox rate {rate} outside [0, 100]"
        # NOTE: heatmap PNG existence is deferred to the 3.12 runtime — not asserted.
