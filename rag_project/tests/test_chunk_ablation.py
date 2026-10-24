"""Static + smoke tests for ``chunk_ablation_experiment.py`` dataset discovery (BUG 5b).

Spec: eval-pipeline-fixes, Task 8.2 (Requirements 2.6, 3.7).

Task 8.1 changed the script to source the synthetic-dataset location from
``configs.config.SYNTHETIC_DATA_PATH`` instead of the hard-coded, never-written
``outputs/synthetic_data.json``. This module verifies that fix at two levels
*without* requiring the LangChain/Chroma/scientific stack (absent on the Python
3.9 analysis machine):

1. STATIC checks (always run — pure text/AST, stdlib only). Assert the script
   imports ``SYNTHETIC_DATA_PATH`` from ``configs.config``, that the existence
   check and the ``open(...)`` both reference that name (not a literal), and that
   the stale ``outputs/synthetic_data.json`` literal is gone (Req 2.6).

2. BEHAVIORAL existence-check test (always run — hermetic). Compile ONLY the top
   of the REAL script (every top-level statement up to and including the
   ``with open(SYNTHETIC_DATA_PATH) ... json.load`` block) and execute it with
   the heavy top-of-script imports replaced by lightweight ``sys.modules`` stubs
   and ``configs.config.SYNTHETIC_DATA_PATH`` monkeypatched to a temp path. This
   exercises the actual, fixed check/load statements (not a re-implementation)
   and asserts:
     (a) with a dataset present at that path, the script proceeds PAST the
         existence check and reads the file (Req 2.6);
     (b) with it absent, the script exits with ``SystemExit(1)`` (guards the fix
         / Req 3.7 precondition).

The full chunk-size ablation run (the divergence CSV + ``outputs/plots/
divergence_proof.png``) genuinely needs the heavy stack and builds vectorstores,
so it is **(deferred)** to the Python 3.12 runtime — see ``test_full_ablation_run``
below, which is skipped everywhere by default and never required for this file to
pass.
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest

# Absolute path to the script under test: <project root>/chunk_ablation_experiment.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "chunk_ablation_experiment.py")

# The stale, never-written path the fix removed.
STALE_LITERAL = "outputs/synthetic_data.json"

# The config symbol the fixed script must source the dataset location from.
CONFIG_MODULE = "configs.config"
CONFIG_PATH_NAME = "SYNTHETIC_DATA_PATH"


def _read_script_source() -> str:
    with open(SCRIPT_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


# ─────────────────────────────────────────────────────────────────────────────
# 1. STATIC checks — no third-party deps required.
# ─────────────────────────────────────────────────────────────────────────────
class TestStaticDatasetPathWiring:
    """Assert the dataset path comes from config, not a hard-coded literal."""

    def test_imports_synthetic_data_path_from_config(self):
        """``from configs.config import SYNTHETIC_DATA_PATH`` is present."""
        tree = ast.parse(_read_script_source())
        found = any(
            isinstance(node, ast.ImportFrom)
            and node.module == CONFIG_MODULE
            and any(alias.name == CONFIG_PATH_NAME for alias in node.names)
            for node in ast.walk(tree)
        )
        assert found, f"expected `from {CONFIG_MODULE} import {CONFIG_PATH_NAME}`"

    def test_existence_check_uses_config_path_name(self):
        """``os.path.exists(...)`` is called on the ``SYNTHETIC_DATA_PATH`` name."""
        tree = ast.parse(_read_script_source())

        def _is_os_path_exists(call: ast.Call) -> bool:
            f = call.func
            return (
                isinstance(f, ast.Attribute)
                and f.attr == "exists"
                and isinstance(f.value, ast.Attribute)
                and f.value.attr == "path"
                and isinstance(f.value.value, ast.Name)
                and f.value.value.id == "os"
            )

        exists_calls = [
            n for n in ast.walk(tree) if isinstance(n, ast.Call) and _is_os_path_exists(n)
        ]
        assert exists_calls, "expected an os.path.exists(...) call in the script"
        assert any(
            call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == CONFIG_PATH_NAME
            for call in exists_calls
        ), f"os.path.exists must be called on {CONFIG_PATH_NAME}, not a literal"

    def test_open_reads_config_path_name(self):
        """``open(...)`` reads the ``SYNTHETIC_DATA_PATH`` name (not a literal)."""
        tree = ast.parse(_read_script_source())
        open_calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "open"
        ]
        assert open_calls, "expected an open(...) call in the script"
        assert any(
            call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == CONFIG_PATH_NAME
            for call in open_calls
        ), f"open(...) must read {CONFIG_PATH_NAME}, not a hard-coded path"

    def test_stale_hardcoded_literal_is_gone(self):
        """The never-written ``outputs/synthetic_data.json`` literal must be absent."""
        source = _read_script_source()
        assert STALE_LITERAL not in source, (
            f"stale hard-coded path {STALE_LITERAL!r} must no longer appear"
        )

    def test_no_stale_synthetic_path_variable(self):
        """The old ``SYNTHETIC_PATH = "..."`` assignment is gone (uses config now)."""
        tree = ast.parse(_read_script_source())
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "SYNTHETIC_PATH" not in assigned, (
            "stale `SYNTHETIC_PATH` variable should be replaced by SYNTHETIC_DATA_PATH"
        )

    def test_ablation_uses_the_ms_marco_corpus(self):
        """Synthetic MS MARCO questions must not be evaluated against SQuAD text."""
        source = _read_script_source()
        assert "load_msmarco_documents" in source
        assert "load_squad_documents" not in source


# ─────────────────────────────────────────────────────────────────────────────
# 2. BEHAVIORAL existence-check test — hermetic via lightweight import stubs.
# ─────────────────────────────────────────────────────────────────────────────
class _AnyAttrModule(types.ModuleType):
    """Stub module whose every (non-dunder) attribute access yields a throwaway
    dummy, so ``from <mod> import <anything>`` succeeds without the real
    dependency. Dunder lookups raise ``AttributeError`` so the import machinery's
    protocol probes (``__path__``, ``__spec__``, ...) behave normally.
    """

    def __getattr__(self, name):  # noqa: D401 - see class docstring
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _DUMMY


class _Dummy:
    """Cheap stand-in for any imported class/function; never actually invoked."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self


_DUMMY = _Dummy()

# Every heavy / irrelevant top-of-script import. Parent packages are listed
# before their submodules and are stubbed too, because module2/module3's real
# ``__init__`` eagerly import the heavy submodules (and would fail / pull the
# full stack on import).
_HEAVY_MODULE_NAMES = (
    "numpy",
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "module1",
    "module1.data_preprocessing",
    "module2",
    "module2.rag_architectures",
    "module3",
    "module3.evaluation",
    "langchain_chroma",
)


@contextlib.contextmanager
def _stubbed_heavy_imports():
    """Install lightweight stand-ins for the heavy top-of-script imports so the
    REAL existence-check / data-load statements can execute on any runtime
    without the LangChain/Chroma/scientific stack. Originals are restored on
    exit. ``dotenv`` is stubbed only when the real package is unavailable, since
    ``configs.config`` (imported for real, to prove the path is sourced from it)
    calls ``load_dotenv`` at import time.
    """
    saved = {name: sys.modules.get(name) for name in _HEAVY_MODULE_NAMES}
    saved_dotenv = sys.modules.get("dotenv")
    try:
        if importlib.util.find_spec("dotenv") is None:
            dotenv_stub = types.ModuleType("dotenv")
            dotenv_stub.load_dotenv = lambda *a, **k: False
            sys.modules["dotenv"] = dotenv_stub

        for name in _HEAVY_MODULE_NAMES:
            stub = _AnyAttrModule(name)
            if "." not in name:
                stub.__path__ = []  # present a package-like shape
            sys.modules[name] = stub

        # Link parent → child so dotted imports resolve against the stubs.
        for name in _HEAVY_MODULE_NAMES:
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(sys.modules[parent], child, sys.modules[name])

        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        if saved_dotenv is None:
            sys.modules.pop("dotenv", None)
        else:
            sys.modules["dotenv"] = saved_dotenv


def _compile_top_slice_through_data_load():
    """Compile ONLY the top of the real script: every top-level statement up to
    and including the first ``with`` block — the
    ``with open(SYNTHETIC_DATA_PATH) as f: data = json.load(f)`` that loads the
    dataset. This runs the REAL imports, existence check, and data load while
    stopping before the heavy ablation body. Returns a code object.
    """
    tree = ast.parse(_read_script_source())
    cut = next(
        (i for i, node in enumerate(tree.body) if isinstance(node, ast.With)),
        None,
    )
    assert cut is not None, "expected a top-level `with open(...)` block in the script"
    sliced = ast.Module(body=tree.body[: cut + 1], type_ignores=[])
    ast.fix_missing_locations(sliced)
    return compile(sliced, SCRIPT_PATH, "exec")


class TestDatasetDiscoveryBehavior:
    """Exercise the fixed existence-check/load against a temp SYNTHETIC_DATA_PATH."""

    def test_proceeds_past_check_when_dataset_present(self, tmp_path, monkeypatch):
        """(Req 2.6) With a dataset at the configured path, the script locates it
        and proceeds past the existence check to read it."""
        dataset = tmp_path / "synthetic_dataset.json"
        fixture = [
            {"question": "What is the capital of France?", "ground_truth": "Paris"},
            {"question": "Who wrote Hamlet?", "ground_truth": "Shakespeare"},
        ]
        dataset.write_text(json.dumps(fixture), encoding="utf-8")

        code = _compile_top_slice_through_data_load()
        with _stubbed_heavy_imports():
            import configs.config as config  # real module; dotenv stubbed if absent

            monkeypatch.setattr(config, CONFIG_PATH_NAME, str(dataset))
            namespace = {"__name__": "chunk_ablation_slice", "__file__": SCRIPT_PATH}
            # Must NOT raise SystemExit: the check passes and the load proceeds.
            exec(code, namespace)

        # Proof it proceeded PAST the existence check AND read from the config path.
        assert namespace["data"] == fixture

    def test_exits_when_dataset_absent(self, tmp_path, monkeypatch):
        """(Req 3.7 precondition) With no file at the configured path, the script
        exits cleanly with code 1 before attempting to read it."""
        missing = tmp_path / "synthetic_dataset.json"
        assert not missing.exists()

        code = _compile_top_slice_through_data_load()
        with _stubbed_heavy_imports():
            import configs.config as config

            monkeypatch.setattr(config, CONFIG_PATH_NAME, str(missing))
            namespace = {"__name__": "chunk_ablation_slice", "__file__": SCRIPT_PATH}
            with pytest.raises(SystemExit) as excinfo:
                exec(code, namespace)

        assert excinfo.value.code == 1
        # The file was never opened, so no `data` was bound.
        assert "data" not in namespace


# ─────────────────────────────────────────────────────────────────────────────
# 3. (deferred) Full ablation run — needs the Python 3.12 LangChain/Chroma stack.
# ─────────────────────────────────────────────────────────────────────────────
def _heavy_stack_importable() -> bool:
    return all(
        importlib.util.find_spec(mod) is not None
        for mod in ("numpy", "pandas", "matplotlib", "langchain_chroma")
    )


@pytest.mark.skipif(
    not (_heavy_stack_importable() and os.getenv("RUN_DEFERRED_ABLATION") == "1"),
    reason=(
        "Full chunk-size ablation (writes outputs/chunk_ablation_results.csv and "
        "outputs/plots/divergence_proof.png) is deferred to the Python 3.12 runtime "
        "with the LangChain/Chroma stack and a synthetic dataset present at "
        "SYNTHETIC_DATA_PATH. Set RUN_DEFERRED_ABLATION=1 on that runtime to run it."
    ),
)
def test_full_ablation_run(tmp_path):
    """(deferred) Run the real script end-to-end and confirm the divergence CSV
    and ``divergence_proof.png`` are produced. Skipped everywhere by default;
    exercised only on the provisioned 3.12 runtime."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"  # headless matplotlib
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, (
        f"script exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert os.path.exists(
        os.path.join(PROJECT_ROOT, "outputs", "chunk_ablation_results.csv")
    )
    assert os.path.exists(
        os.path.join(PROJECT_ROOT, "outputs", "plots", "divergence_proof.png")
    )
