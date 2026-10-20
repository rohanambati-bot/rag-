"""AST inventory of index opens, consumers, mutations and filesystem lifecycle.

Provider-free/std-library only. Run with --output outputs/security-v4-index-inventory.json
for a create-only receipt. This is a static review aid, not a proof of arbitrary
Python dataflow or protection against external/nonparticipating Chroma clients.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_FILES = {
    "module1/data_preprocessing.py", "preprocess_full.py", "main.py", "query.py",
    "preprocess_train.py", "train_in_shards.py", "chunk_ablation_experiment.py",
    "serving.py", "scripts/validate_live_lifecycle.py", "scripts/validate_research.py",
    "scripts/validate_serving.py", "scripts/run_retrieval_ablation_local.py",
    "scripts/optimize_retrieval.py",
}
HELPERS = {
    "load_vector_store": "open-reader", "build_vector_store": "first-build",
    "rebuild_vector_store_safely": "build-publish-reopen",
    "persistent_index_writer": "open-writer", "copy_vector_store": "snapshot",
    "run_isolated_index_builder": "isolated-build",
    "index_publication_guard": "reader-barrier", "build_lock_for": "builder-lock",
    "_publish_vector_store": "publish", "_publish_vector_store_locked": "publish",
    "_publish_vector_store_without_readers": "publish-guarded-internal",
    "_promote_vector_store_without_readers": "publish-guarded-internal",
    "_recover_previous_vector_store": "recovery",
    "_recover_previous_vector_store_without_readers": "recovery-guarded-internal",
    "cleanup_stale_staging_dirs": "clear-private-stage",
    "_cleanup_staged_vector_stores": "clear-private-stage",
}
RAW_OWNERS = {
    ("module1/data_preprocessing.py", "persistent_index_writer"),
    ("module1/data_preprocessing.py", "load_vector_store"),
    ("module1/data_preprocessing.py", "rebuild_vector_store_safely"),
}


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + "." + node.attr
    return "<dynamic>"


class Inventory(ast.NodeVisitor):
    def __init__(self, path):
        self.path, self.scope, self.aliases, self.rows = path, [], {}, []

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def visit_Import(self, node):
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = alias.name

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        name = _name(node.func)
        first, *rest = name.split(".")
        resolved = ".".join([self.aliases.get(first, first), *rest])
        leaf = resolved.rsplit(".", 1)[-1]
        operation = HELPERS.get(leaf)
        boundary = "participating" if self.path in SUPPORTED_FILES else "review-caller"
        raw = ("Chroma" in resolved.split(".") and (
            leaf == "Chroma" or leaf in {"from_documents", "from_texts"}
        )) or leaf in {"PersistentClient", "EphemeralClient", "Client"} and "chromadb" in resolved
        if raw:
            keys = {keyword.arg for keyword in node.keywords}
            persistent = "persist_directory" in keys or leaf == "PersistentClient"
            dynamic = None in keys or "client" in keys or leaf == "Client"
            operation = "raw-persistent-open" if persistent else "raw-dynamic-open" if dynamic else "ephemeral-open"
            if (self.path, self.scope[-1] if self.scope else "") in RAW_OWNERS:
                boundary = "guarded-shared-implementation"
            elif self.path.startswith("tests/"):
                boundary = "test-fixture"
            elif operation == "ephemeral-open":
                boundary = "ephemeral-no-persistent-path"
            else:
                boundary = "cross-scope-review" if self.path.startswith(("module2/", "module5/")) else "UNGUARDED"
        elif leaf in {"add_documents", "add_texts", "delete_collection", "reset_collection"}:
            operation = "mutate-or-clear"
        elif leaf in {"get", "add", "delete", "query", "similarity_search", "similarity_search_with_score", "as_retriever"} and any(
            token in name for token in ("store", "collection", "shard_vs", "final_vs")
        ):
            operation = "mutate-or-clear" if leaf in {"add", "delete"} else "read-consumer"
        elif resolved in {"os.replace", "shutil.rmtree", "shutil.copytree"} and self.path in SUPPORTED_FILES:
            operation = "filesystem-rename-rollback" if leaf == "replace" else "filesystem-clear" if leaf == "rmtree" else "filesystem-snapshot"
        if operation:
            self.rows.append({
                "path": self.path, "line": node.lineno, "scope": ".".join(self.scope),
                "call": resolved, "operation": operation, "boundary": boundary,
            })
        self.generic_visit(node)


def build_inventory(root=ROOT):
    root = Path(root)
    rows, hashes = [], {}
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if not name.startswith(".") and name not in {
            "__pycache__", "data", "outputs", "node_modules",
        })
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            path = Path(directory) / filename
            relative = path.relative_to(root).as_posix()
            source = path.read_bytes()
            hashes[relative] = hashlib.sha256(source).hexdigest()
            visitor = Inventory(relative)
            visitor.visit(ast.parse(source, filename=relative))
            rows.extend(visitor.rows)
    gaps = [row for row in rows if row["boundary"] == "UNGUARDED"]
    return {
        "schema": "index-lifecycle-inventory/v1", "files_sha256": hashes,
        "supported_entrypoints": sorted(SUPPORTED_FILES), "entries": rows,
        "unguarded_raw_opens": gaps,
        "restrictions": [
            "Participating POSIX/local-filesystem lifetime leases; stop-serving rebuild, no hot swap.",
            "Staging native writers exit (spawn/join) before publication; cached paths cannot be rebuilt in-process.",
            "External clients, network filesystems, dynamic imports and arbitrary Python alias/dataflow are outside this AST proof.",
            "module2 RAPTOR constructors are ephemeral in this tree; module5 injected clients are a separate uploader boundary.",
            "Legacy partial train/shard indexes are preserved, not resumed in place; embedding restarts in a new stage.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_inventory()
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered)
    else:
        print(rendered, end="")
    return bool(report["unguarded_raw_opens"])


if __name__ == "__main__":
    raise SystemExit(main())
